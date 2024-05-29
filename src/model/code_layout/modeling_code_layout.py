
import torch
import torch.nn as nn
import math

# from modules.task_heads import PretrainMLMHead

import logging

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithNoAttention, \
    TokenClassifierOutput, MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel

from src.model.globalpointer import GlobalPointer
from src.model.multilabel_categorical_ce import globalpointer_loss
from .configuration_code_layout import CodeLayoutConfig
from ...modules.utils import BaseModelOutputWithRel2DPos

logger = logging.getLogger(__name__)


def relative_position_bucket(relative_position: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32,
                             max_distance: int = 128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for small
    absolute relative_position and larger buckets for larger absolute relative_positions. All relative positions
    >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket. This should
    allow for more graceful generalization to longer sequences than the model has been trained on.

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """

    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class LayoutLMv2SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if config.fast_qkv:
            self.qkv_linear = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def forward(
            self,
            hidden_states,
            attention_mask,
            rel_pos,
            rel_2d_pos,
    ):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        # [BSZ, NAT, L, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        attention_scores = attention_scores.float().masked_fill_(attention_mask.to(torch.bool), float("-inf"))
        attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        # if head_mask is not None:
        #     attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = (context_layer, attention_probs)
        return outputs


class LayoutLMv2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv2SelfAttention(config)
        self.output = LayoutLMv2SelfOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask,
            rel_pos,
            rel_2d_pos,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            rel_pos,
            rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # return outputs
        return (attention_output, self_outputs[1:])


class LayoutLMv2SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->LayoutLMv2
class LayoutLMv2Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->LayoutLM
class LayoutLMv2Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, spatial_position_embeddings=None):
        # todo add before mlp
        # if spatial_position_embeddings is not None:
        #     hidden_states = hidden_states + spatial_position_embeddings
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # add before layerNorm
        if spatial_position_embeddings is not None:
            hidden_states = hidden_states + spatial_position_embeddings
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutLMv2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(LayoutLMv2Embeddings, self).__init__()
        self.config = config # Added: for debug
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_x_position_embeddings + 1, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_y_position_embeddings + 1, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_h_position_embeddings + 1, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_w_position_embeddings + 1, config.shape_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def _calc_spatial_position_embeddings(self, bbox):
        assert bbox.shape[2] == 6
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 4]) # bbox[:, :, 3] - bbox[:, :, 1]
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 5]) # bbox[:, :, 2] - bbox[:, :, 0]

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings


class LayoutLMv2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
        self.output = LayoutLMv2Output(config)

    def forward(
            self,
            hidden_states,
            attention_mask,
            rel_pos,
            rel_2d_pos,
            spatial_position_embeddings=None
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            rel_pos,
            rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # layer_output = apply_chunking_to_forward(
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        # )
        layer_output = self.feed_forward_chunk(attention_output, spatial_position_embeddings)

        # outputs = (layer_output,) + outputs
        # return outputs
        return (layer_output, outputs)

    def feed_forward_chunk(self, attention_output, spatial_position_embeddings=None):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, spatial_position_embeddings)
        return layer_output


class LayoutLMv2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LayoutLMv2Layer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        self.use_2d_position_embedding_residual = config.use_2d_position_embedding_residual

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)

        self.gradient_checkpointing = False

    def _calculate_1d_position_embeddings(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = nn.functional.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _calculate_2d_position_embeddings(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)

        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = nn.functional.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = nn.functional.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
            self,
            hidden_states,
            spatial_position_embeddings,
            attention_mask,
            position_2d,
            position_1d,
            output_hidden_states=False
    ):

        all_hidden_states = () if output_hidden_states else None

        rel_pos = (
            self._calculate_1d_position_embeddings(hidden_states, position_1d)
            if self.has_relative_attention_bias
            else None
        )
        rel_2d_pos = (
            self._calculate_2d_position_embeddings(hidden_states,
                                                   position_2d) if self.has_spatial_attention_bias else None
        )

        if not self.use_2d_position_embedding_residual:
            #     hidden_states = hidden_states + spatial_position_embeddings
            spatial_position_embeddings = None

        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                rel_pos,
                rel_2d_pos,
                spatial_position_embeddings
            )
            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithRel2DPos(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            rel_2d_pos=rel_2d_pos
        )


class PretrainMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, sequence_output, labels):
        hidden_states = self.dense(sequence_output)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        mlm_logits = self.decoder(hidden_states)
        mlm_loss = self.loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        return {
            "loss": mlm_loss,
            "logits": mlm_logits
        }


class CodeLayoutPretrainModel(PreTrainedModel):
    config_class = CodeLayoutConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LayoutLMv2Encoder):
            module.gradient_checkpointing = value


class CodeLayoutModel(CodeLayoutPretrainModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = LayoutLMv2Embeddings(config)

        self.encoder = LayoutLMv2Encoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _calc_text_embeddings(self, input_ids, position_1d, position_2d):
        inputs_embeds = self.embeddings.word_embeddings(input_ids)

        position_embeddings_1d = self.embeddings.position_embeddings(position_1d)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(position_2d)

        embeddings = inputs_embeds + spatial_position_embeddings
        if self.config.enable_position_1d:
            embeddings += position_embeddings_1d
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings, spatial_position_embeddings

    def forward(
            self,
            input_ids,
            attention_mask,
            position_1d,
            position_2d,
            **kwargs
    ):
        text_input_emb, spatial_position_embeddings = self._calc_text_embeddings(
            input_ids=input_ids,
            position_1d=position_1d,
            position_2d=position_2d
        )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_outputs = self.encoder(
            text_input_emb,
            spatial_position_embeddings,
            extended_attention_mask,
            position_1d=position_1d,
            position_2d=position_2d
        )

        return BaseModelOutputWithRel2DPos(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            rel_2d_pos=encoder_outputs.rel_2d_pos
        )


class CodeLayoutForMLMPretraining(CodeLayoutPretrainModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = CodeLayoutModel(config)
        self.mlm_head = PretrainMLMHead(config)

    def forward(self,
                input_ids,
                attention_mask,
                position_1d,
                position_2d,
                labels,
                **kwargs
                ):
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_1d=position_1d,
            position_2d=position_2d,
        )
        last_hidden_state = model_output.last_hidden_state

        mlm_output = self.mlm_head(last_hidden_state, labels)

        mlm_loss = mlm_output['loss']
        mlm_logits = mlm_output['logits']

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=mlm_logits,
            hidden_states=last_hidden_state
        )


class CodeLayoutForTokenClassification(CodeLayoutPretrainModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.model = CodeLayoutModel(config)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
                input_ids,
                attention_mask,
                position_1d,
                position_2d,
                labels=None,
                **kwargs
                ):
        '''
        这里可以增加 TODO
        1、输出使用最后N层
        2、multi_sample_drop_out trick
        可增点
        '''
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_1d=position_1d,
            position_2d=position_2d,
        )

        sequence_output = model_outputs.last_hidden_state

        # sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )


class CodeLayoutForGlobalPointer(CodeLayoutPretrainModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = CodeLayoutModel(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        # self.dropout = nn.Dropout(classifier_dropout)
        self.dropouts = nn.ModuleList([nn.Dropout(classifier_dropout) for _ in range(5)])
        self.global_pointer = GlobalPointer(hidden_size=self.config.hidden_size,
                                            heads=config.num_labels,
                                            head_size=128,
                                            RoPE=False,
                                            tril_mask=False)
        self.criterion = globalpointer_loss
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
                input_ids,
                attention_mask,
                position_1d,
                position_2d,
                content_mask=None,
                grid_labels=None,
                **kwargs
                ):
        '''
        content_mask：在qa任务中，解码结果mask query部分
        '''
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_1d=position_1d,
            position_2d=position_2d,
        )

        sequence_output = model_outputs.last_hidden_state

        use_last_position_residual = True
        use_multi_dropout = False

        if use_last_position_residual:
            position_embeddings_1d = self.model.embeddings.position_embeddings(position_1d)
            spatial_position_embeddings = self.model.embeddings._calc_spatial_position_embeddings(position_2d)

            sequence_output = sequence_output + spatial_position_embeddings + position_embeddings_1d
            sequence_output = self.LayerNorm(sequence_output)
        content_mask = content_mask if content_mask is not None else attention_mask
        logits = None
        loss = None
        if use_multi_dropout:
            if grid_labels is not None:
                """
                    y_true:(batch_size, ent_type_size, seq_len, seq_len)
                    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
                """
                for i, dropout in enumerate(self.dropouts):
                    sequence_output = dropout(sequence_output)
                    if i == 0:
                        logits, _ = self.global_pointer(sequence_output, attention_mask=content_mask)
                        loss = self.criterion(grid_labels, logits)
                    else:
                        logits_i, _ = self.global_pointer(sequence_output, attention_mask=content_mask)
                        logits = logits + logits_i
                        loss_i = self.criterion(grid_labels, logits_i)
                        loss = loss + loss_i

                logits = logits / len(self.dropouts)
                loss = loss / len(self.dropouts)
            else:
                loss = None
                logits, _ = self.global_pointer(sequence_output, attention_mask=content_mask)
        else:
            logits, mask = self.global_pointer(sequence_output, attention_mask=content_mask)
            if grid_labels is not None:
                """
                    y_true:(batch_size, ent_type_size, seq_len, seq_len)
                    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
                """
                loss = self.criterion(grid_labels, logits)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )
