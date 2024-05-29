from typing import Tuple
from transformers.configuration_utils import PretrainedConfig


class CodeLayoutConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size: int = 250002,
            hidden_size: int = 768,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 12,
            intermediate_size: int = 3072,
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            max_position_embeddings: int = 514,
            type_vocab_size: int = 2,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            max_2d_x_position_embeddings: int = 256,
            max_2d_y_position_embeddings: int = 512,
            max_2d_w_position_embeddings: int = 256,
            max_2d_h_position_embeddings: int = 512,
            max_rel_pos: int = 128,
            rel_pos_bins: int = 32,
            fast_qkv: bool = True,
            max_rel_2d_pos: int = 256,
            rel_2d_pos_bins: int = 64,
            convert_sync_batchnorm: bool = True,
            coordinate_size: int = 128,
            shape_size: int = 128,
            has_relative_attention_bias: bool = False,
            has_spatial_attention_bias: bool = True,
            use_2d_position_embedding_residual: bool = False,

            max_text_length: int = 512,
            enable_position_1d: bool = False,
            output_hidden_states: bool = False,
            classifier_dropout: float = 0.1,
            # num_output_layer: int = 1,
            # num_multi_sample_drop: int = 1,

            ## vision settings
            img_size: Tuple[int, int] = (512, 256),
            patch_size: int = 16,
            vision_decoder_embed_dim: int = 512,
            vision_decoder_depth: int = 4,
            vision_decoder_num_heads: int = 16,
            pad_token_id: int = 1,

            tokenizer_class: str = 'XLMRobertaTokenizer',

            **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.max_2d_x_position_embeddings = max_2d_x_position_embeddings
        self.max_2d_y_position_embeddings = max_2d_y_position_embeddings
        self.max_2d_w_position_embeddings = max_2d_w_position_embeddings
        self.max_2d_h_position_embeddings = max_2d_h_position_embeddings
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fast_qkv = fast_qkv
        self.max_rel_2d_pos = max_rel_2d_pos
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.convert_sync_batchnorm = convert_sync_batchnorm
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.use_2d_position_embedding_residual = use_2d_position_embedding_residual
        self.max_text_length = max_text_length
        self.enable_position_1d = enable_position_1d
        self.output_hidden_states = output_hidden_states
        self.classifier_dropout = classifier_dropout
        # self.num_output_layer = num_output_layer
        # self.num_multi_sample_drop = num_multi_sample_drop

        self.img_size = img_size
        self.patch_size = patch_size
        self.vision_decoder_embed_dim = vision_decoder_embed_dim
        self.vision_decoder_depth = vision_decoder_depth
        self.vision_decoder_num_heads = vision_decoder_num_heads

        self.pad_token_id = pad_token_id

        self.tokenizer_class = tokenizer_class


if __name__ == '__main__':
    config = CodeLayoutConfig()
    config.save_pretrained('./')
