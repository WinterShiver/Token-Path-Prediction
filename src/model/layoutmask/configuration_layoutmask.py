from typing import Tuple
from transformers.configuration_utils import PretrainedConfig


class LayoutMaskConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 98979,
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
        use_last_layer_position_embedding_residual: bool = True,
        use_multi_dropout: bool = False,
        max_length: int = 512,
        enable_position_1d: bool = True,

        ## vision settings
        img_size: Tuple[int, int] = (512, 256),
        patch_size: int = 16,
        vision_decoder_embed_dim: int = 512,
        vision_decoder_depth: int = 1,
        vision_decoder_num_heads: int = 16,
        pad_token_id: int = 1,

        num_labels: int = 2, 
        mpm_box_num: int = 4,
        dropout_num: int =1,

        **kwargs
    ):
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
        self.use_last_layer_position_embedding_residual = use_last_layer_position_embedding_residual
        self.use_multi_dropout = use_multi_dropout
        self.max_length = max_length
        self.enable_position_1d = enable_position_1d

        self.img_size = img_size
        self.patch_size = patch_size
        self.vision_decoder_embed_dim = vision_decoder_embed_dim
        self.vision_decoder_depth = vision_decoder_depth
        self.vision_decoder_num_heads = vision_decoder_num_heads

        self.pad_token_id = pad_token_id
        self.num_labels = num_labels
        self.mpm_box_num = mpm_box_num
        self.dropout_num = dropout_num

if __name__ == '__main__':
    config = LayoutMaskConfig()
    config.save_pretrained('./')
        