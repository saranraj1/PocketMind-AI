import os
from pocketmind.config import PocketMindConfig

def test_debug_config_loads():
    config_path = "configs/debug.yaml"
    assert os.path.exists(config_path)
    config = PocketMindConfig.from_yaml(config_path)
    assert config.vocab_size == 256
    assert config.context_length == 128
    assert config.d_model == 128
    assert config.n_layers == 4
    assert config.n_heads == 4
    assert config.ffn_dim == 384
    assert config.dropout == 0.0
    assert config.attention_window == 64
    assert config.recurrent_state_dim == 128
    assert config.tie_embeddings is True
    assert config.norm == "rmsnorm"
    assert config.activation == "silu"
    assert config.position_encoding == "rope"
