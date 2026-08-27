import math

import torch
import torch.nn as nn
from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

QWEN_CONFIG = {
    "emb_dim": 1024,
    "hidden_dim": 3072,
    "n_layers": 28,
    "n_kv_heads": 8,
    "n_heads": 16,
    "head_dim": 128,
    "vocab_size": 151936,
    "context_len": 32768,
    "rope_theta": 1000000,
    "rms_norm_eps": 1e-6,
    "dtype": torch.float32
}

class FeedForward(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=cfg["dtype"])
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=cfg["dtype"])
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"])

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = torch.nn.functional.silu(x_fc1) * x_fc2

        return self.fc3(x)

def compute_rope_params(head_dim, theta_base=10000, context_length=4096, dtype=torch.float32):
    assert head_dim%2==0, "head_dim must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)

class GQAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg["emb_dim"]
        self.d_out = cfg["n_heads"] * cfg["head_dim"]
        self.W_q = nn.Linear(cfg["emb_dim"], self.d_out, bias=False, dtype=cfg["dtype"])
        self.W_k = nn.Linear(cfg["emb_dim"], cfg["n_kv_heads"]*cfg["head_dim"], bias=False, dtype=cfg["dtype"])
        self.W_v = nn.Linear(cfg["emb_dim"], cfg["n_kv_heads"] * cfg["head_dim"], bias=False, dtype=cfg["dtype"])
        self.group_size = cfg["n_heads"]//cfg["n_kv_heads"]
        self.proj = nn.Linear(self.d_out, self.d_in, bias=False, dtype=cfg["dtype"])
        # QK-norm: per-head RMSNorm on head_dim, applied after splitting into
        # heads but before RoPE - Qwen3's distinguishing feature vs Qwen2/Llama,
        # stabilizes attention logits independent of query/key projection scale.
        self.q_norm = nn.RMSNorm(cfg["head_dim"], eps=cfg["rms_norm_eps"], dtype=cfg["dtype"])
        self.k_norm = nn.RMSNorm(cfg["head_dim"], eps=cfg["rms_norm_eps"], dtype=cfg["dtype"])


    def forward(self, x, cos, sin):
        b, seq_len, emb_dim = x.shape
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        queries = queries.view(b, seq_len, self.cfg["n_heads"], self.cfg["head_dim"]).transpose(1,2)
        keys = keys.view(b, seq_len, self.cfg["n_kv_heads"], self.cfg["head_dim"]).transpose(1,2)
        values = values.view(b, seq_len, self.cfg["n_kv_heads"], self.cfg["head_dim"]).transpose(1,2)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        attn = nn.functional.scaled_dot_product_attention(queries, keys, values, is_causal=True)
        attn = attn.transpose(1,2).reshape(b, seq_len, self.d_out)

        return self.proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = GQAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])

    def forward(self, x, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, cos, sin)
        x = x+shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x

class QwenModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.trf_blks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.tok = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        self.final_norm = nn.RMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        cos, sin = compute_rope_params(cfg["head_dim"], theta_base=cfg["rope_theta"],
                                       context_length=cfg["context_len"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # Weight tying: same nn.Parameter object, not a copy - so both branches
        # of _init_weights below touch the same tensor, which is harmless
        # (both use the same N(0, 0.02^2) distribution).
        self.out.weight = self.tok.weight
        self.cfg = cfg
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # nn.RMSNorm's own default init (weight=1) is already correct here - left untouched.

        # Same residual-scaling trick as gpt2_model.py: down-weight the
        # projections that write directly into the residual stream (attention
        # out proj, FFN down proj) so activation variance doesn't compound
        # across all 28 stacked layers.
        residual_std = 0.02 / math.sqrt(2 * self.cfg["n_layers"])
        for name, param in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("ff.fc3.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def forward(self, x):
        x = self.tok(x)
        for block in self.trf_blks:
            x = block(x, self.cos, self.sin)
        x = self.final_norm(x)

        return self.out(x)

if __name__ == "__main__":
    model = QwenModel(QWEN_CONFIG)

    x = torch.randint(1, QWEN_CONFIG["vocab_size"], (2, 6))
    out = model(x)
    print(out.shape)
    print(out.dtype)

    params = sum(p.numel() for p in model.parameters())
    print(params)
