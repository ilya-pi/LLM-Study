import torch
import torch.nn as nn
import tiktoken
from chapter03_1 import MultiHeadAttention

GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
                *[DummyTransformerBlock(cfg)
                  for _ in range(cfg["n_layers"])]
                )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias = False
                )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
                )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
            ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
                # Implement 5 layers

                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
                ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

# A function to compute gradients
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target and output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass to calculate gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absoute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"],
                )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # Add the original input back

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class TransformerBlockSeparateDropout(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["att_drop_rate"],
                qkv_bias=cfg["qkv_bias"],
                )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["shortcut_drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # Add the original input back

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
                )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
                )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device)
                )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

def createModelAndCalculateSize(conf):
    model = GPTModel(conf)
    total_params = sum(p.numel() for p in model.parameters())
    total_size_in_bytes = total_params * 4
    total_size_in_mb = total_size_in_bytes / (1024 * 1024) # Convert to Megabytes
    return total_params, total_size_in_mb

"""
for name in GPT_CONFIGS:
    print(name)
    conf = GPT_CONFIGS[name]
    total_params, size_in_mb = createModelAndCalculateSize(conf)
    print(f"Model {name} has {total_params} parameters and needs {size_in_mb} Mb of memmory.")
"""

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :] # Take last row, which is the newest word
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)
    
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)
    
    # Layer Normalizatin
    
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)
    
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)
    
    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Normalized layer outputs:\n", out_norm)
    print("Mean:\n" , mean)
    print("Variance:\n", var)
    
    torch.set_printoptions(sci_mode=False)
    print("Mean:\n", mean)
    print("Variance:\n", var)


    # Trying LayerNorm in practice
    
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)


    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)


    # First we implement a neural net without shortcut connections
    
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0., -1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)


    print("Model gradients without shortcut:")
    print_gradients(model_without_shortcut, sample_input)
    
    # Now to compare with a model that has gradients
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
            layer_sizes, use_shortcut=True
            )
    print("Model gradients with shortcuts:")
    print_gradients(model_with_shortcut, sample_input)


    # Instantiating transformer block and feeding it some sample data
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    
    print("Transfrormer input shape:", x.shape)
    print("Transformer output shape:", output.shape)

    # Sample batch to our GPT model
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)
    
    # Analyzing the size of the model we coded up earlier
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
    
    total_params_gpt2 = (
            total_params - sum(p.numel() for p in model.out_head.parameters())
            )
    print(f"Number of trainable parameters "
          f"considering weight tying: {total_params_gpt2:,}")
    
    # Feed forward module and multi-head attention module amount of parameters
    one_of_transformers = model.trf_blocks[0]
    feed_forward = one_of_transformers.ff
    attention = one_of_transformers.att
    
    feed_forward_params = sum(p.numel() for p in feed_forward.parameters())
    attention_params = sum(p.numel() for p in attention.parameters())
    
    print(f"Feed forward has {feed_forward_params:,} trainable weights")
    print(f"Attention has {attention_params:,} trainable weights")
    
    # Assesing memmory requirements
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024) # Convert to Megabytes
    print(f"Total size of the model: {total_size_mb:.2f} MB")
    
    # Exercise 4.2 Calculating number of parameters and require memmory for GPT-2 medium, GPT-2 large and GPT-2 XL
    
    GPT_CONFIGS = {
        "GPT-2 medium": {
            "vocab_size": 50257,    # Vocabulary size
            "context_length": 1024, # Context length
            "emb_dim": 1024,         # Embedding dimension
            "n_heads": 16,          # Number of attention heads
            "n_layers": 24,         # Number of layers
            "drop_rate": 0.1,       # Dropout rate
            "qkv_bias": False       # Query-Key-Value bias
        },
        "GPT-2 large": {
            "vocab_size": 50257,    # Vocabulary size
            "context_length": 1024, # Context length
            "emb_dim": 1280,         # Embedding dimension
            "n_heads": 20,          # Number of attention heads
            "n_layers": 36,         # Number of layers
            "drop_rate": 0.1,       # Dropout rate
            "qkv_bias": False       # Query-Key-Value bias
        },
        "GPT-2 XL": {
            "vocab_size": 50257,    # Vocabulary size
            "context_length": 1024, # Context length
            "emb_dim": 1600,         # Embedding dimension
            "n_heads": 25,          # Number of attention heads
            "n_layers": 48,         # Number of layers
            "drop_rate": 0.1,       # Dropout rate
            "qkv_bias": False       # Query-Key-Value bias
        },
    }


    # Lets try it out
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Adds batch dimension
    print("encoded_tensor.shape:", encoded_tensor.shape)
    
    model.eval() # Puts model into eval state to disable random components such as dropout and etc
    out = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=10,
            context_size=GPT_CONFIG_124M["context_length"]
            )
    print("Output:", out)
    print("Output length:", len(out[0]))
    
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

    # Exercise 4.3 Using separate dropout parameters
    
    GPT_CONFIG_124M_SEPARATE_DROOPOUTS = {
            "vocab_size": 50257,    # Vocabulary size
            "context_length": 1024, # Context length
            "emb_dim": 768,         # Embedding dimension
            "n_heads": 12,          # Number of attention heads
            "n_layers": 12,         # Number of layers
            "emb_drop_rate": 0.1,       # Embeddings dropout rate
            "shortcut_drop_rate": 0.1,  # Shortcut dropout rate
            "att_drop_rate": 0.1,       # Attention dropout rate
            "qkv_bias": False       # Query-Key-Value bias
    }
    
    class GPTModelSeparateDropoutParameters(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["emb_drop_rate"])
    
            self.trf_blocks = nn.Sequential(
                    *[TransformerBlockSeparateDropout(cfg) for _ in range(cfg["n_layers"])]
                    )
    
            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(
                    cfg["emb_dim"], cfg["vocab_size"], bias=False
                    )
    
        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
    
            pos_embeds = self.pos_emb(
                    torch.arange(seq_len, device=in_idx.device)
                    )
    
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
    
            return logits

if __name__=="__main__":
    main()
