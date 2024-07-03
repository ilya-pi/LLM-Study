import urllib.request
from chapter02 import create_dataloader_v1
from chapter04 import GPTModel
from chapter05 import GPT_CONFIG_124M, generate, text_to_token_ids, token_ids_to_text, calc_loss_loader
import torch
import tiktoken
import numpy as np

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

# This implementation is mainly a lot of guess work on behalf of author of the book
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        # Map Query, Key and Value weights
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        
        # Map Query, Key and Value biases
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b.T)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b.T)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b.T)

        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def main():
    # Load file that has logic to download GPT-2 weights in Tensor Flow format
    url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch05/"
            "01_main-chapter-code/gpt_download.py"
    )
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    
    from gpt_download import download_and_load_gpt2
    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2"
    )
    
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())
    
    print(params["wte"])
    print("Token embedding weight tensor dimensions:", params["wte"].shape)
    
    model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
            }
    
    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    
    NEW_CONFIG.update({"context_length": 1024})
    
    # Biases are not used anymore as they don't seem to improve modeling performance, need to update it for consistency
    NEW_CONFIG.update({"qkv_bias": True})
    
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Now with update config we can init our model
    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    # Let's try to load it now
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    
    torch.manual_seed(123)
    token_ids = generate(
            model=gpt,
            idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
            max_new_tokens=25,
            context_size=NEW_CONFIG["context_length"],
            top_k=50,
            temperature=1.5
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    
    # Exercise 5.5 Losses as measured on the verdict
    
    # Load the verdict short story
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    
    train_ratio = 0.50
    split_idx = int(train_ratio *len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    torch.manual_seed(123)
    
    train_loader = create_dataloader_v1(
            train_data,
            batch_size=2,
            max_length=NEW_CONFIG["context_length"],
            stride=NEW_CONFIG["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0
    )
    val_loader = create_dataloader_v1(
            val_data,
            batch_size=2, # A more common batch size would be 1024, this is just for the demonstration purpose
            max_length=NEW_CONFIG["context_length"],
            stride=NEW_CONFIG["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
    )
    
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, gpt, device)
        val_loss = calc_loss_loader(val_loader, gpt, device)
    
    print("Train lossess:\n", train_loss)
    print("Validation losses:\n", val_loss)
