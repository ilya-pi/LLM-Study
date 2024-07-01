import torch
from chapter02 import create_dataloader_v1
from chapter04 import GPTModel
from chapter05 import GPT_CONFIG_124M
from chapter05 import train_model_simple
import tiktoken

def main():
    checkpoint = torch.load("model_and_optimizer.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    
    
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cpu")
    num_epochs=2
    
    # Train and validation loaders
    
    # Load the verdict short story
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    
    # Check the number of characters and tokens in the dataset
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)
    
    train_ratio = 0.90
    split_idx = int(train_ratio *len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    
    train_loader = create_dataloader_v1(
            train_data,
            batch_size=2,
            max_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0
    )
    val_loader = create_dataloader_v1(
            val_data,
            batch_size=2, # A more common batch size would be 1024, this is just for the demonstration purpose
            max_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
    )
    
    print("Loaded model and optimezer")
    
    train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=1,
            start_context="Every effort moved you", tokenizer=tokenizer
    )
    
if __name__ == "__main__":
    main()
