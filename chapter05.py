import torch
import tiktoken
import matplotlib.pyplot as plt
from chapter02 import create_dataloader_v1
from chapter04 import GPTModel
from chapter04 import generate_text_simple


GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256, # Shortened from 1024 to make it easier to train on a laptop
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
}

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # .unsqueeze(0) adds the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # Remove batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                    input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
                train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
                val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
                model=model, idx=encoded,
                max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ")) # Compact print format
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Resets loss gradiens from the previous epoch
            loss = calc_loss_batch(
                    input_batch, target_batch, model, device
            )
            loss.backward() # Calculate loss gradients
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f} "
                      f"Val loss {val_loss:.3f} ")

        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                    logits < min_val,
                    torch.tensor(float('-inf')).to(logits.device),
                    logits
                    )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

def main():
    print("\n\n\nChapter 5\n\n\n")
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()


    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    
    token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(start_context, tokenizer),
            max_new_tokens=10,
            context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    
    
    # Calculating loss function for training
    
    inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                           [40, 1107, 588]])    # "I really like"]
    
    targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                            [588, 428, 11311]]) # " really like chocholate"]
    
    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    print("Probabilities received for the next word:", probas.shape)
    
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)
    
    print(f"Target batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1:"
          f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)
    
    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)
    
    # Applying logarithm to the probability scores
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print("Log_probas:", log_probas)
    
    avg_log_probas = torch.mean(log_probas)
    print("Average log probabilities:", avg_log_probas)
    
    neg_avg_log_probas = avg_log_probas * -1
    print("Negative average log probas:", neg_avg_log_probas)
    
    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)
    
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("Loss:", loss)
    
    perplexity = torch.exp(loss) # Which among the n words of vocabulary to generate as the next token
    print(f"Perplexity is {perplexity}")
    
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
    
    torch.manual_seed(123)
    
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
    
    # Iterate through the data loaders to check that they were created correctly
    
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)



    # Trying this calc_loss_batch function in action
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    # device = torch.device("cpu")
    model.to(device)
    with torch.no_grad(): # B
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


    # Oke, trying to pre-train the first model, fingers crossed
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0004, weight_decay=0.1
    )
    num_epochs = 10
    print("\n\nStarting to traing!\n\n")
    train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=1,
            start_context="Every effort moved you", tokenizer=tokenizer
    )


    # Below will show graph of discrepancy of losses in training vs losses in 
    # epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    # plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    
    # print("epochs tensor:", epochs_tensor)
    # print("tokens seen:", tokens_seen)
    # print("train losses:", train_losses)
    # print("val losses:", val_losses)
    
    model.to("cpu")
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids("Every effort moves you", tokenizer),
            max_new_tokens=25,
            context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    
    # Let's now try this new generate function
    
    torch.manual_seed(123)
    token_ids = generate(
            model=model,
            idx=text_to_token_ids("Every effort moves you", tokenizer),
            max_new_tokens=15,
            context_size=GPT_CONFIG_124M["context_length"],
            top_k=25,
            temperature=1.4
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    
    print("Saving the pretrained model")
    torch.save(model.state_dict(), "model.pth")
    
    print("Loading just saved model")
    
    model_loaded = GPTModel(GPT_CONFIG_124M)
    model_loaded.load_state_dict(torch.load("model.pth"))
    model_loaded.eval()
    
    token_ids = generate(
            model=model_loaded,
            idx=text_to_token_ids("It was another Thursday in the office", tokenizer),
            max_new_tokens=15,
            context_size=GPT_CONFIG_124M["context_length"],
            top_k=1,
            temperature=1.,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    
    # Saving model together with optimizer to continue training from that step
    
    print("Saving model together with the optimizer now")
    torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_and_optimizer.pth"
    ) 

if __name__ == "__main__":
    main()
