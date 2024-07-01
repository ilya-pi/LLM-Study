import torch
import matplotlib.pyplot as plt

def main():
    vocab = {
            "closer": 0,
            "every": 1,
            "effort": 2, 
            "forward": 3,
            "inches": 4,
            "moves": 5,
            "pizza": 6,
            "toward": 7,
            "you": 8,
            }
    inverse_vocab = {v: k for k, v in vocab.items()}
    
    # Assumed next token logits
    next_token_logits = torch.tensor(
            [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
            )
    
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print("Argmax nest token: ", inverse_vocab[next_token_id])
    
    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print("Multinomial next token:", inverse_vocab[next_token_id])
    
    def print_sampled_tokens(probas):
        torch.manual_seed(123)
        sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
        sample_ids = torch.bincount(torch.tensor(sample))
        for i, freq in enumerate(sample_ids):
            print(f"{freq} x {inverse_vocab[i]}")
    print_sampled_tokens(probas)
    
    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=0)
    
    
    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f"Temperature = {T}")
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    # plt.show()
    
    # Exercise 5.1
    
    for i in range(len(temperatures)):
        temperature = temperatures[i]
        scaled_probas = softmax_with_temperature(probas, temperature)
        print(f"Next token frequency for temperature = {temperature}:")
        print_sampled_tokens(scaled_probas)
    
    # Top-K selection
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print()
    print("Top logits:", top_logits)
    print("Top positions:", top_pos)
    
    # Set all logit values below the lowest of topk to -Inf
    new_logits = torch.where(
            condition=next_token_logits < top_logits[-1],
            input=torch.tensor(float('-inf')),
            other=next_token_logits
    )
    print("New logits are:", new_logits)
    topk_probas = torch.softmax(new_logits, dim=0)
    print("TopK Probas:", topk_probas)
    
if __name__ == "__main__:
    main()
