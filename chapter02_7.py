import torch

def main():
    input_ids = torch.tensor([2, 3, 5, 1])
    
    # We have vocabulary of only 6 words
    
    # Embeddings size will be 3 (in GPT-3 it is 12,288 dimensions)
    
    vocab_size = 6
    output_dim = 3
    
    torch.manual_seed(123) # For deterministic output
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    
    # To obtain the embedding vector
    print(embedding_layer(torch.tensor([3])))
    
    # Applying embedding to all 4 earlier token IDs
    print(embedding_layer(input_ids))
    
    
    ## For further LLM input we assume 256-dimensional vector representation for embeddings and using BPE tokenizer with a voc of 50257
    
    output_dim = 256
    vocab_size = 50257
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    
    ## Initiating data loader with sampling window
    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    

if __name__=="__main__":
    main()
