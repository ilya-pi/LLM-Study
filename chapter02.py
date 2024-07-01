import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

def main():
    ## Test to check intuition how it works together
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    
    # To illustrate the meaning of stride=1
    second_batch = next(data_iter)
    print(second_batch)
    
    # Sampling with a batch size bigger then 1
    
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=8, stride=4)
    
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    
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
    
    # Lets now embed this 8x4 token ID tensor using 256 vector embeddings
    token_embeddings = token_embedding_layer(inputs)
    print("\nToken embeddings shape: ", token_embeddings.shape)
    
    # Adding absolute position embeddings
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print("torch.arange: ", torch.arange(context_length))
    print("Absolute position embeddings: ", pos_embeddings)
    
    # And finally creating input embeddings by adding positional embeddings to the tokeb embeddings
    input_embeddings = token_embeddings + pos_embeddings
    print("Input embeddings shape: ", input_embeddings.shape)
    
if __name__=="__main__":
    main()

