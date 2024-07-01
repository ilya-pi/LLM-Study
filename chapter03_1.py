import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # x are embeddings
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # x are embeddings
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

    def forward(self, x):
        
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
            )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
                [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)]
                )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
                "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length), diagonal = 1),
                )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)
        return context_vec

def main():
    inputs = torch.tensor(
            [[0.43, 0.15, 0.89], # Your    (x^1)
             [0.55, 0.87, 0.66], # journey (x^2)
             [0.57, 0.85, 0.64], # starts  (x^3)
             [0.22, 0.58, 0.33], # with    (x^4)
             [0.77, 0.25, 0.10], # one     (x^5)
             [0.05, 0.80, 0.55]  # step    (x^6)
             ]
            )
    d_in, d_out = inputs.shape[1], 2
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print("Self attention context v1:", sa_v1(inputs))
    
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print("Self attention context v2:", sa_v2(inputs))
    
    # Now assigning weights from Self Attention 2 to Self Attention 1 to directly compare context vector results
    print("Weights of Self Attention v2:", sa_v2.W_key.weight.T)
    sa_v1.W_query = nn.Parameter(sa_v2.W_query.weight.T)
    sa_v1.W_key = nn.Parameter(sa_v2.W_key.weight.T)
    sa_v1.W_value = nn.Parameter(sa_v2.W_value.weight.T)
    
    print("The following from Self Attention v1 and Self Attention v2 should be the same:\n", sa_v1(inputs), "\n", sa_v2(inputs))
    
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    # Steps 1: Apply Softmax
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
    print("Attention weights:\n", attn_weights)
    
    # Step 2: Mask diagonal weights with 0's
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print("Our simple mask looks like this:", mask_simple)
    masked_simple = attn_weights*mask_simple
    print("Masked weihts:\n", masked_simple)
    
    # Step 3: To renormalize the values with softmax again
    rows_sums = masked_simple.sum(dim=1, keepdim=True)
    print("Row sums:", rows_sums)
    masked_simple_rows = masked_simple / rows_sums
    print("Simple renormalization:", masked_simple_rows)
    
    # The more efficient trick with negative infinities
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print("More efficient masked values:\n", masked)
    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
    print("Attention weights:\n", attn_weights)
    
    # Applying dropout implementation to prevent over-fitting
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    print("Example of a dropout mask:\n", dropout(example))
    # Applying dropout to the attention matrix itself
    torch.manual_seed(123)
    print("Attention weights matrix with the dropout weights:\n", dropout(attn_weights))
    
    # Implementatin in a class
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)

    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)
    
    # Illustrating example of a multihead attention wrapper with a simple example
    
    torch.manual_seed(123)
    context_length = batch.shape[1] # This is the number of tokens
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(
            d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs = mha(batch)
    
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    
    # Exercise 3.2
    mha2d = MultiHeadAttentionWrapper(
            d_in, 1, context_length, 0.0, num_heads=2
            )
    
    context_vecs_2d = mha2d(batch)
    
    print(context_vecs_2d)
    print("context_vecs_2d.shape: ", context_vecs_2d.shape)
    
    # Batch matrix multiplication example
    
    a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                        [0.8993, 0.0390, 0.9268, 0.7388],
                        [0.7179, 0.7058, 0.9156, 0.4340]],
    
                       [[0.0772, 0.3565, 0.1479, 0.5331],
                        [0.4066, 0.2318, 0.4545, 0.9737],
                        [0.4606, 0.5159, 0.4220, 0.5786]]]])
    print("Matrix a:\n", a)
    print("a.transpose(2, 3):\n", a.transpose(2, 3))
    print(a @ a.transpose(2, 3))
    
    # Above is the same as
    first_head = a[0, 0, :, :]
    first_res = first_head @ first_head.T
    print("First head:\n", first_res)
    
    second_head = a[0, 1, :, :]
    second_res = second_head @ second_head.T
    print("\nSecond head:\n", second_res)
    
    # Using multi-head attention class
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
    
    # Exercise 3.3 Initializing GPT-2 size attention modules
    context_length = 1024
    d_in, d_out = 768, 768
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
    print("New MultiHeadAttention of size comparable to GPT-2", mha)

if __name__=="__main__":
    main()
