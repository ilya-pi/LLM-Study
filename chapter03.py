import torch

def softmax_naive(x):
    print("arg x", x)
    print("torch.exp(x)", torch.exp(x))
    return torch.exp(x) / torch.exp(x).sum(dim=0)

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
    
    print("Embedding for our sentence ", inputs) 
    
    # Computing context vector for embedding x^2
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    print("Attention scores for query 2: ", attn_scores_2)
    
    # We now need to normalize it, as it is useful for interpretation and for maintaining training stability in an LLM
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())


    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())
    
    # But to avoid numerical instability such as overflow and underflow it is better to use PyTorchs implementation of softmax
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())
    
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i,x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i]*x_i
    print("Context of the second token is: ", context_vec_2)
    
    # Now to compute context vectors for all the pairs of inputs
    attn_scores = torch.empty(6, 6) # batch size is 6
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    print("Attention scores: ", attn_scores)
    
    # We can compute the same with the matrix multiplication
    attn_score = inputs @ inputs.T
    print(attn_scores)
    
    # Normalizing each row
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print("Attention weights (normalized attention scores): ", attn_weights)
    
    # Verifying that all rows indeed sum up to 1
    row_2_sum = sum(attn_weights[1])
    print("Row 2 sum:", row_2_sum)
    print("All row sums:", attn_weights.sum(dim=-1))
    
    # Calculation context vectors
    all_context_vecs = attn_weights @ inputs
    print("All context vectors:", "\n", all_context_vecs)
    
    # Comparing with the previously computed context vector for second query
    print("Previous 2nd context vector:", context_vec_2)
    
    
    ## For illustrative purpose we will first compute context vector z^2 for x^2
    print(inputs[1])
    
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2 # output dimension is different for illustration purposes
    
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    
    print("Matrices for query, key and value:\n", W_query, W_key, W_value)
    
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)
    
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)
    
    ## Computing attention score w^22
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print("Attention score w^22:", attn_score_22)
    
    # Generalize it to all attention scores
    attn_scores_2 = query_2 @ keys.T
    print("All attention scores for second query:", attn_scores_2)
    print("Previously computed attention score for second key:", attn_score_22)
    
    # Normalizing with softmax
    d_k = keys.shape[-1]
    print("Keys dimensions: ", d_k)
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
    print("Attention weights: ", attn_weights_2)
    
    # Computing final context vector for x^2
    context_vec_2 = attn_weights_2 @ values
    print("Context vector for x^2:", context_vec_2)
    
    ## Organizing everything above into a python class

if __name__=="__main__":
    main()
