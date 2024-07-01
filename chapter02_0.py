
# Full calss for a simple text tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        # Split om punctuation and spaces
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        # Sanitize empty strings
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Mark replaced unknown tokens
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Add space after punctuation symbols
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

def main():
    # Read text from file
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of characters:", len(raw_text))
    print(raw_text[:99])
    
    
    # Split text on punctuation and whitespaces
    import re
    result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    print(result[:99])
    
    # Remove whitespace characters
    result = [item for item in result if item.strip()]
    print(result[:99])
    
    # Move to a different variable for the convenience of following along study material
    preprocessed = result
    print(f"Total number of tokens is {len(preprocessed)}")
    print(f"First 30  are {preprocessed[:30]}")
    
    # Map tokens to token IDs
    all_tokens = sorted(list(set(preprocessed)))
    # Add "unnknown" and "end of text" tokens
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_tokens)
    print(vocab_size)
    
    # Creating a vocabulary
    vocab = {token:integer for integer,token in enumerate(all_tokens)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i > 50:
            break
    
    # Check for new tokens
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)

    tokenizer = SimpleTokenizerV1(vocab)
    
    # Turn a sample text sequence into IDs
    text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    
    # Convert IDs sequence back into text
    print(tokenizer.decode(ids))
    
    # Applying tokenizer to a new text not seen in the training set
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))
    
    ## Byte pair encoding
    from importlib.metadata import version
    import tiktoken
    print("tiktoken version:", version("tiktoken"))
    
    # Instantiating Byte-Pair Tokenizer from tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    # Decode back into text
    strings = tokenizer.decode(integers)
    print(strings)
    
    # Exercise 2.1
    unknownWords = "Akwirw ier"
    integers2 = tokenizer.encode(unknownWords, allowed_special={"<|endoftext|>"})
    print(integers2)
    for item in integers2:
        print(f"{item} -> {tokenizer.decode([item])}")
    print(tokenizer.decode(integers2))
    
    # 2.6 Now onto data sampling with the sliding window
    
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))
    
    enc_sample = enc_text[50:]
    
    context_size = 4
    
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(f"x :{x}")
    print(f"y:       {y}")
    
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)
    
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
        
if __name__=="__main__":
    main()
