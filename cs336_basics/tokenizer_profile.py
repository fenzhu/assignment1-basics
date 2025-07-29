from cs336_basics.BPE import BPE_Train
from cs336_basics.Tokenizer import CSTokenizer


special_tokens = ["<|endoftext|>"]
# vocab, merges = BPE_Train("../data/TinyStoriesV2-GPT4-valid.txt", 10000, special_tokens)
vocab, merges = BPE_Train("../data/tinystories_sample_5M.txt", 10000, special_tokens)

# print(vocab)
# print(merges)

tokenizer = CSTokenizer(vocab, merges)

input = "hello, world! <|endoftext|> This is a test input."
tokens = tokenizer.encode(input)
print(f"Encoded tokens for '{input}': {tokens}")
print(tokenizer.decode(tokens))
