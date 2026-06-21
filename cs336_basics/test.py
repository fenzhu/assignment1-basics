from .tokenizer import BPETokenizer

from tests.common import FIXTURES_PATH
import time

# input_path = FIXTURES_PATH / "tinystories_sample.txt"
input_path = FIXTURES_PATH / "corpus.en"


start_time = time.time()
tokenizer = BPETokenizer(input_path, 500, ["<|endoftext|>"])
end_time = time.time()
print("cost time", end_time - start_time)

# merges = tokenizer.getMerges()
# print(merges)
# i = 1
# for merge in merges:
#     # print(i, merge)
#     i += 1

# path2 = FIXTURES_PATH / "tinystories_sample.txt"
# tokenizer = BPETokenizer(path2, 500, ["<|endoftext|>"])
