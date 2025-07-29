import base64
import json
import time
from BPE import BPE_Train


start_time = time.time()

vocab, merges = BPE_Train(
    "../data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"]
)

end_time = time.time()

print(f"BPE train cost time {end_time-start_time}s")

# 序列化字典到第一个文件
m = 0
vo = ""
for k, v in vocab.items():
    if len(v) > m:
        m = len(v)
        vo = str(v)
print(f"longest token: {vo}")


with open("vocab", "w") as f:
    json.dump(
        {str(k): str(v) for k, v in vocab.items()}, f, ensure_ascii=False, indent=2
    )

# 序列化列表到第二个文件
with open("merge", "w") as f:
    json.dump([(str(a), str(b)) for a, b in merges], f, ensure_ascii=False, indent=2)
