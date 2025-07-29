str = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"


def BPE_Example(str):
    words = str.split(" ")

    indices: dict[tuple[bytes], int] = {}
    for w in words:
        key = tuple(c.encode("utf-8") for c in w)
        indices[key] = indices.get(key, 0) + 1

    print(indices)
    Merge(indices, 12)


from collections import defaultdict
import multiprocessing
import time
from tqdm import tqdm

from cs336_basics.worker_logic import find_chunk_boundaries, init_worker, worker


def Merge(
    inputIndices: dict[tuple[bytes], int], mergeNum=3
) -> tuple[dict[tuple[bytes], int], list[tuple[bytes, bytes]]]:

    indices = {}
    for k, v in inputIndices.items():
        indices[k] = v

    merges: list[tuple[bytes, bytes]] = []

    for _ in range(mergeNum):

        count = defaultdict(int)
        for word, freq in indices.items():
            for i, inext in zip(word, word[1:]):
                count[(i, inext)] += freq

        if len(count) == 0:
            # 无法再合并了
            break

        pair = max(count, key=lambda k: (count[k], k))

        newIndices = {}
        for word, freq in indices.items():

            newWord = []
            length = len(word)

            i = 0
            while i < length:
                if i + 1 < length and pair == (word[i], word[i + 1]):
                    newWord.append(word[i] + word[i + 1])
                    i += 2
                else:
                    newWord.append(word[i])
                    i += 1

            newIndices[tuple(newWord)] = freq

        merges.append(pair)
        indices = newIndices
    return (indices, merges)


# BPE_Example(str)


def BPE_Train(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    train_start_time = time.time()

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]]

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes")

    with open(input_path, "rb") as f_main:
        boundaries = find_chunk_boundaries(
            f_main,
            num_processes,
            [special_token.encode("utf-8") for special_token in special_tokens],
        )

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((start, end, special_tokens))

    results = []

    start_time = time.time()

    with multiprocessing.Pool(
        processes=num_processes, initializer=init_worker, initargs=(input_path,)
    ) as pool:
        async_results = [pool.apply_async(worker, t) for t in tasks]
        with tqdm(total=len(tasks), desc="Preprocessing chunks") as pbar:
            for res in async_results:
                results.append(res.get())
                pbar.update(1)

    end_time = time.time()
    print(f"Pre-tokenization finished in {end_time - start_time:.2f}s")

    wordCount = defaultdict(int)
    for wc in results:
        for k, v in wc.items():
            wordCount[k] += v

    # fileContent = ""
    # with open(input_path, "r", encoding="utf-8") as file:
    #     fileContent = file.read()

    # wordCount = BPE_Pretoken(fileContent, special_tokens)

    mergeNum = max(0, vocab_size - 256 - len(special_tokens))
    wc, merges = Merge(wordCount, mergeNum)

    for i in range(256):
        vocab[i] = bytes([i])
    i = 256
    for token in special_tokens:
        vocab[i] = token.encode("utf-8")
        i += 1

    for a, b in merges:
        vocab[i] = a + b
        i += 1

    train_end_time = time.time()
    print(f"BPE train finished in {train_end_time- start_time:.2f}s")
    return (vocab, merges)


# BPE_Train("../data/TinyStoriesV2-GPT4-valid.txt", 300, ["<|endoftext|>"])
