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
import re
import regex


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

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]]

    fileContent = ""
    with open(input_path, "r", encoding="utf-8") as file:
        fileContent = file.read()
    parts = re.split(
        "|".join([re.escape(special) for special in special_tokens]), fileContent
    )

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # for part in parts:

    allWords = []
    mergeNum = max(10, vocab_size - 256 - len(special_tokens))

    # for i in range(10):
    for part in parts:
        # part = parts[i]
        # print(part)

        allWords.append(regex.findall(PAT, part.lstrip()))

    wordCount = defaultdict(int)
    for word in regex.findall(PAT, part.lstrip()):
        key = tuple([c.encode("utf-8") for c in word])
        wordCount[key] += 1

    wc, merges = Merge(wordCount, mergeNum)

    for i in range(256):
        vocab[i] = bytes([i])
    i = 256
    for token in special_tokens:
        vocab[i] = token.encode("utf-8")
        i += 1

    # print(merges)
    for a, b in merges:
        vocab[i] = a + b
        i += 1
    # print(vocab)

    return (vocab, merges)


# BPE_Train("../data/TinyStoriesV2-GPT4-valid.txt", 300, ["<|endoftext|>"])
