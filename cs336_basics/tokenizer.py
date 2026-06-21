PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re
from collections import defaultdict
import multiprocessing
import time

a = re.findall(PAT, "some text that'll  pre-tokenize")


def process_chunk(chunk):
    special_list = ["<|endoftext|>"]
    pattern = "|".join(map(re.escape, special_list))
    docs = re.split(pattern, chunk)

    bigDoc = "".join(docs)
    preTokens = re.findall(PAT, bigDoc)
    return preTokens


class BPETokenizer:
    def __init__(self, path, size, special_list):
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, word in enumerate(special_list):
            self.vocab[256 + i] = word.encode("utf-8")
        self.merges = []
        self.size = size
        self.special_list = special_list

        # 以二进制读取模式打开文件
        with open(path, "rb") as f:
            start_time = time.time()

            fileData = f.read().decode("utf-8")
            pattern = "|".join(map(re.escape, special_list))
            docs = re.split(f"({pattern})", fileData)

            preTokens = []
            for doc in docs:
                if doc in special_list:
                    preTokens.append(doc)
                else:
                    preTokens.extend(re.findall(PAT, doc))
            # print(preTokens)
            # bigDoc = "".join(docs)
            # preTokens = re.finditer(PAT, bigDoc)

            end_time = time.time()
            print("pretoken cost", end_time - start_time)

            start_time = time.time()
            self.start(preTokens)
            end_time = time.time()
            print("merge cost", end_time - start_time)

            # todo 貌似比直接读取还慢，先关闭多线程了
            # from .pretokenization_example import find_chunk_boundaries

            # num_processes = 4
            # boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            # chunks = []
            # for start, end in zip(boundaries[:-1], boundaries[1:]):
            #     f.seek(start)
            #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
            #     chunks.append(chunk)
            # with multiprocessing.Pool(processes=num_processes) as pool:
            #     result = pool.map(process_chunk, chunks)

            #     preTokens = []
            #     for a in result:
            #         preTokens.extend(a)

            #     end_time = time.time()
            #     print("pretoken cost", end_time - start_time)

            #     self.start(preTokens)

    def start(self, pretokens):
        indices = []
        pairCount = defaultdict(int)
        pairIndices = defaultdict(set)
        for match in pretokens:
            # word = match.group()
            word = match
            if word in self.special_list:
                indices.append([256])
                continue

            index = list(map(int, word.encode("utf-8")))
            indices.append(index)
            for a, b in zip(index, index[1:]):
                pairCount[(a, b)] += 1
                pairIndices[(a, b)].add(word)

        while len(self.vocab) < self.size:
            indices, pairCount, pairIndices = self.merge(
                indices, pairCount, pairIndices
            )

    def merge(self, indices, pairCount, pairIndices):
        maxCount = max(pairCount.values())
        candidates = [p for p, count in pairCount.items() if count == maxCount]
        maxPair = max(candidates, key=lambda p: (self.vocab[p[0]], self.vocab[p[1]]))

        idA, idB = maxPair
        bytesA = self.vocab[idA]
        bytesB = self.vocab[idB]
        self.merges.append((bytesA, bytesB))

        nextId = len(self.vocab)
        self.vocab[nextId] = bytesA + bytesB

        newIndices = []
        s = pairIndices[maxPair].copy()
        for index in indices:
            word = b"".join([self.vocab[i] for i in index]).decode("utf-8")
            if word not in s:
                newIndices.append(index)
                continue

            for a, b in zip(index, index[1:]):
                pairCount[(a, b)] -= 1
                pairIndices[(a, b)].discard(word)

            i = 0
            newIndex = []
            while i < len(index):
                if i + 1 < len(index) and index[i] == idA and index[i + 1] == idB:
                    newIndex.append(nextId)
                    i += 2
                else:
                    newIndex.append(index[i])
                    i += 1

            for a, b in zip(newIndex, newIndex[1:]):
                pairCount[(a, b)] += 1
                pairIndices[(a, b)].add(word)

            newIndices.append(newIndex)

        del pairIndices[maxPair]
        del pairCount[maxPair]

        return newIndices, pairCount, pairIndices

    def getVocab(self):
        return self.vocab

    def getMerges(self):
        return self.merges
