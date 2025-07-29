from collections import defaultdict
import os
import re
from typing import BinaryIO

import regex


f = None


def init_worker(file_path):
    """Initializer for each worker process: opens the file handle."""
    global f
    f = open(file_path, "rb")


def worker(start, end, special_tokens):
    """Worker function that reads a chunk of a file and processes it."""
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return BPE_Pretoken(chunk, special_tokens)


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def BPE_Split(text: str, special_tokens: list[str]) -> list[str]:

    parts = re.split(
        "|".join([re.escape(special) for special in special_tokens]),
        text,
    )

    allWords = []
    for part in parts:
        # 使用extend扁平化合并
        allWords.extend(regex.findall(PAT, part))

    return allWords


def BPE_Pretoken(text: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    """
    Pretokenize the input text into a dictionary of byte tuples and their frequencies.
    """
    allWords = BPE_Split(text, special_tokens)

    wordCount = defaultdict(int)
    for word in allWords:

        # 单个字符可能由多个字节组成，例如emoji😃
        # 如果先拆分字符再转bytes，会导致初始key中存在包含多个字节的元素
        # key = tuple([c.encode("utf-8") for c in word])

        # 先转为bytes，再将每个字节都转为bytes对象，确保初始key的每个元素都是单个字节
        key = tuple(bytes([b]) for b in word.encode("utf-8"))
        wordCount[key] += 1

    return wordCount


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # assert isinstance(
    #     split_special_tokens, list[bytes]
    # ), "Must represent special token as a bystring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            for split_special_token in split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break

            if found_at != -1:
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))
