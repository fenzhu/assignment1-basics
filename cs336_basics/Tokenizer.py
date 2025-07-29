from typing import Iterable, Iterator

from cs336_basics.worker_logic import BPE_Split_Reserve


class CSTokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens=None,
    ):
        self.vocab = vocab
        self.reverse_vocab = {}
        for token, v in vocab.items():
            self.reverse_vocab[v] = token

        self.merges = merges
        self.special_tokens = special_tokens or []

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        """
        1. Pre-Token, 将输入文本预先分词, 先处理special_tokens, 再正则分词, 和训练时保持一致
        2. 按照merges列表的顺序, 在每个分词内分别执行合并
        3. 将合并完成的每个分词, 映射回vocab字典中的Token Id
        """
        tokens = []

        words = BPE_Split_Reserve(text, self.special_tokens)
        for word in words:
            byte_list = self.__merge_word(word)
            tokens.extend([self.reverse_vocab[b] for b in byte_list])

        return tokens

    def __merge_word(self, word: str) -> list[bytes]:
        if word in self.special_tokens:
            return [word.encode("utf-8")]

        key = [bytes([b]) for b in word.encode("utf-8")]

        for ma, mb in self.merges:
            index = [i for i in range(len(key))]
            for i, a, b in zip(index, key, key[1:]):
                if a == ma and b == mb:
                    key = key[:i] + [a + b] + key[i + 2 :]
                    break

        return key

    def decode(self, ids: list[int]) -> str:
        byte_list = []
        for token in ids:
            byte_list.append(self.vocab[token])
        return b"".join(byte_list).decode("utf-8", errors="replace")

    # 注意大文件的情况, 分块处理
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            tokens = self.encode(text)
            for token in tokens:
                yield token
