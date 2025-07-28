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


def Merge(indices: dict[tuple[bytes], int], mergeNum=3):

    for _ in range(mergeNum):
        count = defaultdict(int)
        for word, freq in indices.items():
            for i, inext in zip(word, word[1:]):
                count[(i, inext)] += freq

        pair = max(count, key=lambda k: (count[k], k))
        print(pair)

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

        indices = newIndices


BPE_Example(str)
