import os
import time
from typing import BinaryIO
import multiprocessing
from tqdm import tqdm
from worker_logic import worker, init_worker

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bystring"

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
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

if __name__ == "__main__":
    # file_path = "../data/TinyStoriesV2-GPT4-train.txt"
    file_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    # file_path = "../data/tinystories_sample_5M.txt"
    special_token = "<|endoftext|>"

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes")

    with open(file_path, "rb") as f_main:
        boundaries = find_chunk_boundaries(f_main, num_processes, special_token.encode("utf-8"))

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((start, end, special_token))

    results = []
    start_time = time.time()

    with multiprocessing.Pool(
        processes=num_processes, initializer=init_worker, initargs=(file_path,)
    ) as pool:
        async_results = [pool.apply_async(worker, t) for t in tasks]
        with tqdm(total=len(tasks), desc="Preprocessing chunks") as pbar:
            for res in async_results:
                results.append(res.get())
                pbar.update(1)

    end_time = time.time()
    print(f"Pre-tokenization finished in {end_time - start_time:.2f}s")