import os
import time
from typing import BinaryIO
import multiprocessing
from tqdm import tqdm
from worker_logic import find_chunk_boundaries, worker, init_worker



if __name__ == "__main__":
    # file_path = "../data/TinyStoriesV2-GPT4-train.txt"
    file_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    # file_path = "../data/tinystories_sample_5M.txt"
    special_token = "<|endoftext|>"

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes")

    with open(file_path, "rb") as f_main:
        boundaries = find_chunk_boundaries(
            f_main, num_processes, special_token.encode("utf-8")
        )

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
