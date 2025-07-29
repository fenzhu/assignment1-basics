from BPE import BPE_Pretoken

f = None

def init_worker(file_path):
    """Initializer for each worker process: opens the file handle."""
    global f
    f = open(file_path, "rb")

def worker(start, end, special_token):
    """Worker function that reads a chunk of a file and processes it."""
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return BPE_Pretoken(chunk, [special_token])
