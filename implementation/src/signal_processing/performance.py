import numpy as np
from scipy.signal import iirnotch
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

@lru_cache(maxsize=8)
def get_filter_coeffs(f0, q, fs):
    nyq = fs / 2.0
    b, a = iirnotch(f0 / nyq, q)
    return b, a

class ParallelProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)

    def run_async(self, func, *args):
        return self.executor.submit(func, *args)

    def shutdown(self):
        self.executor.shutdown(wait=True)