import time
import torch
from contextlib import ContextDecorator

class Timer(ContextDecorator):
    def __init__(self, label="", use_gpu=False):
        self.label = label
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.start = None
        self.end = None
        self.elapsed = None

        if self.use_gpu:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.use_gpu:
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.use_gpu:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed = self.start_event.elapsed_time(self.end_event) / 1000.0  # seconds
        else:
            self.end = time.perf_counter()
            self.elapsed = self.end - self.start

        # print(f"{self.label}Elapsed time: {self.elapsed:.6f} seconds")
        return False  # Don't suppress exceptions
