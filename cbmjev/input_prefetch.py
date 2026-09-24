"""Bounded ordered content IO; model execution remains in the caller thread."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


def validate_input_prefetch(input_workers, input_prefetch_batches):
    if type(input_workers) is not int or input_workers < 0:
        raise ValueError("input_workers must be a nonnegative integer")
    if type(input_prefetch_batches) is not int or input_prefetch_batches < 1:
        raise ValueError("input_prefetch_batches must be a positive integer")


def _read_batch(examples, indices):
    # No transforms, model calls, torch RNG or CUDA in executor threads.
    return [examples[i] for i in indices]


@contextmanager
def ordered_input_batches(examples, index_batches, *, input_workers=0,
                          input_prefetch_batches=2):
    """Yield an ordered batch iterator with at most window submitted futures.

    The caller must keep consumption inside the context. Context exit cancels
    queued reads and waits for in-flight reads even when model execution raises.
    Threaded use requires deterministic, thread-safe ``examples[i]`` content IO.
    Completed futures count toward the window, preventing unbounded read-ahead.
    """
    validate_input_prefetch(input_workers, input_prefetch_batches)
    indices = (tuple(int(i) for i in batch) for batch in index_batches)
    if input_workers == 0:
        yield (_read_batch(examples, batch) for batch in indices)
        return
    executor = ThreadPoolExecutor(max_workers=input_workers,
                                  thread_name_prefix="cbmjev-input")
    pending = deque()

    def submit_next():
        batch = next(indices, None)
        if batch is not None:
            pending.append(executor.submit(_read_batch, examples, batch))

    def batches():
        for _ in range(input_prefetch_batches):
            submit_next()
        while pending:
            yield pending.popleft().result()
            submit_next()

    iterator = batches()
    try:
        yield iterator
    finally:
        iterator.close()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
