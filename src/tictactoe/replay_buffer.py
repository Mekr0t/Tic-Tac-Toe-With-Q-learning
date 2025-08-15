import pickle, pathlib, collections, tempfile, random

class DiskReplayBuffer:
    """
    In-memory deque that spills to disk when maxlen exceeded.
    Uses a single temp file; good enough for 1-2 M steps on a laptop.
    """
    def __init__(self, maxlen: int = 100_000, filename: str | None = None):
        self._maxlen = maxlen
        self._path   = pathlib.Path(filename or tempfile.mktemp(prefix="t3_buffer_"))
        self._cache  = collections.deque(maxlen=maxlen)

    def append(self, transition):
        self._cache.append(transition)
        if len(self._cache) == self._maxlen:
            self._flush()

    def extend(self, transitions):
        self._cache.extend(transitions)
        if len(self._cache) >= self._maxlen:
            self._flush()

    def sample(self, n):
        if self._path.exists():
            with self._path.open("rb") as f:
                data = pickle.load(f)
            return random.sample(data, min(n, len(data)))
        return random.sample(list(self._cache), min(n, len(self._cache)))

    def _flush(self):
        with self._path.open("wb") as f:
            pickle.dump(list(self._cache), f)
        self._cache.clear()

    def __len__(self):
        return len(self._cache) + (self._path.stat().st_size if self._path.exists() else 0)