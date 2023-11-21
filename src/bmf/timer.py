"""Follow time passing."""
import time


class Timer:
    """Time operations."""

    def __init__(self):
        self._start()

    def _start(self):
        """Start a timer."""
        self._lap_times = [time.perf_counter()]

    def lap(self) -> float:
        """Measure time elapsed since last lap and start a new lap."""
        last_time = self._lap_times[-1]
        curr_time = time.perf_counter()
        self._lap_times.append(curr_time)
        return curr_time - last_time

    def elapsed(self) -> float:
        """Read the total time."""
        return time.perf_counter() - self._lap_times[0]

    def reset(self) -> float:
        """Stop the timer, and report the elapsed time and restart."""
        elapsed_time = self.elapsed()
        self._start()
        return elapsed_time
