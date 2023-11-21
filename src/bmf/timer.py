"""Follow time passing"""
import time


class Timer:
    """Time operations"""

    def __init__(self):
        """Start a timer"""
        self._lap_times = [time.perf_counter()]

    def lap(self) -> float:
        """Return the time since the last call to lap() (or start() if this is the first)."""
        last_time = self._lap_times[-1]
        curr_time = time.perf_counter()
        self._lap_times.append(curr_time)
        return curr_time - last_time

    def elapsed(self) -> float:
        """Read the total time"""
        return time.perf_counter() - self._lap_times[0]

    def reset(self) -> float:
        """Stop the timer, and report the elapsed time and restart"""
        elapsed_time = self.elapsed()
        self.__init__()
        return elapsed_time
