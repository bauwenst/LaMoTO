"""
Utility class for outputting a discrete series.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List


class Schedule(ABC):

    def __init__(self):
        self._it_cunt = 0
        self._cum     = 0

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _next(self) -> int:
        pass

    @abstractmethod
    def isDone(self) -> bool:
        pass

    def reset(self):
        self._reset()
        self._it_cunt = 0
        self._cum     = 0

    def next(self) -> Tuple[int,int]:
        """Returns the next value of the schedule, and also its enumerate() index."""
        if self.isDone():
            raise StopIteration("Failed to get next schedule value since it is depleted (schedule.isDone() == True).")
        else:
            n = self._next()
            i = self.count()
            self._cum     += n
            self._it_cunt += 1
            return n, i

    def sum(self) -> int:
        return self._cum

    def count(self) -> int:
        return self._it_cunt


class ConstantSchedule(Schedule):

    def __init__(self, value: int, repeats: int):
        super().__init__()
        self._value = value
        self._repeats = repeats

    def _reset(self):
        pass

    def _next(self) -> int:
        return self._value

    def isDone(self) -> bool:
        return self.count() >= self._repeats  # count is 1 after 1 next() call, and so on.


class NoSchedule(ConstantSchedule):

    def __init__(self):
        super().__init__(value=-1, repeats=0)


class ScheduleCascade(Schedule):

    def __init__(self, schedules: List[Schedule]):
        super().__init__()
        assert len(schedules) > 0
        assert all(schedule.count() == 0 for schedule in schedules)  # Used to ensure the assertion after this makes sense.
        assert all(not schedule.isDone() for schedule in schedules)  # We require every schedule to have at least one step. This means that we never need to move the "current schedule" index by more than 1.
        self._schedules = schedules

        self._current_schedule_idx = 0

    def _next(self) -> int:
        # Calling _next() is only done when not self.isDone(), so we know the following is not out-of-bounds:
        current_schedule = self._schedules[self._current_schedule_idx]

        # ...and because we know every schedule has at least one step, the following will work:
        value,_ = current_schedule.next()

        # ...and if that was the only step in the schedule, we roll to the next one:
        if current_schedule.isDone():
            self._current_schedule_idx += 1

        return value

    def _reset(self):
        self._current_schedule_idx = 0
        for schedule in self._schedules:
            schedule.reset()

    def isDone(self) -> bool:
        return self._current_schedule_idx == len(self._schedules)


class ExponentialSchedule(Schedule):
    """
    Outputs b*a^n for n = 0...repeats-1.
    """

    def __init__(self, start: int, base: int, repeats: int):
        super().__init__()
        self._start   = start
        self._base    = base
        self._repeats = repeats

    def _reset(self):
        pass

    def _next(self) -> int:
        return self._start * (self._base ** self.count())

    def isDone(self) -> bool:
        return self.count() >= self._repeats  # count is 1 after 1 next() call, and so on.


if __name__ == "__main__":
    s = ScheduleCascade([
        ConstantSchedule(value=5, repeats=3),
        ExponentialSchedule(start=15, base=2, repeats=4)
    ])
    while not s.isDone():
        print("So far:", s.sum(), "minutes. Now waiting:", s.next())
    print("Waited for", s.sum(), "minutes across", s.count(), "waits.")