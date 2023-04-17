__all__ = ['Interval']

from dataclasses import dataclass

@dataclass(order=True, frozen=True)
class Interval():
    start: float
    end: float

    def __contains__(self, value):
        return self.start <= value <= self.end