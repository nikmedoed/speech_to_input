from typing import NamedTuple

from dataclasses import dataclass


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float

    def add_offset(self, offset: float | int):
        self.start += offset
        self.end += offset
        return self