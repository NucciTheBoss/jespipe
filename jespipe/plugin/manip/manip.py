from abc import ABC, abstractmethod


class Manipulation(ABC):
    @abstractmethod
    def manipulate(self):
        pass
