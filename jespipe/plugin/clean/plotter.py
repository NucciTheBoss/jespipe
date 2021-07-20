from abc import ABC, abstractmethod


class Plot(ABC):
    @abstractmethod
    def plot(self):
        pass
