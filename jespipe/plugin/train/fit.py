from abc import ABC, abstractmethod


class Fit(ABC):
    @abstractmethod
    def model_fit(self):
        pass
