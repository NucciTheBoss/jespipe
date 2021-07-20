from abc import ABC, abstractmethod


class Evaluate(ABC):
    @abstractmethod
    def model_evaluate(self, model):
        pass
