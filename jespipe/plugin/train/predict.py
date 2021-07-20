from abc import ABC, abstractmethod


class Predict(ABC):
    @abstractmethod
    def model_predict(self):
        pass
