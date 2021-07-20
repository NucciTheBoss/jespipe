from abc import ABC, abstractmethod


class Attack(ABC):
    @abstractmethod
    def attack(self):
        pass
