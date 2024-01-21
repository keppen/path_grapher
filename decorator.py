from grids import Potential
from abc import abstractmethod


class Decorator(Potential):
    _pot_obj: Potential = None

    def __init__(self, pot_obj: Potential) -> None:
        self._pot_obj = pot_obj

    @property
    def pot_obj(self) -> Potential:
        return self._pot_obj

    @abstractmethod
    def operate(self):
        ...
