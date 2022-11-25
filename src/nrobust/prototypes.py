from abc import ABC, abstractmethod


class Protomodel(ABC):
    """
    Prototype class, intended to be used in inheritance,
    not to be called.
    """
    def __init__(self):
        # upon instantiation calling data loading methods
        # and general sanity checks.
        self.y = None
        self.x = None
        self.results = None

    @abstractmethod
    def fit(self):
        # Public method to fit model
        pass


class Protoresult(ABC):
    """
    Prototype class for results object, intended to be used in inheritance,
    not to be called.
    """
    @abstractmethod
    def summary(self):
        # Public method to print summary
        pass

    @abstractmethod
    def plot(self):
        # Public method to plot general results
        pass
