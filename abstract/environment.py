from abc import ABC, abstractmethod


class EnvironmentBase(ABC):

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
