from abc import ABC, abstractmethod

from gymnasium.spaces import Dict, Box, MultiDiscrete, Discrete
import numpy as np


class ObsClass(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def gen_observation_space(self, num_targets, dim=6):
        pass

    @abstractmethod
    def get_observation(self, env):
        pass

class Type1(ObsClass):
    
    def __init__(self) -> None:
        super().__init__()

    def gen_observation_space(self, num_targets, dim=6):
        dictspace = {}

        for i in range(num_targets):
            targetstate = f"target{i+1}_state"
            dictspace[targetstate] = Box(low=-10.0, high=10.0, shape=(dim,), dtype=np.float32)

            targetcov = f"target{i+1}_cov"
            dictspace[targetcov] = Box(low=-10.0, high=10.0, shape=(dim,), dtype=np.float32)

        return Dict(dictspace)
    
    def get_observation(self, env):

        obs = {}

        for i, target in enumerate(env.kalman_objects):
            obs[f"target{i+1}_state"] = np.float32(target.x)
            obs[f"target{i+1}_cov"] = np.float32(np.diag(target.P))

        return obs
    
class Type2(ObsClass):
    def __init__(self) -> None:
        super().__init__()

    def gen_observation_space(self, num_targets, dim=6):
        return Box(low=-np.inf, high=np.inf, shape=(num_targets,), dtype=np.float32)

    def get_observation(self, env):
        obs = np.zeros(env.kalman_objects.size, dtype=np.float32)
        for i, kalman_object in enumerate(env.kalman_objects):
            obs[i] = np.trace(kalman_object.P)

        return obs
    


