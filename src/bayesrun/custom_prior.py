from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tqdm import tqdm

class AbstractPrior(ABC):
    @abstractmethod
    def sample(self, n: int = 1) -> np.ndarray:
        pass
    
    def get_names(self) -> np.ndarray:
        return self.names
    
    def estimate_mean_and_std(self, n: int = 1000) -> np.ndarray:
        samples = self.sample(n)
        return np.mean(samples, axis=0), np.std(samples, axis=0)
    
class SIRPrior(AbstractPrior):
    def __init__(self):
        self.names = np.array(["beta", "gamma"])

    def sample(self, n: int = 1) -> np.ndarray:
        beta = np.random.uniform(0,1,n)
        gamma = np.random.uniform(0,1,n)
        if n == 1:
            return np.r_[beta, gamma]
        return np.r_[beta, gamma].reshape(n,-1)
    
class SIRPriorDependency(AbstractPrior):
    def __init__(self):
        self.names = np.array(["beta", "gamma"])

    def sample(self, n: int = 1) -> np.ndarray:
        beta = np.random.uniform(0,1,n)
        gamma = beta**2 + np.random.normal(-0.1,0.1,n)
        if n == 1:
            return np.r_[beta, gamma]
        return np.r_[beta, gamma].reshape(n,-1)