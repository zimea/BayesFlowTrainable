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
    

from pyabc import Distribution, RV
class PriorJagiella2017(AbstractPrior):
    def __init__(self):
        self.names = np.array(['log_division_rate', 'log_division_depth', 'log_initial_spheroid_radius', 'log_initial_quiescent_cell_fraction', 'log_ecm_production_rate', 'log_ecm_degradation_rate', 'log_ecm_division_threshold'])

    def sample(self, n: int = 1) -> np.ndarray:
        limits = {
            'log_division_rate':(-3, -1),
            'log_division_depth':(1, 3),
            'log_initial_spheroid_radius':(0, 1.2),
            'log_initial_quiescent_cell_fraction':(-5, 0),
            'log_ecm_production_rate':(-5, 0),
            'log_ecm_degradation_rate':(-5, 0),
            'log_ecm_division_threshold':(-5, 0)
        }
        prior = Distribution(
            **{key: RV("uniform", a, b - a) for key, (a, b) in limits.items()}
        )
        params = []
        for i in tqdm(range(n)):
            p = prior.rvs()
            params.append(p)

        return pd.DataFrame(params).to_numpy()
    

class PriorJagiella2017Delog(AbstractPrior):
    def __init__(self):
        self.names = np.array(['division_rate', 'division_depth', 'initial_spheroid_radius', 'initial_quiescent_cell_fraction', 'ecm_production_rate', 'ecm_degradation_rate', 'ecm_division_threshold'])

    def sample(self, n: int = 1) -> np.ndarray:
        limits = {
            'division_rate':(-3, -1),
            'division_depth':(1, 3),
            'initial_spheroid_radius':(0, 1.2),
            'initial_quiescent_cell_fraction':(-5, 0),
            'ecm_production_rate':(-5, 0),
            'ecm_degradation_rate':(-5, 0),
            'ecm_division_threshold':(-5, 0)
        }
        prior = Distribution(
            **{key: RV("uniform", 10**a, 10**b - 10**a) for key, (a, b) in limits.items()}
        )
        params = []
        for i in tqdm(range(n)):
            p = prior.rvs()
            params.append(p)

        return pd.DataFrame(params).to_numpy()