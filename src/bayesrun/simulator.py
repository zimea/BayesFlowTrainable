from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import odeint

class AbstractSimulator(ABC):
    @abstractmethod
    def simulate(self, params: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_samples(self, n: int = 1000) -> np.ndarray:
        pass

class SIRSimulator(AbstractSimulator):
    def __init__(self, prior, **kwargs):
        self.prior = prior
        try:
            self.n_tot = kwargs['n_tot']
            self.I_0 = kwargs['I_0']
            self.R_0 = kwargs['R_0']
            self.t = kwargs['t']
        except KeyError as e:
            raise KeyError("Please provide the following keyword arguments: n_tot, I_0, R_0, t. \n" + str(e))

    def simulate(self, params: np.ndarray) -> np.ndarray:        
        params_dict = {}
        for i, p in enumerate(self.prior.get_names()):
            params_dict[p] = params[i]

        def simulate_cell_numbers(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt

        S, I, R = odeint(simulate_cell_numbers, (self.n_tot-self.I_0-self.R_0, self.I_0, self.R_0), t=self.t, args=(self.n_tot, params_dict['beta'], params_dict['gamma'])).T

        return np.stack([S, I, R], axis=-1)
    
    def get_samples(self, n: int = 1000) -> dict:
        priors = self.prior.sample(n)
        samples = []
        for i in range(0,n):
            samples.append(self.simulate(priors[i]))
        samples = np.asarray(samples)

        return {'sim_data': samples, 'prior_draws': priors}
