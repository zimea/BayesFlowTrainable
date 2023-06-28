from abc import ABC, abstractmethod
from bayesrun.custom_prior import AbstractPrior
import numpy as np
from bayesrun.util.enums import NormalizationType

class AbstractConfigurator(ABC):
    def __init__(self, data_sample: dict, prior: AbstractPrior, **kwargs):
        self.data = data_sample
        self.prior = prior

    @abstractmethod
    def configure_input(self, forward_dict: dict) -> dict:
        pass

    @abstractmethod
    def normalize_input(self, forward_dict: dict) -> dict:
        pass

    @abstractmethod
    def normalize_params(self, params: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def unnorm_input(self, forward_dict: dict) -> dict:
        pass

    @abstractmethod
    def unnorm_params(self, params: np.ndarray) -> np.ndarray:
        pass

    def __mean_norm__(self, data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (data - mean) / std
    
    def __mean_unnorm__(self, data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return data * std + mean

class ConfiguratorJagiella2017(AbstractConfigurator):
    def __init__(self, data_sample: dict, prior: AbstractPrior, **kwargs):
        super().__init__(data_sample, prior)
        self.mean_sim, self.std_sim = self.data['sim_data'].mean(axis=(0, 1), keepdims=True), self.data['sim_data'].std(axis=(0, 1), keepdims=True)
        self.mean_growth, self.std_growth = self.data['growth_curve'].mean(axis=(0, 1), keepdims=True), self.data['growth_curve'].std(axis=(0, 1), keepdims=True)

        self.prior_means, self.prior_stds = self.prior.estimate_mean_and_std()

        try:
            self.pos_function = kwargs['pos_function']
            self.nr_pos_embs = kwargs['nr_positional_embeddings']
        except KeyError:
            print('No positional encoding function provided.')

        try:
            self.sim_normalization = kwargs['normalization']['simulations']
            self.param_normalization = kwargs['normalization']['parameters']
        except KeyError:
            self.sim_normalization = NormalizationType.NONE
            self.param_normalization = NormalizationType.NONE

    def configure_input(self, forward_dict: dict) -> dict:
        out_dict = {}

        forward_dict = self.normalize_input(forward_dict)
        pp_ecmp = forward_dict['sim_data']
        pos = self.pos_function(seq_len=pp_ecmp.shape[1])
        pos = np.tile(pos, (pp_ecmp.shape[0], 1)).reshape((pp_ecmp.shape[0], pp_ecmp.shape[1], self.nr_pos_embs))
        insert_encodings = tuple([2] * self.nr_pos_embs)
        pp_ecmp = np.insert(pp_ecmp, insert_encodings, pos, axis=-1)
        
        # Extract prior draws and z-standardize with previously computed means
        params = forward_dict['prior_draws'].astype(np.float64)
        params = (params - self.prior_means) / self.prior_stds

        # Add to bayesflow keys
        out_dict = {
            'summary_conditions': (pp_ecmp, forward_dict['growth_curve']),
            'targets': params
        }

        return out_dict
    
    def normalize_input(self, forward_dict: dict) -> dict:
        if self.sim_normalization == NormalizationType.MEAN:
            forward_dict['sim_data'] = self.__mean_norm__(forward_dict['sim_data'], self.mean_sim, self.std_sim)
            forward_dict['growth_curve'] = self.__mean_norm__(forward_dict['growth_curve'], self.mean_growth, self.std_growth)
            return forward_dict
        print('No normalization applied.')
        return forward_dict
    
    def normalize_params(self, params: np.ndarray) -> np.ndarray:
        if self.param_normalization == NormalizationType.MEAN:
            return self.__mean_norm__(params, self.prior_means, self.prior_stds)
        print('No normalization applied.')
        return params
    
    def unnorm_input(self, forward_dict: dict) -> dict:
        if self.param_normalization == NormalizationType.MEAN:
            forward_dict['sim_data'] = self.__mean_unnorm__(forward_dict['sim_data'], self.mean_sim, self.std_sim)
            forward_dict['growth_curve'] = self.__mean_unnorm__(forward_dict['growth_curve'], self.mean_growth, self.std_growth)
        return forward_dict
    
    def unnorm_params(self, params: np.ndarray) -> np.ndarray:
        if self.param_normalization == NormalizationType.MEAN:
            return self.__mean_unnorm__(params, self.prior_means, self.prior_stds)
        return params