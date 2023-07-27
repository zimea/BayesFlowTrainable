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

class SIRConfigurator(AbstractConfigurator):
    def __init__(self, data_sample: dict, prior: AbstractPrior, **kwargs):
        super().__init__(data_sample, prior)
        self.sim_means, self.sim_stds = self.data['sim_data'].mean(axis=(0, 1), keepdims=True), self.data['sim_data'].std(axis=(0, 1), keepdims=True)
        self.prior_means, self.prior_stds = self.prior.estimate_mean_and_std()

        try:
            self.sim_normalization = kwargs['normalization']['simulations']
            self.param_normalization = kwargs['normalization']['parameters']
        except KeyError:
            self.sim_normalization = NormalizationType.NONE
            self.param_normalization = NormalizationType.NONE

    def configure_input(self, forward_dict: dict) -> dict:
        """Configures dictionary of prior draws and simulated data into BayesFlow format."""
    
        out_dict = {}
        
        # standardization sim_data
        sim_data = forward_dict['sim_data'].astype(np.float32)
        norm_data = (sim_data - self.sim_means) / self.sim_stds
        
        # standardization priors
        params = forward_dict['prior_draws'].astype(np.float32)
        norm_params = (params - self.prior_means) / self.prior_stds
        
        # remove nan, inf and -inf
        keep_idx = np.all(np.isfinite(norm_data), axis=(1, 2))
        if not np.all(keep_idx):
            print('Invalid value encountered...removing from batch')
            
        # add to dict
        out_dict['summary_conditions'] = norm_data[keep_idx]
        out_dict['targets'] = norm_params[keep_idx]
        
        return out_dict
    
    def normalize_input(self, forward_dict: dict) -> dict:
        if self.sim_normalization == NormalizationType.MEAN:
            forward_dict['sim_data'] = self.__mean_norm__(forward_dict['sim_data'], self.mean_sim, self.std_sim)
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
        return forward_dict
    
    def unnorm_params(self, params: np.ndarray) -> np.ndarray:
        if self.param_normalization == NormalizationType.MEAN:
            return self.__mean_unnorm__(params, self.prior_means, self.prior_stds)
        return params
    
class SIRConfiguratorEta(AbstractConfigurator):
    def __init__(self, data_sample: dict, prior: AbstractPrior, **kwargs):
        super().__init__(data_sample, prior)
        self.prior = prior
        self.sim_means, self.sim_stds = self.data['sim_data'].mean(axis=(0, 1), keepdims=True), self.data['sim_data'].std(axis=(0, 1), keepdims=True)

        # calculate prior means and stds
        prior_samples = self.prior.sample(1000)
        self.prior_names = self.prior.get_names()
        beta_idx = np.where(self.prior_names == 'beta')[0][0]
        gamma_idx = np.where(self.prior_names == 'gamma')[0][0]
        prior_samples[:, beta_idx] = (np.sqrt(1 + prior_samples[:, beta_idx]) - 1) / prior_samples[:, gamma_idx]
        self.prior_means, self.prior_stds = prior_samples.mean(axis=0, keepdims=True), prior_samples.std(axis=0, keepdims=True)

        try:
            self.sim_normalization = kwargs['normalization']['simulations']
            self.param_normalization = kwargs['normalization']['parameters']
        except KeyError:
            self.sim_normalization = NormalizationType.NONE
            self.param_normalization = NormalizationType.NONE

    def configure_input(self, forward_dict: dict) -> dict:
        """Configures dictionary of prior draws and simulated data into BayesFlow format."""
    
        out_dict = {}
        
        # standardization sim_data
        sim_data = forward_dict['sim_data'].astype(np.float32)
        norm_data = (sim_data - self.sim_means) / self.sim_stds
        
        # standardization priors
        params = forward_dict['prior_draws'].astype(np.float32)
        prior_names = self.prior.get_names()
        # find the index of the parameter beta in prior_names
        beta_idx = np.where(prior_names == 'beta')[0][0]
        gamma_idx = np.where(prior_names == 'gamma')[0][0]
        #params[:, beta_idx] = (np.sqrt(1 + params[:, beta_idx]) - 1) / (1/params[:, gamma_idx])
        params[:, beta_idx] = (np.sqrt(1 + params[:, beta_idx]) - 1) / np.sqrt(params[:, gamma_idx])
        norm_params = (params - self.prior_means) / self.prior_stds
        
        # remove nan, inf and -inf
        keep_idx = np.all(np.isfinite(norm_data), axis=(1, 2))
        if not np.all(keep_idx):
            print('Invalid value encountered...removing from batch')
            
        # add to dict
        out_dict['summary_conditions'] = norm_data[keep_idx]
        out_dict['targets'] = norm_params[keep_idx]
        
        return out_dict
    
    def normalize_input(self, forward_dict: dict) -> dict:
        if self.sim_normalization == NormalizationType.MEAN:
            forward_dict['sim_data'] = self.__mean_norm__(forward_dict['sim_data'], self.mean_sim, self.std_sim)
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
        return forward_dict
    
    def unnorm_params(self, params: np.ndarray) -> np.ndarray:
        if self.param_normalization == NormalizationType.MEAN:
            return self.__mean_unnorm__(params, self.prior_means, self.prior_stds)
        return params
    
class SIRConfiguratorEtaSqrt(AbstractConfigurator):
    def __init__(self, data_sample: dict, prior: AbstractPrior, **kwargs):
        super().__init__(data_sample, prior)
        self.prior = prior
        self.sim_means, self.sim_stds = self.data['sim_data'].mean(axis=(0, 1), keepdims=True), self.data['sim_data'].std(axis=(0, 1), keepdims=True)

        # calculate prior means and stds
        prior_samples = self.prior.sample(1000)
        self.prior_names = self.prior.get_names()
        beta_idx = np.where(self.prior_names == 'beta')[0][0]
        gamma_idx = np.where(self.prior_names == 'gamma')[0][0]
        prior_samples[:, beta_idx] = (np.sqrt(1 + prior_samples[:, beta_idx]) - 1) / np.sqrt(prior_samples[:, gamma_idx])
        self.prior_means, self.prior_stds = prior_samples.mean(axis=0, keepdims=True), prior_samples.std(axis=0, keepdims=True)

        try:
            self.sim_normalization = kwargs['normalization']['simulations']
            self.param_normalization = kwargs['normalization']['parameters']
        except KeyError:
            self.sim_normalization = NormalizationType.NONE
            self.param_normalization = NormalizationType.NONE

    def configure_input(self, forward_dict: dict) -> dict:
        """Configures dictionary of prior draws and simulated data into BayesFlow format."""
    
        out_dict = {}
        
        # standardization sim_data
        sim_data = forward_dict['sim_data'].astype(np.float32)
        norm_data = (sim_data - self.sim_means) / self.sim_stds
        
        # standardization priors
        params = forward_dict['prior_draws'].astype(np.float32)
        prior_names = self.prior.get_names()
        # find the index of the parameter beta in prior_names
        beta_idx = np.where(prior_names == 'beta')[0][0]
        gamma_idx = np.where(prior_names == 'gamma')[0][0]
        #params[:, beta_idx] = (np.sqrt(1 + params[:, beta_idx]) - 1) / (1/params[:, gamma_idx])
        params[:, beta_idx] = (np.sqrt(1 + params[:, beta_idx]) - 1) / np.sqrt(params[:, gamma_idx])
        params = (params - self.prior_means) / self.prior_stds
        
        # remove nan, inf and -inf
        keep_idx = np.all(np.isfinite(norm_data), axis=(1, 2))
        if not np.all(keep_idx):
            print('Invalid value encountered...removing from batch')
            
        # add to dict
        out_dict['summary_conditions'] = norm_data[keep_idx]
        out_dict['targets'] = params[keep_idx]
        
        return out_dict
    
    def normalize_input(self, forward_dict: dict) -> dict:
        if self.sim_normalization == NormalizationType.MEAN:
            forward_dict['sim_data'] = self.__mean_norm__(forward_dict['sim_data'], self.mean_sim, self.std_sim)
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
        return forward_dict
    
    def unnorm_params(self, params: np.ndarray) -> np.ndarray:
        if self.param_normalization == NormalizationType.MEAN:
            return self.__mean_unnorm__(params, self.prior_means, self.prior_stds)
        return params

class SIRConfiguratorEtaFlow(AbstractConfigurator):
    def __init__(self, data_sample: dict, prior: AbstractPrior, **kwargs):
        super().__init__(data_sample, prior)
        self.prior = prior
        self.sim_means, self.sim_stds = self.data['sim_data'].mean(axis=(0, 1), keepdims=True), self.data['sim_data'].std(axis=(0, 1), keepdims=True)

        # calculate prior means and stds
        prior_samples = self.prior.sample(1000)
        self.prior_names = self.prior.get_names()
        beta_idx = np.where(self.prior_names == 'beta')[0][0]
        gamma_idx = np.where(self.prior_names == 'gamma')[0][0]
        prior_samples[:, beta_idx] = (np.sqrt(1 + prior_samples[:, beta_idx]) - 1) / prior_samples[:, gamma_idx]
        self.prior_means, self.prior_stds = prior_samples.mean(axis=0, keepdims=True), prior_samples.std(axis=0, keepdims=True)

        try:
            self.sim_normalization = kwargs['normalization']['simulations']
            self.param_normalization = kwargs['normalization']['parameters']
        except KeyError:
            self.sim_normalization = NormalizationType.NONE
            self.param_normalization = NormalizationType.NONE

    def configure_input(self, forward_dict: dict) -> dict:
        """Configures dictionary of prior draws and simulated data into BayesFlow format."""
    
        out_dict = {}
        
        # standardization sim_data
        sim_data = forward_dict['sim_data'].astype(np.float32)
        norm_data = (sim_data - self.sim_means) / self.sim_stds
        
        # standardization priors
        params = forward_dict['prior_draws'].astype(np.float32)
        prior_names = self.prior.get_names()
        # find the index of the parameter beta in prior_names
        beta_idx = np.where(prior_names == 'beta')[0][0]
        gamma_idx = np.where(prior_names == 'gamma')[0][0]
        #params[:, beta_idx] = (np.sqrt(1 + params[:, beta_idx]) - 1) / (1/params[:, gamma_idx])
        params[:, beta_idx] = (np.sqrt(1 + params[:, beta_idx]) - 1) / np.sqrt(params[:, gamma_idx])
        norm_params = (params - self.prior_means) / self.prior_stds
        
        # remove nan, inf and -inf
        keep_idx = np.all(np.isfinite(norm_data), axis=(1, 2))
        if not np.all(keep_idx):
            print('Invalid value encountered...removing from batch')
            
        # add to dict
        out_dict['summary_conditions'] = norm_data[keep_idx]
        out_dict['parameters'] = norm_params[keep_idx]
        
        return out_dict
    
    def normalize_input(self, forward_dict: dict) -> dict:
        if self.sim_normalization == NormalizationType.MEAN:
            forward_dict['sim_data'] = self.__mean_norm__(forward_dict['sim_data'], self.mean_sim, self.std_sim)
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
        return forward_dict
    
    def unnorm_params(self, params: np.ndarray) -> np.ndarray:
        if self.param_normalization == NormalizationType.MEAN:
            return self.__mean_unnorm__(params, self.prior_means, self.prior_stds)
        return params
    
class SIRConfiguratorEtaFlowSqrt(AbstractConfigurator):
    def __init__(self, data_sample: dict, prior: AbstractPrior, **kwargs):
        super().__init__(data_sample, prior)
        self.prior = prior
        self.sim_means, self.sim_stds = self.data['sim_data'].mean(axis=(0, 1), keepdims=True), self.data['sim_data'].std(axis=(0, 1), keepdims=True)

        # calculate prior means and stds
        prior_samples = self.prior.sample(1000)
        self.prior_names = self.prior.get_names()
        beta_idx = np.where(self.prior_names == 'beta')[0][0]
        gamma_idx = np.where(self.prior_names == 'gamma')[0][0]
        prior_samples[:, beta_idx] = (np.sqrt(1 + prior_samples[:, beta_idx]) - 1) / np.sqrt(prior_samples[:, gamma_idx])
        self.prior_means, self.prior_stds = prior_samples.mean(axis=0, keepdims=True), prior_samples.std(axis=0, keepdims=True)

        try:
            self.sim_normalization = kwargs['normalization']['simulations']
            self.param_normalization = kwargs['normalization']['parameters']
        except KeyError:
            self.sim_normalization = NormalizationType.NONE
            self.param_normalization = NormalizationType.NONE

    def configure_input(self, forward_dict: dict) -> dict:
        """Configures dictionary of prior draws and simulated data into BayesFlow format."""
    
        out_dict = {}
        
        # standardization sim_data
        sim_data = forward_dict['sim_data'].astype(np.float32)
        norm_data = (sim_data - self.sim_means) / self.sim_stds
        
        # standardization priors
        params = forward_dict['prior_draws'].astype(np.float32)
        prior_names = self.prior.get_names()
        # find the index of the parameter beta in prior_names
        beta_idx = np.where(prior_names == 'beta')[0][0]
        gamma_idx = np.where(prior_names == 'gamma')[0][0]
        #params[:, beta_idx] = (np.sqrt(1 + params[:, beta_idx]) - 1) / (1/params[:, gamma_idx])
        params[:, beta_idx] = (np.sqrt(1 + params[:, beta_idx]) - 1) / np.sqrt(params[:, gamma_idx])
        norm_params = (params - self.prior_means) / self.prior_stds
        
        # remove nan, inf and -inf
        keep_idx = np.all(np.isfinite(norm_data), axis=(1, 2))
        if not np.all(keep_idx):
            print('Invalid value encountered...removing from batch')
            
        # add to dict
        out_dict['summary_conditions'] = norm_data[keep_idx]
        out_dict['parameters'] = norm_params[keep_idx]
        
        return out_dict
    
    def normalize_input(self, forward_dict: dict) -> dict:
        if self.sim_normalization == NormalizationType.MEAN:
            forward_dict['sim_data'] = self.__mean_norm__(forward_dict['sim_data'], self.mean_sim, self.std_sim)
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
        return forward_dict
    
    def unnorm_params(self, params: np.ndarray) -> np.ndarray:
        if self.param_normalization == NormalizationType.MEAN:
            return self.__mean_unnorm__(params, self.prior_means, self.prior_stds)
        return params
    
class SIRConfiguratorEtaFlowSqrtNoise(AbstractConfigurator):
    def __init__(self, data_sample: dict, prior: AbstractPrior, **kwargs):
        super().__init__(data_sample, prior)
        self.prior = prior
        self.sim_means, self.sim_stds = self.data['sim_data'].mean(axis=(0, 1), keepdims=True), self.data['sim_data'].std(axis=(0, 1), keepdims=True)

        # calculate prior means and stds
        prior_samples = self.prior.sample(1000)
        self.prior_names = self.prior.get_names()
        beta_idx = np.where(self.prior_names == 'beta')[0][0]
        gamma_idx = np.where(self.prior_names == 'gamma')[0][0]
        prior_samples[:, beta_idx] = (np.sqrt(1 + prior_samples[:, beta_idx]) - 1) / np.sqrt(prior_samples[:, gamma_idx])
        self.prior_means, self.prior_stds = prior_samples.mean(axis=0, keepdims=True), prior_samples.std(axis=0, keepdims=True)

        try:
            self.sim_normalization = kwargs['normalization']['simulations']
            self.param_normalization = kwargs['normalization']['parameters']
        except KeyError:
            self.sim_normalization = NormalizationType.NONE
            self.param_normalization = NormalizationType.NONE

    def configure_input(self, forward_dict: dict) -> dict:
        """Configures dictionary of prior draws and simulated data into BayesFlow format."""
    
        out_dict = {}
        
        # standardization sim_data
        sim_data = forward_dict['sim_data'].astype(np.float32)
        norm_data = (sim_data - self.sim_means) / self.sim_stds
        
        # standardization priors
        params = forward_dict['prior_draws'].astype(np.float32)
        prior_names = self.prior.get_names()
        # find the index of the parameter beta in prior_names
        beta_idx = np.where(prior_names == 'beta')[0][0]
        gamma_idx = np.where(prior_names == 'gamma')[0][0]
        #params[:, beta_idx] = (np.sqrt(1 + params[:, beta_idx]) - 1) / (1/params[:, gamma_idx])
        params[:, beta_idx] = (np.sqrt(1 + params[:, beta_idx]) - 1) / np.sqrt(params[:, gamma_idx])
        norm_params = (params - self.prior_means) / self.prior_stds
        norm_params = norm_params + np.random.normal((-1)*self.prior_stds, self.prior_stds)
        
        # remove nan, inf and -inf
        keep_idx = np.all(np.isfinite(norm_data), axis=(1, 2))
        if not np.all(keep_idx):
            print('Invalid value encountered...removing from batch')
            
        # add to dict
        out_dict['summary_conditions'] = norm_data[keep_idx]
        out_dict['parameters'] = norm_params[keep_idx]
        
        return out_dict
    
    def normalize_input(self, forward_dict: dict) -> dict:
        if self.sim_normalization == NormalizationType.MEAN:
            forward_dict['sim_data'] = self.__mean_norm__(forward_dict['sim_data'], self.mean_sim, self.std_sim)
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
        return forward_dict
    
    def unnorm_params(self, params: np.ndarray) -> np.ndarray:
        if self.param_normalization == NormalizationType.MEAN:
            return self.__mean_unnorm__(params, self.prior_means, self.prior_stds)
        return params