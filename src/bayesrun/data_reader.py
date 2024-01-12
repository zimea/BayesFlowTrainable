from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from BayesFlowTrainable.src.bayesrun.util.custom_types import dataset
from bayesrun.util.custom_types import dataset

class AbstractDataReader(ABC):
    @abstractmethod
    def get_data(self, test_ratio: float = 0.2, validation_ratio: float = 0.2, **kwargs) -> dataset:
        pass

class TumorDataReader(AbstractDataReader):
    def get_data(self, test_ratio: float = 0.2, validation_ratio: float = 0.2, **kwargs) -> dataset:

        # run checks
        try:
            obs_paths = kwargs['obs_paths']
            param_paths = kwargs['param_paths']
        except KeyError:
            raise KeyError("Please provide paths to the data as a list of strings under the keys 'obs_paths' and 'param_paths'")        

        assert(len(obs_paths) == len(param_paths))

        # get additional config
        if 'profile_depth' in kwargs.keys():
            profile_depth = kwargs['profile_depth']
        else:
            profile_depth = 1000 # default value of the simulations

        # load data
        observables = []
        params = []
        for i in range(len(obs_paths)):
            observables.append(pd.read_pickle(obs_paths[i]))
            params.append(pd.read_pickle(param_paths[i]))
        obs = np.stack(pd.concat(observables).to_numpy())
        tumor_size = np.stack(obs[:,0])[:, :, None]
        radial_features = np.stack([np.stack(obs[:,1]),np.stack(obs[:,2])], axis=-1)[:,:profile_depth,:]
        params = pd.concat(params).to_numpy()


        # TODO: add permutation of data  
        test_split = int(test_ratio * params.shape[0])
        validation_split = int(validation_ratio * params.shape[0])
        train_split = params.shape[0] - test_split - validation_split
             
        train = {"prior_draws": params[:train_split], "sim_data": radial_features[:train_split], 'growth_curve': tumor_size[:train_split]}
        test = {"prior_draws": params[train_split:test_split + train_split], "sim_data": radial_features[train_split:test_split + train_split], 'growth_curve': tumor_size[train_split:test_split + train_split]}
        validation = {"prior_draws": params[test_split + train_split:], "sim_data": radial_features[test_split + train_split:], 'growth_curve': tumor_size[test_split + train_split:]}

        return train, test, validation
    
class MorpheusTumorDataReader(AbstractDataReader):
    def get_data(self, test_ratio: float = 0.2, validation_ratio: float = 0.2, **kwargs) -> dataset:
        
        # run checks
        try:
            path = kwargs['path']
        except KeyError:
            raise KeyError("Please provide path to the data as a string under the key 'path'")

        # load data
        data = pd.read_pickle(path)

        # get additional config
        if 'profile_depth' in kwargs.keys():
            profile_depth = kwargs['profile_depth']
        else:
            profile_depth = data['ecm'].shape[1]

        # remove all lines from data where data[growth].shape[0] < 19
        data = data[data['growth'].apply(lambda x: x.shape[0] >= 20)]
        growth_curve = np.expand_dims(np.stack(data['growth'].to_numpy()), axis=-1)
        ecm = np.stack(data['ecm'].to_numpy())
        profile = np.stack(data['profile'].to_numpy())
        sim_data = np.concatenate([np.expand_dims(ecm, axis=-1), np.expand_dims(profile, axis=-1)], axis=-1)[:,:profile_depth,:]
        param_columns = data.columns[3:]
        params = data[param_columns].to_numpy()

        # TODO: add permutation of data  
        test_split = int(test_ratio * params.shape[0])
        validation_split = int(validation_ratio * params.shape[0])
        train_split = params.shape[0] - test_split - validation_split

        train = {"prior_draws": params[:train_split], "sim_data": sim_data[:train_split], 'growth_curve': growth_curve[:train_split]}
        test = {"prior_draws": params[train_split:test_split + train_split], "sim_data": sim_data[train_split:test_split + train_split], 'growth_curve': growth_curve[train_split:test_split + train_split]}
        validation = {"prior_draws": params[test_split + train_split:], "sim_data": sim_data[test_split + train_split:], 'growth_curve': growth_curve[test_split + train_split:]}

        return train, test, validation
