import os
from types import ModuleType
from typing import Type
from bayesflow.trainers import Trainer
from bayesrun.custom_prior import AbstractPrior
from bayesrun.util.custom_types import dataset
from bayesrun.input_configuration import AbstractConfigurator
from bayesrun.data_reader import AbstractDataReader
import tensorflow as tf
from tensorflow.keras.models import load_model

class BayesFlowInterface:
    def __init__(self, config: ModuleType, checkpoint_dir: str, tensorboard_dir: str):
        """Initializes the BayesFlowInterface.
        Args:
            config (ModuleType): The config module.
            storage_dir (str, optional): The storage directory. Defaults to None.
            workdir (str, optional): The working directory. Defaults to None.
        """
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir

    def load_run(self) -> None:
        """Loads a run from the storage directory."""
        assert os.path.isdir(self.checkpoint_dir), "Checkpoint directory does not exist"
        assert os.path.isdir(self.tensorboard_dir), "Tensorboard directory does not exist"
        self.initialize_components(load_trained_model = True)

    def initialize_components(self, load_trained_model: bool = False) -> None: # TODO: test load_trained_model
        """Initializes the components of the BayesFlowInterface.
        Args:
            load_trained_model (bool, optional): Whether to load a trained model. Defaults to False.
        Raises:
            AttributeError: If the config file does not provide teh required configs."""

        config = self.config

        # read data
        try:
            DataReader = config.data_reader['Class']
            data_config = config.data_reader['config']
            test_ratio = config.data_reader['test_ratio']
            validation_ratio = config.data_reader['validation_ratio']
        except AttributeError as e:
            raise AttributeError(
                "Please provide a data reader class, data config and test and validation ratio in the config file. \n" + str(e)
            )
        print("Reading data\n")
        self.data_reader = DataReader()
        self.train, self.test, self.validation = self.data_reader.get_data(test_ratio=test_ratio, validation_ratio=validation_ratio, **data_config)

        # initialize prior
        print("Initializing prior\n")
        try:
            self.prior = config.prior['Class'](**config.prior['config'])
        except AttributeError as e:
            raise AttributeError("Please provide a prior class in the config file. \n" + str(e))
        
        # initialize configutator
        print("Initializing configurator\n")
        try:
            self.configurator = config.configurator['Class'](data_sample=self.train, prior=self.prior, **config.configurator['config'])
        except AttributeError as e:
            raise AttributeError("Please provide a configurator class and required configs in the config file. \n" + str(e))

        # initialize summary network
        print("Initializing summary network\n")
        try:
            if load_trained_model:
                self.summary_net = load_model(os.path.join(self.checkpoint_dir,'summaryNet'))
            else:
                self.summary_net = config.summary_net['Class'](**config.summary_net['config'])
        except AttributeError as e:
            raise AttributeError("Please provide a summary network class and required configs in the config file. \n" + str(e))

        # initialize invertible network
        print("Initializing invertible network\n")
        try:
            if load_trained_model:
                self.inference_net = load_model(os.path.join(self.checkpoint_dir,'inferenceNet'))
            else:
                self.inference_net = config.inference_net['Class'](target_dim=self.prior.get_names().shape[0], num_dense=config.inference_net['n_layers'], **config.inference_net['config'])
        except AttributeError as e:
            raise AttributeError("Please provide an inference network class and required configs in the config file. \n" + str(e))

        # initialize amortizer
        print("Initializing amortizer\n")
        try:
            self.amortizer = config.amortizer['Class'](self.inference_net, self.summary_net, name=config.run_name) 
        except AttributeError as e:
            raise AttributeError("Please provide an amortizer class and required configs in the config file. \n" + str(e))    
        
        self.trainer = Trainer(amortizer=self.amortizer, configurator=self.configurator.configure_input, checkpoint_path=self.checkpoint_dir, **config.trainer)

    def save_networks(self) -> None:
        """Saves the summary and inference network."""
        self.summary_net.save(os.path.join(self.checkpoint_dir,'summaryNet'))
        self.inference_net.save(os.path.join(self.checkpoint_dir,'inferenceNet'))
        print("Summary and inference network saved\n")
    
    def get_trainer(self) -> Type[Trainer]:
        assert self.trainer is not None, "Please run initialize_components first"
        return self.trainer
    
    def get_prior(self) -> Type[AbstractPrior]:
        assert self.prior is not None, "Please run initialize_components first"
        return self.prior
    
    def get_data(self) -> dataset:
        assert self.train is not None, "Please run initialize_components first"
        return self.train, self.test, self.validation
    
    def get_configurator(self) -> Type[AbstractConfigurator]:
        assert self.configurator is not None, "Please run initialize_components first"
        return self.configurator
    
    def get_summary_net(self) -> tf.keras.Model:
        assert self.summary_net is not None, "Please run initialize_components first"
        return self.summary_net
    
    def get_inference_net(self) -> tf.keras.Model:
        assert self.inference_net is not None, "Please run initialize_components first"
        return self.inference_net
    
    def get_amortizer(self):
        assert self.amortizer is not None, "Please run initialize_components first"
        return self.amortizer
    
    def det_data_reader(self) -> Type[AbstractDataReader]:
        assert self.data_reader is not None, "Please run initialize_components first"
        return self.data_reader
    