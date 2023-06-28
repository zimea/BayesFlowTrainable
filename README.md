# BayesFlow_tumor

## Install
Dependencies in environment.yaml. 
Build and install the python package.

## Run BayesFlow
Implement the reuqired prior in custom_prior.py, the code to read the dataset in data_reader.py, the configuration in input_configuration.py and the summary networks in summary_networks.py.
Adapt the config file.
Run: python3 'path/to/trainer.py' -c 'path/to/configfile'

## Storage
Weights and models are stored in checkpoints, tensorboard details in runs and logs of the run in logs.