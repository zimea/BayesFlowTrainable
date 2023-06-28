import os, sys
import argparse
import importlib.util
from bayesflow.trainers import Trainer
import bayesflow.diagnostics as diag
import datetime
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from shutil import copy, rmtree
from bayesrun.util.logger import Logger
import contextlib
import traceback
from bayesrun.bayesflow_interface import BayesFlowInterface

parser = argparse.ArgumentParser(
    prog="ExperimentRunner",
    description="Starts individual experiments and logs results",
)

parser.add_argument(
    "-c", "--configfile", type=str, help="path to the config file", default="config.py"
)
parser.add_argument(
    "-s", "--storage_dir", type=str, help="path to the storage directory", default="."
)

# load config file
args = parser.parse_args()
configfile = os.path.abspath(args.configfile)
spec = importlib.util.spec_from_file_location("config", configfile)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

# get storage and initialize log directory
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = config.run_name
if hasattr(args, 'storage_dir'):
    storage_dir = os.path.abspath(args.storage_dir)
else:
    storage_dir = os.path.abspath(os.getcwd())

try:
    log_folder = config.log_folder
except AttributeError:
    log_folder = 'logs'
logdir = os.path.abspath(os.path.join(storage_dir, log_folder, run_name, now))
os.makedirs(logdir)
assert(os.path.isdir(logdir))

if __name__ == "__main__":
    try:            
        sys.stdout = Logger(os.path.join(logdir, "log_training" + ".txt"), log_console=config.log_to_console)

        print("Config file: {}\n".format(configfile))

        # Initialize tensorboard and tensorflow logs
        tensorboard_folder = config.tensorboard_folder if hasattr(config, 'tensorboard_folder') else 'runs'
        checkpoint_folder = config.checkpoint_folder if hasattr(config, 'checkpoint_folder') else 'checkpoints'
        tensorboard_dir = os.path.abspath(os.path.join(storage_dir, tensorboard_folder, run_name, now))
        checkpoint_dir = os.path.abspath(os.path.join(storage_dir, checkpoint_folder, run_name, now))
        print("Tensorboard directory: {}\n".format(tensorboard_dir))
        print("Checkpoint directory: {}\n".format(checkpoint_dir))
        writer = SummaryWriter(tensorboard_dir)
        os.makedirs(checkpoint_dir)
        assert(os.path.isdir(checkpoint_dir))
        copy(configfile, os.path.join(checkpoint_dir, "config.py"))

        # Initialize bayesflow interface TODO: prettier solution
        bayesflow_interface = BayesFlowInterface(config, checkpoint_dir=checkpoint_dir)
        bayesflow_interface.initialize_components()
        train, test, validation = bayesflow_interface.get_data()
        trainer, configurator, prior = bayesflow_interface.get_trainer(), bayesflow_interface.get_configurator(), bayesflow_interface.get_prior()
        summary_net, inference_net, amortizer = bayesflow_interface.get_summary_net(), bayesflow_interface.get_inference_net(), bayesflow_interface.get_amortizer()

        # run training
        print("Starting training\n")
        try:
            epochs = config.training['epochs']
            batch_size = config.training['batch_size']
        except AttributeError:
            raise AttributeError("Please provide a training config in the config file")

        sys.stdout.pause()
        start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        losses = trainer.train_offline(simulations_dict=train, epochs=epochs, batch_size=batch_size, validation_sims=validation, **config.training['config'])
        end = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        sys.stdout.resume()
        print("Training finished\n")
        print("Training started at {} and finished at {}\n".format(start, end))
        if config.training['config']['save_models']:
            bayesflow_interface.save_networks()
        
        # plot model summary
        print("Amortizer summary:\n")
        print(str(amortizer.summary()) + "\n")

        # plot diagnostics
        print("Plotting diagnostics\n")
        fig = diag.plot_losses(losses['train_losses'], losses['val_losses'])
        writer.add_figure('Loss', plt.gcf())

        # run diagnostics on test data
        print("Running diagnostics on test data\n")
        param_names = prior.get_names()
        if 'max_test_data' in config.diagnostics:
            for key in test.keys():
                test[key] = test[key][:config.diagnostics['max_test_data']]
        
        test_data_config = trainer.configurator(test)
        posterior_samples = amortizer.sample(test_data_config, n_samples=1000)
        posterior_samples_unnorm = configurator.unnorm_params(posterior_samples)

        # plot test data calibration
        fig = diag.plot_sbc_ecdf(posterior_samples, test_data_config['targets'], param_names=param_names, stacked=True)
        writer.add_figure('SBC_ECDF_Test', plt.gcf())

        # plot recovery
        fig = diag.plot_recovery(posterior_samples, test_data_config['targets'], param_names=param_names)
        writer.add_figure('Recovery', plt.gcf())

    except Exception as e:
        sys.stderr = Logger(os.path.join(logdir, "log_training" + ".err"), sys.stderr, log_console=config.log_to_console)
        sys.stderr.write("An error occurred: \n")
        sys.stderr.write(str(e))
        sys.stderr.write(traceback.format_exc())
        if config.remove_folders_of_failed_runs:
            rmtree(checkpoint_dir)
            rmtree(tensorboard_dir)
