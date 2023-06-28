import os
from bayesrun.data_reader import TumorDataReader
from bayesrun.custom_prior import PriorJagiella2017Delog
from bayesrun.input_configuration import ConfiguratorJagiella2017
from bayesrun.summary_networks import LSTMTransformer
from bayesrun.util.positional_encodings import get_linear_position_encoding
from bayesrun.util.enums import NormalizationType
from bayesflow.experimental.rectifiers import DriftNetwork, RectifiedDistribution

run_name = 'default_run'
storage_dir = None
tensorboard_folder = 'runs'
checkpoint_folder = 'checkpoints'
log_folder = 'logs'
log_to_console = True
remove_folders_of_failed_runs = True

data_reader = {
    'Class': TumorDataReader,
    'test_ratio': 0.1,
    'validation_ratio': 0.1,
    'config': {
        'obs_paths': [os.path.abspath('./data/no_log_observables_1000.pkl')],
        'param_paths': [os.path.abspath('./data/no_log_param_1000.pkl')],
        'profile_depth': 300
        }
    }

prior = {
    'Class': PriorJagiella2017Delog,
    'config': {}
}

configurator = {
    'Class': ConfiguratorJagiella2017,
    'config': {
        'pos_function': get_linear_position_encoding,
        'nr_positional_embeddings': get_linear_position_encoding().shape[-1],
        'normalization': {
            'simulations': NormalizationType.MEAN,
            'parameters': NormalizationType.MEAN
        }
    }
}

summary_net = {
    'Class': LSTMTransformer,
    'config': {
        'n_summary': 16,
        'template_dim': 500,
        'transformer_settings': {
            'input_dim': 2 + configurator['config']['nr_positional_embeddings']
        },
        'attention_settings': {
            'num_heads': 8,
            'key_dim': 10,
            'dense_units': 4,
            'dense_activation': 'relu'
        },
        'dense_concat_settings': {
            'n_layers': 2,
            'activation': 'relu'
        }
    }
}

inference_net = {
    'Class': DriftNetwork,
    'n_layers': 8
}

amortizer = {
    'Class': RectifiedDistribution,
    'config': {}
}

trainer = {}

training = {
    'epochs': 10,
    'batch_size': 64,
    'config': {
        'save_models': True
    }
}

diagnostics = {
    'max_test_data': 100
}