import tensorflow as tf
from bayesflow.networks import TimeSeriesTransformer

class LSTMTransformer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(LSTMTransformer, self).__init__()
        try:
            n_summary = kwargs['n_summary']
            template_dim = kwargs['template_dim']
            transformer_settings = {
                'input_dim': kwargs['transformer_settings']['input_dim']
            }
            attention_settings = {
                'num_heads': kwargs['attention_settings']['num_heads'],
                'key_dim': kwargs['attention_settings']['key_dim'],
                'dense_units': kwargs['attention_settings']['dense_units'],
                'dense_activation': kwargs['attention_settings']['dense_activation']
            }
            dense_concat_settings = {
                'n_layers': kwargs['dense_concat_settings']['n_layers'],
                'activation': kwargs['dense_concat_settings']['activation']
            }
        except KeyError:
            raise KeyError('Missing settings for attention network or dense concatenation network.')

        self.lstm = tf.keras.layers.LSTM(n_summary)
        self.att_dict = {'num_heads': attention_settings['num_heads'], 'key_dim': attention_settings['key_dim']}
        self.dense_dict = {'units': dense_concat_settings['n_layers'], 'activation': dense_concat_settings['activation']}

        self.transformer = TimeSeriesTransformer(
            input_dim=transformer_settings['input_dim'],
            attention_settings=self.att_dict,
            dense_settings=self.dense_dict,
            summary_dim=n_summary,
            template_dim=template_dim
        )

        dense_layers = []
        for l in range(dense_concat_settings['n_layers']):
            dense_layers.append(tf.keras.layers.Dense(n_summary*(dense_concat_settings['n_layers']-l), activation='relu'))
        self.dense = tf.keras.Sequential(dense_layers)

    def call(self, summary_conditions):
            pp_ecmp, growth = summary_conditions
            out1 = self.lstm(growth)
            out2 = self.transformer(pp_ecmp)
            out = tf.concat([out1, out2], axis=-1)
            out = self.dense(out)
            return out