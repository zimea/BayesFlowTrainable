import numpy as np

def get_position_encoding(seq_len=1000, d=2, n=10000):
  P = np.zeros((seq_len, d))
  for k in range(seq_len):
      for i in np.arange(int(d/2)):
          denominator = np.power(n, 2*i/d)
          P[k, 2*i] = np.sin(k/denominator)
          P[k, 2*i+1] = np.cos(k/denominator)
  return P

def get_linear_position_encoding(seq_len=1000):
  P = np.expand_dims(np.linspace(0,1,seq_len), axis=-1)
  return P