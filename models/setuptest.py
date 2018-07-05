import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle

with open('PositiveAugmented.pickle','rb') as f:
   posdat = pickle.load(f)

with open('NegativeAugmented.pickle','rb') as f:
   negdat = pickle.load(f)
