import pickle
import numpy as np
import matplotlib.pyplot as plt

sensitivities = pickle.load(open('C:/Users/CGAO8/Documents/Code/Results0/sensitivities1.pickle', 'rb'))
fpadj = pickle.load(open('C:/Users/CGAO8/Documents/Code/Results0/FPratesAdj1.pickle', 'rb'))
# print(fpadj)
plt.plot(fpadj, sensitivities)
plt.xlabel('Adjusted FPS per scan')
plt.ylabel('Sensitivity')
# plt.show()

aug_sensitivities = pickle.load(open('C:/Users/CGAO8/Documents/Code/Results2/aug_sensitivities1.pickle', 'rb'))
aug_fpadj = pickle.load(open('C:/Users/CGAO8/Documents/Code/Results2/aug_FPratesAdj1.pickle', 'rb'))
# print(aug_fpadj)
plt.plot(aug_fpadj, aug_sensitivities)
# plt.xlabel('Adjusted FPS per scan')
# plt.ylabel('Sensitivity')
plt.legend(['original', 'with augmented'])
plt.show()