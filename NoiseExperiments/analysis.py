import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
np.set_printoptions(suppress = True,precision = 9)

noise_fraction_candidates = [2, 4, 6, 8, 10]
data = np.array([np.mean(np.mean(np.load("Noise_Metrics" + str(noise_fraction) + ".npy"), axis = 0), axis = 0) for noise_fraction in noise_fraction_candidates])
SBLData = np.array([np.mean(np.mean(np.load("SBLResult" + str(s) + ".npy"), axis = 0), axis = 0) for s in noise_fraction_candidates])

plt.figure(figsize = [7, 7])
plt.plot(noise_fraction_candidates, data[:, 0], label = "AR-LASSO with ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(noise_fraction_candidates, data[:, 1], label = "AR-LASSO without ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(noise_fraction_candidates, data[:, 2], label = "R-LASSO with ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(noise_fraction_candidates, data[:, 3], label = "R-LASSO without ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(noise_fraction_candidates, data[:, 4], label = "L1-HTP", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(noise_fraction_candidates, data[:, 7], label = "L2-HTP", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(noise_fraction_candidates, data[:, 5], label = "L1-L1", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(noise_fraction_candidates, SBLData[:, 0], label = "SBL", marker = "o", linewidth = 3, linestyle = 'dashed')

plt.legend(fontsize = 15)
plt.xlabel("Percentage of measurement noise \n (c)", fontsize = 15)
plt.ylabel("Relative reconstruction error", fontsize = 15)
plt.grid()
plt.savefig("FigureNoiseAnalysis.pdf")

print([np.load("SBLResult" + str(s) + ".npy")[49][49][0] for s in noise_fraction_candidates])