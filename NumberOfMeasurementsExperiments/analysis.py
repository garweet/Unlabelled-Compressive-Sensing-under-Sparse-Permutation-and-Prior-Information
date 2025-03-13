import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tex
np.set_printoptions(suppress = True, precision = 9)

candidates_N = [80, 90, 100, 110, 120]
data = np.array([np.mean(np.mean(np.load("N_metrics" + str(N) + ".npy"), axis = 0), axis = 0) for N in candidates_N])
SBLData = np.array([np.mean(np.mean(np.load("SBLResult" + str(N) + ".npy"), axis = 0), axis = 0) for N in candidates_N])

plt.figure(figsize = [7, 7])

plt.plot(candidates_N, data[:, 0], label = "AR-LASSO with ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_N, data[:, 1], label = "AR-LASSO without ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_N, data[:, 2], label = "R-LASSO with ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_N, data[:, 3], label = "R-LASSO without ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_N, data[:, 4], label = "L1-HTP", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_N, data[:, 7], label = "L2-HTP", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_N, data[:, 5], label = "L1-L1", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_N, SBLData[:, 0], label = "SBL", marker = "o", linewidth = 3, linestyle = 'dashed')

plt.legend(fontsize = 15)
plt.xlabel("Number of measurements N \n (a)", fontsize = 15)
plt.ylabel("Relative reconstruction error", fontsize = 15)
plt.grid()
plt.savefig("FigureNAnalysis.pdf")
