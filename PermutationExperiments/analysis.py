import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tex
np.set_printoptions(suppress = True,precision = 9)

candidates_s = [8, 12, 16, 20, 24, 28, 32]
data = np.array([np.mean(np.mean(np.load("s_metrics" + str(s) + ".npy"), axis = 0), axis = 0) for s in candidates_s])
SBLData = np.array([np.mean(np.mean(np.load("SBLResult" + str(s) + ".npy"), axis = 0), axis = 0) for s in candidates_s])

plt.figure(figsize = [7, 7])

plt.plot(candidates_s, data[:, 0], label = "AR-LASSO with ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_s, data[:, 1], label = "AR-LASSO without ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_s, data[:, 2], label = "R-LASSO with ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_s, data[:, 3], label = "R-LASSO without ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_s, data[:, 4], label = "L1-HTP", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_s, data[:, 7], label = "L2-HTP", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_s, data[:, 5], label = "L1-L1", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(candidates_s, SBLData[:, 0], label = "SBL", marker = "o", linewidth = 3, linestyle = 'dashed')

plt.legend(fontsize = 15)
plt.xlabel("Number of permutations s \n (d)", fontsize = 15)
plt.ylabel("Relative reconstruction error", fontsize = 15)
plt.grid()
plt.savefig("FigurePermutationAnalysis.pdf")