import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import clear_output
np.set_printoptions(suppress = True, precision = 9)

NCandidates = [530, 540, 550, 560]
data = np.array([np.mean(np.mean(np.load("NMetrics" + str(N) + ".npy"), axis = 0), axis = 0) for N in NCandidates])
SBLData = np.array([np.mean(np.mean(np.load("SBLResult" + str(N) + ".npy"), axis = 0), axis = 0) for N in NCandidates])

plt.figure(figsize = [7, 7])
plt.plot(NCandidates, data[:, 0], label = "AR-LASSO with ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(NCandidates, data[:, 1], label = "AR-LASSO without ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(NCandidates, data[:, 2], label = "R-LASSO with ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(NCandidates, data[:, 3], label = "R-LASSO without ZSC", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(NCandidates, data[:, 4], label = "L1-HTP", marker = "o", linewidth = 3, linestyle = 'dashed')
plt.plot(NCandidates, data[:, 7], label = "L2-HTP", marker = "o", linewidth = 3, linestyle = 'dashed')
# plt.plot(NCandidates, data[:, 5], label = "L1-L1", marker = "o", linewidth = 3
plt.plot(NCandidates, SBLData[:, 0], label = "SBL", marker = "o", linewidth = 3, color = "gray", linestyle = 'dashed')

plt.legend(fontsize = 15)
plt.xlabel("Number of measurements N \n (b)", fontsize = 15)
plt.ylabel("Relative reconstruction error", fontsize = 15)
plt.grid()
plt.savefig("FigureHighDimensionAnalysis.pdf")



