import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from IPython.display import clear_output
import random  
import scipy
np.random.seed(90)
random.seed(90)

m = 32
N = 90

permutationMatrices = []

for _ in range(50):
    permuted_entries = random.sample(range(m, N), 56)
    permutationMatrices.append(permuted_entries)

np.save("PermutationMatrices.npy", np.array(permutationMatrices))

for i in range(50):
    print(list(permutationMatrices[i]))

print("Program finished successfully")
