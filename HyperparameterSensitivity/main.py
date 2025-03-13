from utils import RLASSO_with_constraint, RLASSO_without_constraint, ARLASSO_with_constraint, ARLASSO_without_constraint, L1_HTP, L2_HTP, L1L1, only_priors, sparsePermutation, addNoise, meanNormalizedError 
import numpy as np
import random  
np.random.seed(90)
random.seed(90)

noOfPermutations = 16                                     # number of permutation
cvMeasurements = 5
m = 32                                                    # number of known correspondences
fileName = "results"                                           
noiseFraction = 0.02                                      
permutationRuns, noiseRuns = 1, 1
p = 240                                                   # dimension of x
k = 14                                                    # sparsity of x
N = 120
metrics = np.zeros(shape = (permutationRuns, noiseRuns, np.shape(np.arange(0.001, 0.900, 0.020))[0], np.shape(np.arange(0.001, 0.900, 0.020))[0]))
x = np.load("x_sparsity_" + str(k) + ".npy")
A = np.load("A_fixed.npy")[:N, :]                         # extract N rows from the generated matrix
A_cv =  np.load("A_cv_fixed.npy")  
y_cv = addNoise(A_cv @ x, noiseFraction)    

for permutationRun in range(permutationRuns):

    P = sparsePermutation(N, m, noOfPermutations)
    noiseLess_y = P @ A @ x

    for noiseRun in range(noiseRuns):
        print("Permutation run {}, Noisy run: {}".format(permutationRun, noiseRun))
        noisy_y = addNoise(noiseLess_y, noiseFraction)
        xhat_ARLASSO_with_constraint, lambda1_ARLASSO_with_constraint, lambda2_ARLASSO_with_constraint, t, crossValidationErrors = ARLASSO_with_constraint(A, noisy_y, A_cv, y_cv, m)   
        metrics[permutationRun][noiseRun] = crossValidationErrors
        np.save(fileName + '.npy', metrics)
        
print("Program finished successfully")