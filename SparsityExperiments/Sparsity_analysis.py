from utils import RLASSO_with_constraint, RLASSO_without_constraint, ARLASSO_with_constraint, ARLASSO_without_constraint, L1_HTP, L2_HTP, L1L1, only_priors, sparsePermutation, addNoise, meanNormalizedError 
import numpy as np
import random  
np.random.seed(90)
random.seed(90)

noOfPermutations = 16                                     # number of permutation
cvMeasurements = 5
m = 32                                                    # number of known correspondences
noiseFraction = 0.20
permutationRuns, noiseRuns = 50, 50
p = 240                                                   # dimension of x
k = 18                                                    # sparsity of x  (to be set)
N = 110
fileName = str(k)
SparsityMetrics = np.zeros(shape = (permutationRuns, noiseRuns, 9))
x = np.load("x_sparsity_" + str(14) + ".npy")
x[14 : k, :] = np.copy(np.load("additionalCoefficients" + ".npy")[: (k - 14), :])
A = np.load("A_fixed.npy")[:N, :]                         # extract N rows from the generated matrix
A_cv =  np.load("A_cv_fixed.npy")  
y_cv = addNoise(A_cv @ x, noiseFraction)     

for permutationRun in range(permutationRuns):

    while True:
        P = sparsePermutation(N, m, noOfPermutations)
        noiseLess_y = P @ A @ x
        z_true = noiseLess_y - A @ x
        sum_abs_z = np.sum(np.abs(z_true))
        max_abs_z = np.max(np.abs(z_true))
        if abs(sum_abs_z - 50) <= 2:
            break

    lambda1_ARLASSO_with_constraint, lambda2_ARLASSO_with_constraint = None, None
    lambda1_ARLASSO_without_constraint, lambda2_ARLASSO_without_constraint = None, None
    lambda1_RLASSO_with_constraint, lambda2_RLASSO_with_constraint = None, None
    lambda1_RLASSO_without_constraint, lambda2_RLASSO_without_constraint = None, None
    lambda_L1L1 = None
    lambda_only_priors = None

    for noiseRun in range(noiseRuns):
        print("Permutation run {}, Noisy run: {}".format(permutationRun, noiseRun))
        noisy_y = addNoise(noiseLess_y, noiseFraction)

        xhat_ARLASSO_with_constraint, lambda1_ARLASSO_with_constraint, lambda2_ARLASSO_with_constraint, _ = ARLASSO_with_constraint(A, noisy_y, A_cv, y_cv, m, lambda1_ARLASSO_with_constraint, lambda2_ARLASSO_with_constraint)
        xhat_ARLASSO_without_constraint, lambda1_ARLASSO_without_constraint, lambda2_ARLASSO_without_constraint, _ = ARLASSO_without_constraint(A, noisy_y, A_cv, y_cv, m, lambda1_ARLASSO_without_constraint, lambda2_ARLASSO_without_constraint)
        xhat_RLASSO_with_constraint, lambda1_RLASSO_with_constraint, lambda2_RLASSO_with_constraint, _ = RLASSO_with_constraint(A, noisy_y, A_cv, y_cv, lambda1_RLASSO_with_constraint, lambda2_RLASSO_with_constraint)
        xhat_RLASSO_without_constraint, lambda1_RLASSO_without_constraint, lambda2_RLASSO_without_constraint, _ = RLASSO_without_constraint(A, noisy_y, A_cv, y_cv, lambda1_RLASSO_without_constraint, lambda2_RLASSO_without_constraint)
        xhat_L1_HTP, _ = L1_HTP(A, noisy_y, A_cv, y_cv)
        xhat_L2_HTP, _ = L2_HTP(A, noisy_y, A_cv, y_cv, m)
        xhat_L1L1, lambda_L1L1 = L1L1(A, noisy_y, A_cv, y_cv, lambda_L1L1)
        xhat_only_priors, lambda_only_priors = only_priors(A, noisy_y, A_cv, y_cv, m, lambda_only_priors)
  
        SparsityMetrics[permutationRun][noiseRun][0] = meanNormalizedError(x, xhat_ARLASSO_with_constraint)
        SparsityMetrics[permutationRun][noiseRun][1] = meanNormalizedError(x, xhat_ARLASSO_without_constraint)
        SparsityMetrics[permutationRun][noiseRun][2] = meanNormalizedError(x, xhat_RLASSO_with_constraint)
        SparsityMetrics[permutationRun][noiseRun][3] = meanNormalizedError(x, xhat_RLASSO_without_constraint)
        SparsityMetrics[permutationRun][noiseRun][4] = meanNormalizedError(x, xhat_L1_HTP)
        SparsityMetrics[permutationRun][noiseRun][5] = meanNormalizedError(x, xhat_L1L1)
        SparsityMetrics[permutationRun][noiseRun][6] = meanNormalizedError(x, xhat_only_priors)
        SparsityMetrics[permutationRun][noiseRun][7] = meanNormalizedError(x, xhat_L2_HTP)
        SparsityMetrics[permutationRun][noiseRun][8] = sum_abs_z
           
        print(SparsityMetrics[permutationRun][noiseRun][0], SparsityMetrics[permutationRun][noiseRun][1], SparsityMetrics[permutationRun][noiseRun][2], SparsityMetrics[permutationRun][noiseRun][3], SparsityMetrics[permutationRun][noiseRun][4], SparsityMetrics[permutationRun][noiseRun][5], SparsityMetrics[permutationRun][noiseRun][6], SparsityMetrics[permutationRun][noiseRun][7], SparsityMetrics[permutationRun][noiseRun][8])
        np.save('SparsityMetrics' + fileName + '.npy', SparsityMetrics)
        
print("Program finished successfully")