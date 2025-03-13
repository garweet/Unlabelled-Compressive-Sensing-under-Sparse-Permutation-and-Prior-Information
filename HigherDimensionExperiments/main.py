from utils import RLASSO_with_constraint, RLASSO_without_constraint, ARLASSO_with_constraint, ARLASSO_without_constraint, L1_HTP, L2_HTP, SBL, L1L1, only_priors, sparsePermutation, addNoise, meanNormalizedError 
import numpy as np
import random  
np.random.seed(90)
random.seed(90)

noOfPermutations = 80                                        # number of permutations (s)
cvMeasurements = 5
m = 60                                                        # number of known correspondences
noiseFraction = 0.02
permutationRuns, noiseRuns = 1, 50
p = 240 * 5                                                   # dimension of beta
k = 14 * 5                                                    # sparsity of beta  (to be set)
N = 530
fileName = str(N)
# NMetrics = np.zeros(shape = (permutationRuns, noiseRuns, 9))
SBLErrors = np.zeros(shape = (permutationRuns, noiseRuns, 1))
x = np.load("x_sparsity_" + str(k) + ".npy")
A = np.load("A_fixed.npy")[:N, :]                             # extract N rows from the generated matrix
A_cv =  np.load("A_cv_fixed.npy")  
y_cv, _ = addNoise(A_cv @ x, noiseFraction)     

for permutationRun in range(permutationRuns):

    while True:
        P = sparsePermutation(N, m, noOfPermutations)
        noiseLess_y = P @ A @ x
        z_true = noiseLess_y - A @ x
        sum_abs_z = np.sum(np.abs(z_true))
        max_abs_z = np.max(np.abs(z_true))
        if abs(sum_abs_z - 660) <= 2:
            break

    lambda1_ARLASSO_with_constraint, lambda2_ARLASSO_with_constraint = None, None
    lambda1_ARLASSO_without_constraint, lambda2_ARLASSO_without_constraint = None, None
    lambda1_RLASSO_with_constraint, lambda2_RLASSO_with_constraint = None, None
    lambda1_RLASSO_without_constraint, lambda2_RLASSO_without_constraint = None, None
    lambda_L1L1 = None
    lambda_only_priors = None

    for noiseRun in range(noiseRuns):
        print("Permutation run {}, Noisy run: {}".format(permutationRun, noiseRun))
        noisy_y, sigma = addNoise(noiseLess_y, noiseFraction)

        # xhat_ARLASSO_with_constraint, lambda1_ARLASSO_with_constraint, lambda2_ARLASSO_with_constraint, _ = ARLASSO_with_constraint(A, noisy_y, A_cv, y_cv, m, lambda1_ARLASSO_with_constraint, lambda2_ARLASSO_with_constraint)
        # xhat_ARLASSO_without_constraint, lambda1_ARLASSO_without_constraint, lambda2_ARLASSO_without_constraint, _ = ARLASSO_without_constraint(A, noisy_y, A_cv, y_cv, m, lambda1_ARLASSO_without_constraint, lambda2_ARLASSO_without_constraint)
        # xhat_RLASSO_with_constraint, lambda1_RLASSO_with_constraint, lambda2_RLASSO_with_constraint, _ = RLASSO_with_constraint(A, noisy_y, A_cv, y_cv, lambda1_RLASSO_with_constraint, lambda2_RLASSO_with_constraint)
        # xhat_RLASSO_without_constraint, lambda1_RLASSO_without_constraint, lambda2_RLASSO_without_constraint, _ = RLASSO_without_constraint(A, noisy_y, A_cv, y_cv, lambda1_RLASSO_without_constraint, lambda2_RLASSO_without_constraint)
        # xhat_L1_HTP, _ = L1_HTP(A, noisy_y, A_cv, y_cv, true_k = k)
        # xhat_L2_HTP, _ = L2_HTP(A, noisy_y, A_cv, y_cv, m, true_k = k, true_s = noOfPermutations)
        # xhat_L1L1, lambda_L1L1 = L1L1(A, noisy_y, A_cv, y_cv, lambda_L1L1)
        # xhat_only_priors, lambda_only_priors = only_priors(A, noisy_y, A_cv, y_cv, m, lambda_only_priors)
        xhat_SBL = SBL(A, noisy_y, A_cv, y_cv, m, sigma)

        # NMetrics[permutationRun][noiseRun][0] = meanNormalizedError(x, xhat_ARLASSO_with_constraint)
        # NMetrics[permutationRun][noiseRun][1] = meanNormalizedError(x, xhat_ARLASSO_without_constraint)
        # NMetrics[permutationRun][noiseRun][2] = meanNormalizedError(x, xhat_RLASSO_with_constraint)
        # NMetrics[permutationRun][noiseRun][3] = meanNormalizedError(x, xhat_RLASSO_without_constraint)
        # NMetrics[permutationRun][noiseRun][4] = meanNormalizedError(x, xhat_L1_HTP)
        # NMetrics[permutationRun][noiseRun][5] = meanNormalizedError(x, xhat_L1L1)
        # NMetrics[permutationRun][noiseRun][6] = meanNormalizedError(x, xhat_only_priors)
        # NMetrics[permutationRun][noiseRun][7] = meanNormalizedError(x, xhat_L2_HTP)
        # NMetrics[permutationRun][noiseRun][8] = sum_abs_z
        SBLErrors[permutationRun][noiseRun] = meanNormalizedError(x, xhat_SBL)
           
        # print(NMetrics[permutationRun][noiseRun][0], NMetrics[permutationRun][noiseRun][1], NMetrics[permutationRun][noiseRun][2], NMetrics[permutationRun][noiseRun][3], NMetrics[permutationRun][noiseRun][4], NMetrics[permutationRun][noiseRun][5], NMetrics[permutationRun][noiseRun][6], NMetrics[permutationRun][noiseRun][7], NMetrics[permutationRun][noiseRun][8])
        print("Error with Bayesian learning: {}".format(SBLErrors[permutationRun][noiseRun][0]))

        # np.save('NMetrics' + fileName + '.npy', NMetrics)
        np.save('SBLResult' + fileName + '.npy', SBLErrors)
        
print("Program finished successfully")