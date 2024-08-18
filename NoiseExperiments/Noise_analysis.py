from utils import RLASSO_with_constraint, RLASSO_without_constraint, ARLASSO_with_constraint, ARLASSO_without_constraint, L1_HTP, L2_HTP, L1L1, only_priors, sparsePermutation, addNoise, meanNormalizedError 
import numpy as np
import random  
np.random.seed(90)
random.seed(90)

noOfPermutations = 16                                     # number of permutation
cvMeasurements = 5
m = 32                                                    # number of known correspondences
fileName = "4"
noiseFraction = 0.04
permutationRuns, noiseRuns = 50, 50
p = 240                                                   # dimension of x
k = 14                                                    # sparsity of x
N = 110
NoiseMetrics = np.zeros(shape = (permutationRuns, noiseRuns, 9))
x = np.load("x_sparsity_" + str(k) + ".npy")
A = np.load("A_fixed.npy")[:N, :]                         # extract N rows from the generated matrix
A_cv =  np.load("A_cv_fixed.npy")  
y_cv = addNoise(A_cv @ x, noiseFraction)     

for permutationRun in range(permutationRuns):
    P = sparsePermutation(N, m, noOfPermutations)
    noiseLess_y = P @ A @ x
    print(noiseLess_y[105:, :])
    z_true = noiseLess_y - A @ x
    max_abs_z = np.max(np.abs(z_true))
    sum_abs_z = np.sum(np.abs(z_true))

    lambda1_ARLASSO_with_constraint, lambda2_ARLASSO_with_constraint = None, None
    lambda1_ARLASSO_without_constraint, lambda2_ARLASSO_without_constraint = None, None
    lambda1_RLASSO_with_constraint, lambda2_RLASSO_with_constraint = None, None
    lambda1_RLASSO_without_constraint, lambda2_RLASSO_without_constraint = None, None
    lambda_L1L1 = None
    lambda_only_priors = None

    for noiseRun in range(noiseRuns):
        print("Permutation run {}, Noisy run: {}".format(permutationRun, noiseRun))
        noisy_y = addNoise(noiseLess_y, noiseFraction)

        xhat_ARLASSO_with_constraint, l1_ARLASSO_with_constraint, l2_ARLASSO_with_constraint, _ = ARLASSO_with_constraint(A, noisy_y, A_cv, y_cv, m, lambda1_ARLASSO_with_constraint, lambda2_ARLASSO_with_constraint)
        xhat_ARLASSO_without_constraint, l1_ARLASSO_without_constraint, l2_ARLASSO_without_constraint, _ = ARLASSO_without_constraint(A, noisy_y, A_cv, y_cv, m, lambda1_ARLASSO_without_constraint, lambda2_ARLASSO_without_constraint)
        xhat_RLASSO_with_constraint, l1_RLASSO_with_constraint, l2_RLASSO_with_constraint, _ = RLASSO_with_constraint(A, noisy_y, A_cv, y_cv, lambda1_RLASSO_with_constraint, lambda2_RLASSO_with_constraint)
        xhat_RLASSO_without_constraint, l1_RASSO_without_constraint, l2_RLASSO_without_constraint, _ = RLASSO_without_constraint(A, noisy_y, A_cv, y_cv, lambda1_RLASSO_without_constraint, lambda2_RLASSO_without_constraint)
        xhat_L1_HTP, _ = L1_HTP(A, noisy_y, A_cv, y_cv)
        xhat_L2_HTP, _ = L2_HTP(A, noisy_y, A_cv, y_cv, m)
        xhat_L1L1, lambda_L1L1 = L1L1(A, noisy_y, A_cv, y_cv, lambda_L1L1)
        xhat_only_priors, lambda_only_priors = only_priors(A, noisy_y, A_cv, y_cv, m, lambda_only_priors)
  
        NoiseMetrics[permutationRun][noiseRun][0] = meanNormalizedError(x, xhat_ARLASSO_with_constraint)
        NoiseMetrics[permutationRun][noiseRun][1] = meanNormalizedError(x, xhat_ARLASSO_without_constraint)
        NoiseMetrics[permutationRun][noiseRun][2] = meanNormalizedError(x, xhat_RLASSO_with_constraint)
        NoiseMetrics[permutationRun][noiseRun][3] = meanNormalizedError(x, xhat_RLASSO_without_constraint)
        NoiseMetrics[permutationRun][noiseRun][4] = meanNormalizedError(x, xhat_L1_HTP)
        NoiseMetrics[permutationRun][noiseRun][5] = meanNormalizedError(x, xhat_L1L1)
        NoiseMetrics[permutationRun][noiseRun][6] = meanNormalizedError(x, xhat_only_priors)
        NoiseMetrics[permutationRun][noiseRun][7] = meanNormalizedError(x, xhat_L2_HTP)
        NoiseMetrics[permutationRun][noiseRun][8] = sum_abs_z
           
        print(NoiseMetrics[permutationRun][noiseRun][0], NoiseMetrics[permutationRun][noiseRun][1], NoiseMetrics[permutationRun][noiseRun][2], NoiseMetrics[per_run][noiseRun][3], NoiseMetrics[permutationRun][noiseRun][4], NoiseMetrics[permutationRun][noiseRun][5], NoiseMetrics[permutationRun][noiseRun][6], NoiseMetrics[permutationRun][noiseRun][7], NoiseMetrics[permutationRun][noiseRun][8])
        print(np.load("Noise_Metrics2.npy")[permutationRun][noiseRun])
        # np.save('Noise_Metrics' + fileName + '.npy', NoiseMetrics)
        
print("Program finished succesfully")