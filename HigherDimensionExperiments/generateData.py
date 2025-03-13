import numpy as np
np.random.seed(90)

p = 240 * 5
k = 14 * 5

x_sparsity_70 = np.concatenate((np.random.normal(0, 1, size = (k, 1)), np.zeros(shape = (p - k, 1))), axis = 0)
np.save("x_sparsity_70", x_sparsity_70)

A = np.random.normal(0, 1, size = (5000, p))
np.save("A_fixed", A)

A_cv = np.random.normal(0, 1, size = (5, p))
np.save("A_cv_fixed", A_cv)
