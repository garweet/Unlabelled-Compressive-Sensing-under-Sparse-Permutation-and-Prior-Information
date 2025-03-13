import numpy as np
np.random.seed(90)

additionalCoefficients = np.random.normal(0, 1, size = (30, 1))
np.save("additionalCoefficients", additionalCoefficients)

