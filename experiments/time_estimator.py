## This is a time estimator for the nonthreaded version of the capacity program

import numpy as np

n = 2**20
t = 0.0109*n*np.log2(n)

print(f"time: {t} s OR {t/60} min OR {t/3600} hrs")
