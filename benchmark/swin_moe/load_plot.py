# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# %%
# import matplotlib.pyplot as plt
import numpy as np
scatter_results = np.load("scatter_results.npy", allow_pickle=True)
np.savetxt("scatter_results.csv", scatter_results, delimiter=",", fmt="%s")
# %%
