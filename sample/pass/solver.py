# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import cvxpy as cp
print(cp.installed_solvers())

#%%
x = cp.Variable(2)
obj = cp.Minimize(x[0] + cp.norm(x, 1))
constraints = [x >= 2]
prob = cp.Problem(obj, constraints)


# Solve with GUROBI.
prob.solve(solver=cp.GUROBI)
print("optimal value with GUROBI:", prob.value)

# %%
