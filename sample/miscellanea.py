# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import pathlib

abs_path = pathlib.Path(__file__)
print(abs_path)
root_path = pathlib.Path("/home/whcui/brainstorm_project/brainstorm")
rel_path = abs_path.relative_to(root_path)
print(rel_path.parts[0])

# %%
