#%%
class FlowTensor(object):
    @property
    def tags(self):
        return getattr(self, "_tags", [])

    def init_flow(self):
        if not hasattr(self, "tags"):
            setattr(self, "tags", [])
        if not hasattr(self, "loads"):
            setattr(self, "loads", [])

    def add_flow(self, tag, load):
        self.tags = tag
        self.loads = load
        return self

flow_tensor = FlowTensor()
flow_tensor.tags.append(1)

print(flow_tensor.tags)
# %%
