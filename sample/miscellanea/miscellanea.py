#%%
class ProtoTensor(object):
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

proto_tensor = ProtoTensor()
proto_tensor.tags.append(1)

print(proto_tensor.tags)
# %%
