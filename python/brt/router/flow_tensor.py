import torch


class FlowTensor(object):
    CHECK_TAGS = False

    def __init__(self, data: torch.Tensor, tag: torch.Tensor, load):
        self.data = data
        self.tag = tag
        self.load = load
    
    def numel(self):
        return self.data.numel()

    def __repr__(self):
        return f"FlowTensor:\ndata: {self.data}\ntags: {self.tag}\nload: {self.load}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        new_args = []
        new_tags = []
        new_loads = []
        for i, arg in enumerate(args):
            new_args.append(arg.data if hasattr(arg, "data") else arg)
            if hasattr(arg, "tag"):
                new_tags.append(arg.tag)
            if hasattr(arg, "load"):
                new_loads.append(arg.load)
        assert len(new_tags) > 0 and len(new_loads) > 0
        ret = func(*new_args, **kwargs)
        ret_flow_tensor = FlowTensor(ret, tag=new_tags[0], load=new_loads[0])
        if cls.CHECK_TAGS:
            for tag in new_tags:
                assert torch.allclose(ret_flow_tensor.tag, tag)
        return ret_flow_tensor
