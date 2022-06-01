
#%%
def wrapper(cls):
    print("wrapping the Inherited class")
    cls._brt = True
    return cls


class BaseClass:
    def __init__(self) -> None:
        pass

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls = wrapper(cls)


class InheritClass(BaseClass):
    def __init__(self) -> None:
        super().__init__()


a = InheritClass()
print(a._brt)

# %%
