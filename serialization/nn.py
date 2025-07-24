from torch import nn

from .core import ParamManager
from .lambda_ import ModuleWrapper

class SerializableModule(nn.Module, ParamManager):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        ParamManager.__init__(self, **kwargs)

def __getattr__(name):
    try:
        return ModuleWrapper(getattr(nn, name))
    except:
        raise AttributeError(f"module {nn} has no attribute {name}")
