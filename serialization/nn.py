import torch

from typing import Optional
from pydantic import Field
from pydantic_core import core_schema

from .core import ParamManager
from .lambda_ import ModuleWrapper


class SerializableModule(torch.nn.Module, ParamManager):
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        ParamManager.__init__(self, **kwargs)


class TorchDevice:
    @classmethod
    def __get_pydantic_core_schema__(cls, *_):

        def validate(v):
            if not isinstance(v, torch.device):
                raise ValueError(f"{v} is not a torch device")
            return v

        return core_schema.no_info_plain_validator_function(validate)
    

class TorchDType:
    @classmethod
    def __get_pydantic_core_schema__(cls, *_):

        def validate(v):
            if not isinstance(v, torch.dtype):
                raise ValueError(f"{v} is not a torch dtype")
            return v

        return core_schema.no_info_plain_validator_function(validate)


class SerializableModel(SerializableModule):
    device: Optional[str | TorchDevice] = Field("cpu", exclude=True, description="Device to initialize on")
    dtype: TorchDType = Field(torch.float32, exclude=True, description="Data type to use for model parameters")

    def __init__(self, **kwargs):
        if not kwargs.get("device"):
            kwargs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(**kwargs)

    def to(self, *args, **kwargs):
        device, dtype, *_, = torch._C._nn._parse_to(*args, **kwargs)
        self.device = self.device if device is None else device
        self.dtype = self.dtype if dtype is None else dtype
        return super().to(*args, **kwargs)

    def layer_forward(self, layer, *args, use_checkpoint=False, preserve_rng_state=True, use_reentrant=False):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                layer, *args,
                preserve_rng_state=preserve_rng_state,
                use_reentrant=use_reentrant
            )
        return layer(*args)

    def checkpoint_sequential_run(self, func, num_checkpoints, inp):
        if num_checkpoints:
            return checkpoint_sequential(func, num_checkpoints, inp)
        return func(inp)



def __getattr__(name):
    try:
        return ModuleWrapper(getattr(torch.nn, name))
    except:
        raise AttributeError(f"module {torch.nn} has no attribute {name}")
