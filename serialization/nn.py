import warnings
import torch

from pydantic import Field
from torch.utils.checkpoint import checkpoint_sequential

from .core import PMAutoCaster, ParamManager
from .lambda_ import ModuleWrapper, SerializableCallable


class SerializableModule(torch.nn.Module, SerializableCallable):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        SerializableCallable.__init__(self, *args, **kwargs)

    def checkpoint_sequential_run(self, func, num_checkpoints, inp, use_reentrant=False):
        if num_checkpoints:
            return checkpoint_sequential(func, num_checkpoints, inp, use_reentrant=use_reentrant)
        return func(inp)


class TorchDevice(PMAutoCaster):
    @classmethod
    def _PM_auto_cast(cls, v):
        cast_result = v
        if type(v) is str:
            cuda_available = torch.cuda.is_available()
            if "cuda" in v and not cuda_available:
                warnings.warn("Cuda is not available, falling back to cpu")
            cast_result = torch.device(v if cuda_available else "cpu")

        if not isinstance(cast_result, torch.device):
           raise ValueError(f"Cannot convert {v} to torch device")

        return cast_result


class TorchDType(PMAutoCaster):
    @classmethod
    def _PM_auto_cast(cls, v):
        cast_result = v
        if type(v) is str:
            if '.' in v:
                _, v = v.split('.', 1)
            cast_result = getattr(torch, v)

        if not isinstance(cast_result, torch.dtype):
           raise ValueError(f"Cannot convert {v} to torch dtype")

        return cast_result


class DeviceContainer(ParamManager):
    device: TorchDevice = Field("cpu", description="Device to initialize on")
    dtype: TorchDType = Field(torch.float32, description="Data type to use for model parameters")

    def param_model_dump(self, *args, **kwargs):
        dump = super().param_model_dump(*args, **kwargs)
        dump["device"] = str(dump["device"])
        dump["dtype"] = str(dump["dtype"])
        return dump


class SerializableModel(SerializableModule, DeviceContainer):
    device: TorchDevice = Field("cpu", exclude=True, description="Device to initialize on")
    dtype: TorchDType = Field(torch.float32, exclude=True, description="Data type to use for model parameters")

    def param_model_dump(self, *args, **kwargs):
        return SerializableModule.param_model_dump(self, *args, **kwargs)

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


def __getattr__(name):
    try:
        return ModuleWrapper(getattr(torch.nn, name))
    except:
        raise AttributeError(f"module {torch.nn} has no attribute {name}")
