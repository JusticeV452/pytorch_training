import os
import gc
import json
import random

import tqdm
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from datetime import datetime
from pydantic import Field
from typing import Optional, List, Any, Tuple, Type
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import RandomSampler

from components.gradient_accumulation import BatchNormAccumulator
from components.loss_manager import LossManager
from image_utils import tensor_to_img, write_rgb
from loss.diversity import DIVERSITY_FUNCS, FeatureDiversityLoss, DummyFDL, compute_kid, compute_fid
from loss.mixup import mixup_data, mixup_bce
from pruning import FineGrainedPruner, DummyPruner
from utils import calc_model_size, count_parameters, percent_chance, not_none, shuffle_tensor
from sampling import RandResBatchSampler
from serialization import AutoLambda, Lambda, ParamManager, SerializableCallable
from serialization.nn import DeviceContainer, TorchDType, TorchDevice

# Flag
use_multiprocess = True

if use_multiprocess:
    import multiprocess
    multiprocess.set_start_method('spawn', force=True)
    # hack here.
    torch.utils.data.dataloader.python_multiprocessing = multiprocess
    new_multiprocess_ctx = multiprocess.get_context()
else:
    new_multiprocess_ctx = None


class ParamEncoder(json.JSONEncoder):
    def default(self, o):
        return repr(o)


def loss_dict_to_cpu(loss_dict):
    on_gpu = {}
    tensors = []
    for key, val in loss_dict.items():
        if isinstance(val, torch.Tensor):
            on_gpu[key] = len(tensors)
            tensors.append(val.view(1))
    cpu_tensors = torch.cat(tensors).cpu().tolist()
    return {
        key: cpu_tensors[on_gpu[key]] if key in on_gpu else loss_dict[key]
        for key in loss_dict
    }


class StateKeys:
    LOSS = "__LOSS__"
    LOSS_COMPS = "__LOSS_COMPS__"


class DataParallel(nn.DataParallel):
    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)


class ModelTrainer(DeviceContainer):
    model_type: Union[Type[torch.nn.Module], torch.nn.Module] = Field(..., description="Model Class")
    loss_func: SerializableCallable = Field(..., description="Loss function")
    name: Optional[str] = Field(None, description="Optional name for the trainer")
    model_kwargs: Optional[dict] = Field(None, description="Model kwargs")
    learn_rate: Optional[float] = Field(1e-3, description="Learning rate for optimizer")
    resume_from: Optional[int] = Field(None, description="Epoch to resume from")
    optim_type: AutoLambda[torch.optim.Optimizer] = Field(
        torch.optim.Adam,
        description="Optimizer class to use"
    )
    use_checkpointing: bool = Field(False, description="Enable checkpointing")
    grad_acc_size: int = Field(1, description="Gradient accumulation steps")
    device: TorchDevice = Field("cuda", description="Device to run training on ('cpu' or 'cuda')")
    dtype: TorchDType = Field(torch.float32, description="Data type for tensors")
    reg_func: AutoLambda[Tuple[torch.nn.Module], float] = Field(
        default=Lambda("lambda model: 0"),
        description="Regularization function"
    )
    sparsity: float = Field(0.0, description="Sparsity level")
    sparsity_dict: Optional[dict] = Field(
        None,
        description="Optional sparsity configuration dictionary"
    )
    scheduler_maker: Optional[AutoLambda[Tuple[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = Field(
        None,
        description="Learning rate scheduler (optional)"
    )
    gpu_id: int = Field(0, description="GPU ID to use")
    trainable_loss: bool = Field(False, description="Whether loss is trainable")
    strict_loading: bool = Field(True, description="Strict loading of state dicts")
    new_save_format: bool = Field(True, description="Use new save format for models")
    batch_norm_acc: bool = Field(False, description="Enable batch norm accumulation")
    load_optim: bool = Field(True, description="Load optimizer state from checkpoint")
    diversity_loss_func: Optional[AutoLambda] = Field(
        None,
        description="Optional diversity loss function"
    )
    show_true_loss: bool = Field(False, description="Show true loss during training")
    data_parallel: bool = Field(False, description="Use torch.nn.DataParallel")
    weight_inherit: bool = Field(False, description="Inherit weights from checkpoint")
    disable_logging: bool = Field(False, description="Disable logging", exclude=True)

    def __init__(self, env=None, **kwargs):
        super().__init__(**kwargs)
        self._env = None
        self.set_env(env)
        self.batch_state = {}
        self.log("loading model...")
        self.model = self.load_model(self.model_type, self.model_kwargs)
    
        sparsity = self.sparsity
        sparsity_dict = self.sparsity_dict
        pruner_type = FineGrainedPruner if (sparsity or sparsity_dict) else DummyPruner
        self.pruner = pruner_type(self.model, sparsity, sparsity_dict=sparsity_dict)
        self.log("model loading finished.")
        if self.trainable_loss:
            self.load_loss_weights()
        self.log("loading optimizer...")
        self.optimizer = self.load_optimizer()
        self.log("optimizer loading finished.")
        self.scheduler = self.scheduler_maker(self.optimizer) if self.scheduler_maker else None
        self.log(f"Model size: {calc_model_size(self.model)},  Num parameters: {count_parameters(self.model)}")

        self.bn_accumulator = None
        if self.batch_norm_acc and self.grad_acc_size > 1:
            self.log("Using BatchNormAccumulator")
            self.bn_accumulator = BatchNormAccumulator(
                self.model,
                num_accumulation_steps=self.grad_acc_size
            )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.get_env_param("use_amp", False))
        self.diversity_loss_man = DummyFDL()
        if self.diversity_loss_func:
            div_func = self.diversity_loss_func
            if not callable(div_func):
                div_func = DIVERSITY_FUNCS[div_func]
            self.diversity_loss_man = FeatureDiversityLoss(div_func, alpha=0.1)
            self.diversity_loss_man.register_hooks(self.model)
            self.log(f"Set diversity_loss_func to {div_func}")

    def log(self, *args, **kwargs):
        if self.disable_logging:
            return
        return print(f"{self.name if self.name else 'trainer'}:", *args, **kwargs)

    def get_env_param(self, param_name, default_val=None):
        if not self._env or not hasattr(self.env, param_name):
            return default_val
        return getattr(self.env, param_name)

    def set_env(self, env):
        assert isinstance(env, TrainingEnv), f"{self.name}: {env} must be an instance of TrainingEnv."
        self._env = env

    @property
    def env(self):
        if not self._env:
            raise Exception("No training environment set")
        return self._env

    def full_name(self):
        prefix = "" if not self._env else self._env.params.name
        return f"{prefix}{'-' if prefix and self.name else ''}{self.name}"
    
    def should_inherit_weights(self):
        weight_inherit = self.weight_inherit
        return weight_inherit and isinstance(weight_inherit, list | tuple) and len(weight_inherit) == 2

    def load_optimizer(self, save_folder=None):
        learn_rate = self.learn_rate
        optim_type = self.optim_type
        resume_from = self.resume_from or self.get_env_param("resume_from")
        optim_params = self.model.parameters()
        if self.trainable_loss:
            optim_params = list(optim_params) + list(self.loss_func.parameters())

        optimizer = optim_type(
            optim_params, lr=0.01 if learn_rate is None else learn_rate,
            eps=1e-4 if self.dtype == torch.float16 else 1e-8
        )
        weights_dir = (
            self.env.weights_dir if save_folder is None else f"{save_folder}/weights"
        ).rstrip('/')
        if self.load_optim and not (self.should_inherit_weights() or self.weight_inherit == -1):
            try:
                optim_state = torch.load(
                    f'{weights_dir}/{resume_from}_{self.full_name()}_optim.torch',
                    map_location='cpu'
                )
                optimizer.load_state_dict(optim_state)
                for g in optimizer.param_groups:
                    if learn_rate is not None:
                        g["lr"] = learn_rate
                del optim_state
            except FileNotFoundError:
                self.log("optimizer loading failed, using new optimizers.")
        else:
            self.log("Using new optimizers.")
        gc.collect()
        torch.cuda.empty_cache()
        return optimizer

    def load_model(self, model_type, model_kwargs, save_folder=None):
        resume_from = self.resume_from or self.get_env_param("resume_from")
        model = model_type if model_kwargs is None else model_type(
            **model_kwargs, device=self.device
        )
        if model_kwargs is None and not issubclass(model_type, ParamManager):
            self.log("Model init args will not be saved as it is not an instance of ParamManager")
        elif not issubclass(model_type, ParamManager):
            model.as_dict = lambda: model_kwargs
        if self.data_parallel:
            model = DataParallel(model)
        weights_dir = (
            self.env.weights_dir if save_folder is None else f"{save_folder}/weights"
        ).rstrip('/')
        state_dict = None
        if self.should_inherit_weights():
            model_name, epoch = self.weight_inherit
            parent_folder, _ = os.path.split(self.get_env_param("save_folder"))
            state_dict = torch.load(
                f"{parent_folder}/{model_name}/weights/{epoch}_{model_name}.torch",
                map_location="cpu"
            )
        elif resume_from and self.weight_inherit != -1:
            try:
                state_dict = torch.load(
                    f"{weights_dir}/{resume_from}_{self.full_name()}.torch",
                    map_location="cpu"
                )
            except Exception as e:
                self.log(f"Could not load from epoch {resume_from}")
                raise e
        if state_dict:
            model.load_state_dict(state_dict, strict=self.strict_loading)
            del state_dict

        return model.to(device=self.device, dtype=self.dtype)

    def load_loss_weights(self, save_folder=None):
        resume_from = self.resume_from or self.get_env_param("resume_from")
        if not self.resume_from:
            return
        weights_dir = (
            self.env.weights_dir if save_folder is None else f"{save_folder}/weights"
        ).rstrip('/')
        base_path = f"{weights_dir}/{resume_from}_{self.full_name()}"
        try:
            state_dict = torch.load(f"{base_path}_loss.torch", map_location='cpu')
            self.loss_func.load_state_dict(state_dict)
            del state_dict
            self.log(f"Loaded loss weights from epoch {resume_from}")
        except Exception as e:
            self.log(f"Could not loss func weights load from epoch {resume_from}")

    def safe_save_weights(self, save_obj, save_path, is_optim=False):
        save_attempts = 3
        saved = False
        while not saved:
            try:
                torch.save(
                    save_obj.state_dict(),
                    save_path,
                    _use_new_zipfile_serialization=self.new_save_format
                )
                return
            except Exception as e:
                save_attempts -= 1
            if not save_attempts:
                return e
            if not is_optim:
                for name, param in save_obj.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        self.log(f"NaN found in gradients of {name}")
                continue
            # Try to fix state
            for param_id, param_state in save_obj.state.items():
                for state_var in ['exp_avg', 'exp_avg_sq']:
                    if state_var not in param_state:
                        continue
                    var_value = param_state[state_var]
                    if torch.isnan(var_value).any():
                        self.log(f"NaN found in {state_var} for parameter ID {param_id}")
                for key, value in param_state.items():
                    if torch.is_tensor(value) and torch.isnan(value).any():
                        self.log(f"Resetting {key} for parameter ID {param_id}")
                        param_state[key] = torch.zeros_like(value)

    def save(self, epoch, save_folder=None):
        if self.gpu_id != 0:
            return
        weights_dir = (
            self.env.weights_dir if save_folder is None else f"{save_folder}/weights"
        ).rstrip('/')
        base_path = f"{weights_dir}/{epoch + 1}_{self.full_name()}"
        for save_type, save_obj in [("", self.model), ("_optim", self.optimizer)]:
            if save_error := self.safe_save_weights(save_obj, f"{base_path}{save_type}.torch", bool(save_type)):
                raise Exception(f"Failed to save {'optimizer' if save_type else 'model'} weights") from save_error
        if self.trainable_loss:
            torch.save(
                self.loss_func.state_dict(),
                f"{base_path}_loss.torch",
                _use_new_zipfile_serialization=self.new_save_format
            )

    def eval(self, enable_grad=False):
        self.model.requires_grad_(enable_grad)
        return self.model.eval()

    def train(self, enable_grad=True):
        self.model.requires_grad_(enable_grad)
        return self.model.train()

    def prune(self):
        return self.pruner.prune(self.model)

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def scheduler_step(self):
        if self.scheduler:
            self.scheduler.step()

    def acc_loss(self, pred, expected, loss_tracker, loss_func=None, always_acc_indivs=False):
        loss_func = self.loss_func if loss_func is None else loss_func
        loss_part, indivs_part = loss_func(pred, expected)
        loss_tracker[0] += loss_part
        if not self.model.training and not always_acc_indivs:
            for loss_name, val in indivs_part.items():
                loss_tracker[1].setdefault(loss_name, 0)
                loss_tracker[1][loss_name] += val
        return loss_tracker

    def get_images_idx(self, images, idx, slice_len=1):
        num_classes = self.model.num_classes
        if idx < 0:
            idx += images.shape[1] // num_classes
        return images[:, idx * num_classes:(idx + slice_len) * num_classes]

    def channel_unpack(self, images):
        num_classes = self.model.num_classes
        num_images = images.shape[1] // num_classes
        return [images[:, i * num_classes:(i + 1) * num_classes] for i in range(num_images)]

    def preprocess(self, inp):
        return inp

    def on_batch_end(self):
        self.batch_state = {}

    def on_epoch_end(self):
        self.scheduler_step()

    def on_training_end(self):
        if self.bn_accumulator:
            self.bn_accumulator.remove_hooks()
        self.diversity_loss_man.remove_hooks()
        self.eval()

    @property
    def last_loss_components(self):
        if StateKeys.LOSS_COMPS not in self.batch_state:
            self.batch_state[StateKeys.LOSS_COMPS] = {}
        return self.batch_state[StateKeys.LOSS_COMPS]

    def calc_loss_modifier(self):
        reg_loss = self.reg_func(self.model)
        diversity_loss = self.diversity_loss_man.get_loss()
        self.update_loss_components({
            "reg_loss": reg_loss,
            "diversity_loss": diversity_loss
        })
        return reg_loss + diversity_loss

    def _calc_loss(self, inp, ex):
        return self.call_loss_func(self.model(inp), ex)

    def calc_loss(self, *args, **kwargs):
        loss = self._calc_loss(*args, **kwargs)
        if not self.model.training:
            self.batch_state[StateKeys.LOSS] = loss
        return loss

    def get_progress_report(self):
        loss = self.batch_state[StateKeys.LOSS]
        label_prefix = self.name + "{}" if self.name else ""
        target_loss_key = label_prefix.format('_') + "target_loss"
        renamed_loss_cmps = {
            target_loss_key: loss,
            label_prefix.format('_') + "loss": loss,
        }
        loss_mod = 0
        for key, val in self.last_loss_components.items():
            loss_mod += val
            renamed_loss_cmps[label_prefix.format('_') + key] = val
        renamed_loss_cmps[target_loss_key] += loss_mod
        to_print = ""
        to_print += f" {label_prefix.format(' ')}Target: {np.format_float_scientific(loss + loss_mod, precision=3)}"
        if self.show_true_loss:
            to_print += f" {label_prefix.format(' ')}Loss: {np.format_float_scientific(loss, precision=3)}"
        return to_print, renamed_loss_cmps

    def calc_metrics(self):
        return {}

    def get_noise(self, inp, rand_mag=.064):
        return torch.rand(inp.shape, device=self.device, dtype=self.dtype) * rand_mag - (rand_mag / 2)

    def add_noise(self, inp, rand_mag=.064):
        return torch.clamp(inp + self.get_noise(inp, rand_mag=rand_mag), 0, 1)

    def do_optimizer_step(self, idx, grad_acc_size):
        return (
            (0 < grad_acc_size < 1 and idx == round(len(self.env.dataloader) * grad_acc_size))
            or (grad_acc_size >= 1 and (idx + 1) % grad_acc_size == 0)
        )

    def loss_step(self, idx, loss):
        self.scaler.scale(loss).backward(retain_graph=False)
        did_step = self.do_optimizer_step(idx, self.grad_acc_size)
        if did_step:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.diversity_loss_man.reset_loss()
        self.prune()
        return did_step

    def update_batch_state(self, state_key, name, val=None, force_update=False):
        if not force_update and self.model.training:
            return
        assert (isinstance(name, str) and not_none(val)) or isinstance(name, dict)
        if state_key not in self.batch_state:
            self.batch_state[state_key] = {}
        if isinstance(name, dict):
            self.batch_state[state_key].update(name)
            return
        self.batch_state[state_key][name] = val

    def update_loss_components(self, loss_name, loss_val=None, force_update=False):
        self.update_batch_state(StateKeys.LOSS_COMPS, loss_name, loss_val, force_update)

    def update_metrics(self, metric_name, metric_val=None):
        self.update_batch_state(StateKeys.METRICS, metric_name, metric_val, force_update=True)

    def call_loss_func(self, *args, **kwargs):
        result = self.loss_func(*args, **kwargs)
        if (is_list_like := isinstance(result, list | tuple)) and not self.model.training:
            if len(result) >= 3:
                self.update_metrics(result[2])
            elif len(result) >= 2:
                self.update_loss_components(result[1])
        return result[0] if is_list_like else result

    def status_update(self, images, expected, pred, epoch, batch_num=None):
        size = images.shape[-2:]
        name = self.env.params.name
        update_rate = self.env.params.update_rate
        save_folder = self.env.params.save_folder

        gc.collect()
        torch.cuda.empty_cache()
        in_img = images[0]
        ex = expected[0]
        out_img, *_,  = (pred[0],)
        show_images = []
        for i in range(len(in_img) // self.model.num_classes):
            show_images.append(in_img[self.model.num_classes * i: self.model.num_classes * (i + 1)])
        show_images.extend([ex, out_img])
        comp_img = torch.cat(show_images, 2).unsqueeze(0)
        comp_img = tensor_to_img(comp_img, size[0], size[1] * len(show_images))
        save_name = '' if name is None else (name + '_')
        init_suffix = "_init" if (epoch + (0 if batch_num is None else batch_num)) == 0 else ''
        batch_str = '' if batch_num is None else f"_{batch_num}"
        write_rgb(f'{save_folder}/{save_name}progress_{epoch//update_rate}{batch_str}{init_suffix}.png', comp_img)
        del comp_img
        gc.collect()
        torch.cuda.empty_cache()
        
        
class TrainingEnv(ParamManager):
    name: Optional[str] = Field(None, description="Optional name for the training run")
    trainer_init_args: List[ModelTrainer | dict] = Field(default_factory=list, description="Trainer init config list")
    update_rate: int = Field(10, description="Frequency of updates")
    save_rate: int = Field(50, description="Frequency of saves")
    resume_from: int = Field(0, description="Epoch or step to resume from")
    use_amp: bool = Field(False, description="Use automatic mixed precision")
    empty_cache_post_step: bool = Field(False, description="Empty cache after each step")
    empty_cache_pre_step: bool = Field(False, description="Empty cache before each step")
    watch_loss_funcs: Optional[Any] = Field(None, description="Loss functions to watch")
    drop_last: bool = Field(False, description="Drop last incomplete batch")
    num_workers: int = Field(0, description="Number of worker processes for data loading")
    save_folder: str = Field("", description="Folder path to save checkpoints")
    init_save: bool = Field(False, description="Save model state at initialization")
    empty_cache_post_epoch: bool = Field(False, description="Empty cache after each epoch")
    empty_cache_post_batch: bool = Field(False, description="Empty cache after each batch")
    reset_rng: Optional[Any] = Field(None, description="RNG reset configuration")
    save_manager: Optional[Any] = Field(None, exclude=True, description="Manager for save operations")
    size_limit_GB: Optional[float] = Field(
        None, description="Max size of weights to keep; uses KeepIntermediateManager; ignored if save_manager specified"
    )
    external_updater: Optional[Any] = Field(None, description="External updater object")
    show_batch_progress: bool = Field(True, description="Show progress bar for batches")
    show_num_unique: bool = Field(False, description="Show number of unique items in batches")
    rand_downscale_options: List[Any] = Field(default_factory=list, description="Random downscale options")
    no_downscale_batch_div: int = Field(4, description="Batch division factor for no downscale")
    efficient_multi_res: bool = Field(True, description="Use efficient multi-resolution strategy")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        name = self.name
        self.save_folder = os.path.join(
            self.params.save_folder,
            name if name is not None else ''
        )
        self.init_save_folder()
        if not_none(self.size_limit_GB) and self.save_manager is None:
            self.save_manager = KeepIntermediateManager(
                self.save_folder,
                self.size_limit_GB * (2 ** 30),
                save_rate=self.save_rate
            )
        print("creating dataset...")
        self.dataset = self.dataset_type(**self.dataset_kwargs,)
        print("dataset finished.")
        self.trainers = []
        for i, trainer_kwargs in enumerate(self.trainer_init_args):
            assert trainer_kwargs
            if isinstance(trainer_kwargs, ModelTrainer):
                trainer_kwargs.set_env(self)
            else:
                trainer_type = ModelTrainer
                if "trainer_type" in trainer_kwargs:
                    trainer_type = trainer_kwargs.pop("trainer_kwargs")
                trainer = trainer_type(self, **trainer_kwargs)
            self.trainers.append(trainer)
        self.loss_man = LossManager()
        if self.reset_rng:
            torch.manual_seed(self.reset_rng)
            torch.cuda.manual_seed_all(self.reset_rng)
            print(f"Reset random states with seed {self.reset_rng}")

    def init_save_folder(self):
        save_folder = self.save_folder
        self.loss_dir = f"{save_folder}/losses/"
        self.loss_path = f"{self.loss_dir}_{self.get_name(suffix='_')}loss.json"
        self.weights_dir = f"{save_folder}/weights/"
        self.run_history_dir = f"{save_folder}/run_args/"
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.run_history_dir, exist_ok=True)

    def get_dataloader(self):
        return DataLoader(
            self.dataset, self.batch_size, shuffle=True,
            drop_last=self.drop_last,
            multiprocessing_context=new_multiprocess_ctx if self.num_workers > 0 else None,
            num_workers=self.num_workers
        )

    def downscale_batch(self, images):
        if self.efficient_multi_res or not self.rand_downscale_options:
            return images
        size = self.dataset.size
        if percent_chance(1 - 1 / (len(self.rand_downscale_options) + 1)):
            downscale_params = random.choice(self.rand_downscale_options)
            if isinstance(downscale_params, int | float):
                downscale_params = [int(downscale_params)]
            if len(downscale_params) == 1:
                downscale_params = (downscale_params[0], 1)
            size_div, batch_div = downscale_params
            batch_len = batch_div[0] if type(batch_div) is list else images.shape[0] // batch_div
            return TF.resize(images, size[0] // size_div)[:batch_len]
        return images[:images.shape[0] // self.no_downscale_batch_div]

    def preprocess(self, images):
        return self.downscale_batch(images)

    def get_progress_report(self, idx, overall_epoch, loss_dict=None):
        to_print = f"Epoch: {overall_epoch}"
        if self.show_batch_progress:
            to_print += f" B%: {round(100 * idx / len(self.dataloader))}%"
        if (
            self.show_num_unique
            and hasattr(self.dataset, "track_unique_samples")
            and self.dataset.track_unique_samples
        ):
            to_print += f" Unq~ {len(self.dataset.unique_samples)}"
        loss_dict = {} if loss_dict is None else loss_dict
        for trainer in self.trainers:
            trainer_report_str, loss_components = trainer.get_progress_report()
            loss_dict.update(loss_components)
            report_str += trainer_report_str
        loss_dict = loss_dict_to_cpu(loss_dict)
        return to_print, loss_dict

    def update_progress_bar(self, display_str):
        self.train_iterator.set_postfix_str(display_str)

    def save_params(self, start_time):
        end_time = datetime.now()
        save_path = os.path.join(self.run_history_dir, end_time.strftime("%m-%d-%Y_%H-%M-%S") + ".json")
        with open(save_path, "w+", encoding="utf-8") as file:
            json.dump(dict({
                "start_time": start_time.strftime("%m-%d-%Y, %H:%M:%S"),
                "duration": f"{(end_time - start_time).seconds} seconds"
            }, **self.params), file, cls=ParamEncoder)

    def save(self, epoch):
        [trainer.save(epoch) for trainer in self.trainers]
        if self.save_manager:
            self.save_manager.cleanup()

    def normalize_loss(self, loss, indivs):
        # TODO: Implement
        return loss, indivs

    def stop(self):
        self._stop = True

    def run(self, rank=0, world_size=1):
        start_time = datetime.now()
        self.world_size = world_size
        self.gpu_id = rank

        try:
            self.dataloader = dataloader = self.get_dataloader()
            self.train_iterator = tqdm.tqdm(range(self.epochs), position=0, leave=True)
            if self.init_save and self.resume_from == 0:
                self.save(0)

            [trainer.zero_grad() for trainer in self.trainers]
            for epoch in self.train_iterator:
                overall_epoch = epoch + self.resume_from

                loss_dict = None
                batch_complete = False
                for idx, batch in enumerate(dataloader):
                    loss_dict = self.run_batch(idx, overall_epoch, batch)

                    report_str, loss_dict = self.get_progress_report(idx, overall_epoch, loss_dict)
                    self.loss_man.update(loss_dict, new_entry=idx == 0)
                    self.update_progress_bar(report_str)

                    if self.external_updater:
                        self.external_updater.update(self, epoch, idx, len(dataloader))

                    if idx == len(dataloader) - 1:
                        self.update_progress_bar(self.get_progress_report(
                            len(dataloader), overall_epoch, loss_dict
                        )[0])
                        batch_complete = True

                    [trainer.on_batch_end() for trainer in self.trainers]
                    if self.empty_cache_post_batch:
                        gc.collect()
                        torch.cuda.empty_cache()
                    if self._stop:
                        break

                [trainer.on_epoch_end() for trainer in self.trainers]

                if batch_complete:
                    if (epoch + 1) % self.save_rate == 0:
                        self.save(overall_epoch)
                    self.loss_man.update_path(self.loss_path, epoch=overall_epoch)

                if self.empty_cache_post_epoch:
                    gc.collect()
                    torch.cuda.empty_cache()
                if self._stop:
                    break
            if not self._stop:
                self.save(overall_epoch)
        except Exception as e:
            raise e
        finally:
            [trainer.on_training_end() for trainer in self.trainers]
            self.save_params(start_time)
        return self.trainers

    def get_loss_name(self, name, trainer=None):
        return f"{trainer.name + '_' if trainer.name else ''}{name}"

    def run_batch(self, idx, overall_epoch, batch):
        images = self.preprocess(batch)
        # defs
        epoch = overall_epoch - self.resume_from
        use_amp = self.use_amp
        watch_loss_funcs = self.watch_loss_funcs

        for trainer in self.trainers:
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, pred, expected, to_plot, *_, = trainer.calc_loss(
                    trainer.preprocess(images)
                )
                loss += trainer.calc_loss_modifier()

            if self.empty_cache_pre_step:
                gc.collect()
                torch.cuda.empty_cache()
            trainer.loss_step(idx, loss)
            if self.empty_cache_post_step:
                gc.collect()
                torch.cuda.empty_cache()

        watching = {}
        loss_dict = {}
        with torch.no_grad():
            for trainer in self.trainers:
                trainer.eval()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, pred, expected, *_, = trainer.calc_loss(images)
                    for func_name, watch_loss_func in watch_loss_funcs.items():
                        watching[func_name] = (watch_loss_func(pred, expected)[0] / self.norm_div)
                    loss_modifier = trainer.calc_loss_modifier()
                    loss += loss_modifier

                if idx == 0 and epoch % self.update_rate == 0:
                    trainer.status_update(to_plot, expected, pred, overall_epoch)
                trainer.train()

                loss_name = self.get_loss_name("loss", trainer)
                loss_dict.update({
                    self.get_loss_name("target_loss", trainer): loss + loss_modifier,
                    loss_name: loss
                })
                loss_dict.update(watching)
                loss_dict[self.get_loss_name("loss_sq", trainer)] = loss_dict[loss_name] ** 2
        return loss_dict


class GANTrainingEnv(TrainingEnv):
    gan_loss_scale: int = Field(1, description="Mulitplier for discrim loss when passed to generator")
    adv_loss_clamp: Optional[int | float] = Field(
        None, description="Set max value of adversarial loss to loss from primary loss function"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.trainers) >= 2
        self.gen = self.trainers[0]
        self.discrim = self.trainers[1]
        self.discrim_loss_scale = self.gan_loss_scale
        if isinstance(self.gan_loss_scale, int | float):
            self.discrim_loss_scale = lambda epoch, **kwargs: self.gan_loss_scale
        if self.watch_loss_funcs is None:
            self.watch_loss_funcs = {}

    def calc_adv_loss(self, overall_epoch, loss, discrim_loss):
        loss_item = loss.item()
        gan_loss_scale = self.discrim_loss_scale(overall_epoch, loss=loss_item)
        adv_loss = (discrim_loss.to(self.gen.device) * gan_loss_scale)
        if self.adv_loss_clamp:
            adv_loss = torch.clamp(adv_loss, min=None, max=loss_item)
        return adv_loss

    def get_dataloader(self):
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        if not self.efficient_multi_res or not self.rand_downscale_options:
            return super().get_dataloader()
        print("Using rand res sampler")
        batch_sampler = RandResBatchSampler(
            sampler=RandomSampler(self.dataset, generator=None),
            num_batches=self.batch_size, drop_last=self.drop_last,
            resolutions=self.rand_downscale_options
        )
        return DataLoader(
            self.dataset, batch_sampler=batch_sampler,
            multiprocessing_context=new_multiprocess_ctx if self.num_workers > 0 else None,
            num_workers=self.num_workers
        )

    def run_batch(self, idx, overall_epoch, batch):
        images = self.preprocess(batch)
        # defs
        epoch = overall_epoch - self.resume_from
        use_amp = self.use_amp
        watch_loss_funcs = self.watch_loss_funcs
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, pred, expected, to_plot, *inference_intermediates = self.gen.calc_loss(
                self.gen.preprocess(images)
            )
            real, fake, discrim_real, discrim_fake = self.discrim.preprocess(
                images, pred, expected, inference_intermediates
            )
            discrim_loss = self.discrim.calc_loss(discrim_real, discrim_fake)
            discrim_loss += self.discrim.calc_loss_modifier()

        if self.empty_cache_pre_step:
            gc.collect()
            torch.cuda.empty_cache()
        self.discrim.loss_step(idx, discrim_loss)
        if self.empty_cache_post_step:
            gc.collect()
            torch.cuda.empty_cache()

        with torch.amp.autocast("cuda", enabled=use_amp):
            self.discrim.eval()
            discrim_loss = self.discrim.calc_loss(real, fake, calc_metrics=True)
            loss -= self.calc_adv_loss(overall_epoch, loss, discrim_loss)
            loss += self.gen.calc_loss_modifier()

        if self.empty_cache_pre_step:
            gc.collect()
            torch.cuda.empty_cache()
        self.gen.loss_step(idx, loss)
        if self.empty_cache_post_step:
            gc.collect()
            torch.cuda.empty_cache()

        watching = {}
        with torch.no_grad():
            self.gen.eval()
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Discrim loss
                discrim_loss = discrim_loss.detach()
                discrim_loss_mod = self.discrim.calc_loss_modifier()

                # Gen loss
                loss, pred, expected, *_, = self.gen.calc_loss(images)
                for func_name, watch_loss_func in watch_loss_funcs.items():
                    watching[func_name] = (watch_loss_func(pred, expected)[0] / self.norm_div)
                loss -= self.calc_adv_loss(overall_epoch, loss, discrim_loss)
                loss_modifier = self.gen.calc_loss_modifier()
                loss += loss_modifier

            if idx == 0 and epoch % self.update_rate == 0:
                self.gen.status_update(to_plot, expected, pred, overall_epoch)
            self.gen.train()
            self.discrim.train()

        gen_loss_name = self.get_loss_name("loss", self.gen)
        loss_dict = {
            self.get_loss_name("target_loss", self.gen): loss + loss_modifier,
            gen_loss_name: loss,
            self.get_loss_name("target_loss", self.discrim): discrim_loss + discrim_loss_mod,
            self.get_loss_name("loss", self.discrim): discrim_loss
        }
        loss_dict.update(watching)
        loss_dict[self.get_loss_name("loss_sq", self.gen)] = loss_dict[gen_loss_name] ** 2
        return loss_dict


class DiscrimTrainer(ModelTrainer):
    display_confidence: bool = Field(True, description="Whether to display model confidence")
    label_flip_prob: float = Field(0.0, description="Probability of flipping labels (for data augmentation)")
    positive_label: float = Field(1.0, description="Label value for positive class")
    negative_label: float = Field(0.0, description="Label value for negative class")
    use_noise: bool = Field(False, description="Whether to add noise to inputs")
    mixup_alpha: Optional[float] = Field(0.4, description="Alpha parameter for mixup augmentation")
    mixup_samples: Optional[float] = Field(None, description="Ratio of images to use for mixup loss")
    mixup_loss_ratio: Optional[float] = Field(None, description="Loss weighting for mixup")
    show_KID: bool = Field(False, description="Show Kernel Inception Distance (KID) metric")
    show_FID: bool = Field(False, description="Show Frechet Inception Distance (FID) metric")
    fm_loss_mult: float = Field(0.0, description="Weight multiplier for feature matching loss")
    input_shuffle: bool = Field(True, description="Whether to shuffle inputs when classifying")

    def get_progress_report(self):
        loss_cmps = self.last_loss_components
        label_prefix = self.name + "{}" if self.name else ""
        to_print, renamed_cmps = super().get_progress_report()

        confusion_metrics = ["TP", "TN", "FP", "FN"]
        acc_key = label_prefix.format('_') + "Acc"
        if not acc_key in renamed_cmps and all(metric in loss_cmps for metric in confusion_metrics):
            total = sum(loss_cmps[metric] for metric in confusion_metrics)
            renamed_cmps[acc_key] = 100 * (loss_cmps["TP"] + loss_cmps["TN"]) / total
        to_print += f" {label_prefix.format(' ')}Acc: {round(renamed_cmps.get(acc_key), 2)}%"

        conf_key = label_prefix.format('_') + "Conf"
        confidence = loss_cmps.get('Conf')
        renamed_cmps[conf_key] = confidence
        if "Conf" in loss_cmps and self.display_confidence:
            to_print += f" {label_prefix.format(' ')}Conf: {round(confidence, 2)}%"
        renamed_cmps[label_prefix.format('_') + "KID"] = loss_cmps.get('KID', -1)
        if self.show_KID:
            to_print += f" KID {np.format_float_scientific(loss_cmps.get('KID', -1), precision=3)}"
        renamed_cmps[label_prefix.format('_') + "FID"] = loss_cmps.get('FID', -1)
        if self.show_FID:
            to_print += f" FID {loss_cmps.get('FID', -1)}"
        return to_print, renamed_cmps

    def _calc_loss(self, images, recon_x, calc_metrics=False):
        device = self.device
        dtype = self.dtype
        images = images.to(device, dtype)
        recon_x = recon_x.to(device, dtype)
        num_reals = images.shape[0]
        num_fakes = recon_x.shape[0]
        real_ex = torch.full((num_reals, 1), self.positive_label, device=device, dtype=dtype)
        fake_ex = torch.full((num_fakes, 1), self.negative_label, device=device, dtype=dtype)

        if self.label_flip_prob:
            # Recommended 0.05
            flip_prob = self.label_flip_prob
            # mask = ~(torch.rand(images.shape[0], 1, device=device, dtype=discrim_dtype) < flip_prob)
            mask = flip_prob < torch.rand(num_reals, 1, device=device, dtype=dtype)
            real_ex = real_ex * mask + self.negative_label * ~mask
            mask_f = flip_prob < torch.rand(num_fakes, 1, device=device, dtype=dtype)
            fake_ex = fake_ex * mask_f + self.positive_label * ~mask_f

        x = torch.cat([images, recon_x])
        ex = torch.cat([real_ex, fake_ex])
        if self.input_shuffle:
            idx = torch.randperm(num_reals + num_fakes, device=device)
            x, ex = x[idx], ex[idx]
        pred, enc = self.model(x)
        loss = self.call_loss_func(pred, ex, calc_metrics=calc_metrics)

        if isinstance(self.mixup_alpha, int | float) and self.mixup_samples:
            num_mixup_samples = int(len(images) * self.mixup_samples)
            mixup_samples, lam = mixup_data(
                images[:num_mixup_samples], recon_x[:num_mixup_samples],
                self.mixup_alpha
            )
            mixup_loss = mixup_bce(self.model.classify(mixup_samples), lam)
            self.update_loss_components("mixup_loss", mixup_loss)
            if (
                isinstance(self.mixup_loss_ratio, int | float)
                and 0 <= self.mixup_loss_ratio <= 1
            ):
                loss *= (1 - self.mixup_loss_ratio)
                mixup_loss *= self.mixup_loss_ratio
            loss += mixup_loss

        self.batch_state.update({"enc": enc})
        if calc_metrics:
            self.update_metrics(self.calc_metrics(images, recon_x))

        return loss

    def calc_loss_modifier(self):
        loss_mod = super().calc_loss_modifier()
        if self.fm_loss_mult:
            # Feature Matching Loss
            enc = self.batch_state.get("enc")
            fm_loss = self.fm_loss_mult * torch.nn.functional.mse_loss(
                enc[:len(enc) // 2], enc[len(enc) // 2:]
            )
            loss_mod -= fm_loss
            self.update_loss_components("fm_loss", fm_loss)
        return loss_mod

    @torch.no_grad()
    def calc_metrics(self, real, fake):
        metrics = {}
        if self.show_KID:
            metrics["KID"] = compute_kid(real, fake)
        if self.show_FID:
            metrics["FID"] = compute_fid(real, fake)
        return metrics

    def preprocess(self, images, pred, *_args):
        real = discrim_real = images
        fake = pred
        discrim_fake = fake.detach()
        return real, fake, discrim_real, discrim_fake
