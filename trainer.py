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

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import RandomSampler
from datetime import datetime

from components.gradient_accumulation import BatchNormAccumulator
from components.loss_manager import LossManager
from image_utils import tensor_to_img, write_rgb
from loss.diversity import DIVERSITY_FUNCS, FeatureDiversityLoss, DummyFDL, compute_kid, compute_fid
from pruning import FineGrainedPruner, DummyPruner
from utils import calc_model_size, count_parameters, percent_chance, shuffle_tensor
from sampling import RandResBatchSampler

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

def raise_key_error(key):
    raise KeyError(key)
    
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._default = lambda _: None
    def set_default(self, default):
        self._default = default
    def __getattr__(self, attr):
        if attr not in self:
            return self._default(attr)
        return self[attr]
    
class ParamEncoder(json.JSONEncoder):
    def default(self, o):
        return repr(o)

class ParamManager:
    def __init__(self, **kwargs):
        self.params = AttrDict(**kwargs)
        self.params.set_default(raise_key_error)
        self.set_default_params()
    def set_default_params(self):
        return

def mixup_data(real, fake, alpha=0.4):
    """Applies MixUp to real and fake images"""
    batch_size = real.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(real.device)
    lam = lam.view(batch_size, 1, 1, 1)  # Reshape for broadcasting
    mixed = lam * real + (1 - lam) * fake
    return mixed, lam

def mixup_bce(pred, lam):
    real_labels = torch.ones(pred.shape[0], 1, device=pred.device, dtype=pred.dtype)
    fake_labels = torch.zeros(pred.shape[0], 1, device=pred.device, dtype=pred.dtype)
    real_loss = lam * F.binary_cross_entropy(pred, real_labels, reduction='none')
    fake_loss = (1 - lam) * F.binary_cross_entropy(pred, fake_labels, reduction='none')
    return (real_loss + fake_loss).mean()

def not_none(obj):
    return type(obj) is not type(None)

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

class TrainingEnv(ParamManager):
    def set_default_params(self):
        default_params = {
            "name": None,
            "update_rate": 10,
            "save_rate": 50,
            "resume_from": 0,
            "use_amp": False,
            "empty_cache_post_step": False,
            "empty_cache_pre_step": False,
            "watch_loss_funcs": None,
            "drop_last": False,
            "num_workers": 0,
            "save_folder": "",
            "init_save": False,
            "empty_cache_post_epoch": False,
            "empty_cache_post_batch": False,
            "reset_rng": None,
            "save_manager": None,
            "external_updater": None,
            "show_batch_progress": True,
            "show_num_unique": False,
            "rand_downscale_options": [],
            "no_downscale_batch_div": 4,
            "efficient_multi_res": True
        }
        for param in default_params:
            if param in self.params:
                continue
            self.params[param] = default_params[param]
        super().set_default_params()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        name = self.params.name
        self.params["save_folder"] = os.path.join(
            self.params.save_folder,
            name if name is not None else ''
        )
        self.init_save_folder()
        print("creating dataset...")
        self.dataset = self.params.dataset_type(**self.params.dataset_kwargs,)
        print("dataset finished.")
        self.trainers = []
        for i, trainer_kwargs in enumerate(self.params.trainer_kwargs_list):
            assert trainer_kwargs
            if isinstance(trainer_kwargs, ModelTrainer):
                self.params.trainer_kwargs_list[i] = trainer_kwargs.params
                trainer = trainer_kwargs
                trainer.set_env(self)
            else:
                trainer_type = ModelTrainer
                if "trainer_type" in trainer_kwargs:
                    trainer_type = trainer_kwargs.pop("trainer_kwargs")
                trainer = trainer_type(self, **trainer_kwargs)
            self.trainers.append(trainer)
        self.loss_man = LossManager()
        if self.params.reset_rng:
            torch.manual_seed(self.params.reset_rng)
            torch.cuda.manual_seed_all(self.params.reset_rng)
            print(f"Reset random states with seed {self.params.reset_rng}")

    def init_save_folder(self):
        save_folder = self.params.save_folder
        self.save_dir = f"{save_folder}/losses/"
        self.weights_dir = f"{save_folder}/weights/"
        self.run_history_dir = f"{save_folder}/run_args/"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.run_history_dir, exist_ok=True)

    def get_dataloader(self):
        return DataLoader(
            self.dataset, self.params.batch_size, shuffle=True,
            drop_last=self.params.drop_last,
            multiprocessing_context=new_multiprocess_ctx if self.params.num_workers > 0 else None,
            num_workers=self.params.num_workers
        )

    def downscale_batch(self, images):
        if self.params.efficient_multi_res or not self.params.rand_downscale_options:
            return images
        size = self.dataset.size
        if percent_chance(1 - 1 / (len(self.params.rand_downscale_options) + 1)):
            downscale_params = random.choice(self.params.rand_downscale_options)
            if isinstance(downscale_params, int | float):
                downscale_params = [int(downscale_params)]
            if len(downscale_params) == 1:
                downscale_params = (downscale_params[0], 1)
            size_div, batch_div = downscale_params
            batch_len = batch_div[0] if type(batch_div) is list else images.shape[0] // batch_div
            return TF.resize(images, size[0] // size_div)[:batch_len]
        return images[:images.shape[0] // self.params.no_downscale_batch_div]

    def preprocess(self, images):
        return self.downscale_batch(images)

    def get_progress_report(self, idx, overall_epoch, loss_dict=None):
        to_print = f"Epoch: {overall_epoch}"
        if self.params.show_batch_progress:
            to_print += f" B%: {round(100 * idx / len(self.dataloader))}%"
        if (
            self.params.show_num_unique
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
        if self.params.save_manager:
            self.params.save_manager.cleanup()

    def normalize_loss(self, loss, indivs):
        # TODO: Implement
        return loss, indivs

    def run(self, rank=0, world_size=1):
        start_time = datetime.now()
        self.world_size = world_size
        self.params.gpu_id = rank
        
        try:
            save_folder = self.params.save_folder
            name = self.params.name
            self.dataloader = dataloader = self.get_dataloader()
            self.train_iterator = tqdm.tqdm(range(self.params.epochs), position=0, leave=True)
            if self.params.init_save and self.params.resume_from == 0:
                self.save(0)

            [trainer.zero_grad() for trainer in self.trainers]
            for epoch in self.train_iterator:
                
                for idx, batch in enumerate(dataloader):
                    overall_epoch = epoch + self.params.resume_from
                    loss_dict = self.run_batch(idx, overall_epoch, batch)

                    report_str, loss_dict = self.get_progress_report(idx, overall_epoch, loss_dict)
                    self.loss_man.update(loss_dict, new_entry=idx == 0)
                    self.update_progress_bar(report_str)

                    if self.params.external_updater:
                        self.params.external_updater.update(self, epoch, idx, len(dataloader))
                    [trainer.on_batch_end() for trainer in self.trainers]
                    if self.params.empty_cache_post_batch:
                        gc.collect()
                        torch.cuda.empty_cache()

                [trainer.on_epoch_end() for trainer in self.trainers]

                self.update_progress_bar(self.get_progress_report(
                    len(dataloader), overall_epoch
                ))
                if (epoch + 1) % self.params.save_rate == 0:
                    self.save(overall_epoch)
                self.loss_man.update_path(f"{save_folder}/losses/_{name}_loss.json", epoch=overall_epoch)

                if self.params.empty_cache_post_epoch:
                    gc.collect()
                    torch.cuda.empty_cache()

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
        epoch = overall_epoch - self.params.resume_from
        use_amp = self.params.use_amp
        watch_loss_funcs = self.params.watch_loss_funcs

        for trainer in self.trainers:
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, pred, expected, to_plot, *_, = trainer.calc_loss(
                    trainer.preprocess(images)
                )
                loss += trainer.calc_loss_modifier()

            if self.params.empty_cache_pre_step:
                gc.collect()
                torch.cuda.empty_cache()
            trainer.loss_step(idx, loss)
            if self.params.empty_cache_post_step:
                gc.collect()
                torch.cuda.empty_cache()

        watching = {}
        loss_dict = {}
        with torch.no_grad():
            for trainer in self.trainers:
                trainer.eval()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, pred, expected, *_, = trainer.calc_loss(images)
                    for func_name, watch_loss_func in watch_loss_funcs.items():
                        watching[func_name] = (watch_loss_func(pred, expected)[0] / self.norm_div)
                    loss_modifier = trainer.calc_loss_modifier()
                    loss += loss_modifier

                if idx == 0 and epoch % self.params.update_rate == 0:
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
    def set_default_params(self):
        default_params = {
            "gan_loss_scale": 1,
            "discrim_inherit": None,
            "discrim_loss_clamp": False,
        }
        for param in default_params:
            if param in self.params:
                continue
            self.params[param] = default_params[param]
        super().set_default_params()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.trainers) >= 2
        self.gen = self.trainers[0]
        self.discrim = self.trainers[1]
        self.discrim_loss_scale = self.params.gan_loss_scale
        if isinstance(self.params.gan_loss_scale, int | float):
            self.discrim_loss_scale = lambda epoch, **kwargs: self.params.gan_loss_scale
        if self.params.watch_loss_funcs is None:
            self.params["watch_loss_funcs"] = {}
    def calc_adv_loss(self, overall_epoch, loss, discrim_loss):
        loss_item = loss.item()
        gan_loss_scale = self.discrim_loss_scale(overall_epoch, loss=loss_item)
        adv_loss = (discrim_loss.to(self.gen.device) * gan_loss_scale)
        if self.params.discrim_loss_clamp:
            adv_loss = torch.clamp(adv_loss, min=None, max=loss_item)
        return adv_loss
    def should_inherit_discrim(self):
        discrim_inherit = self.params.discrim_inherit
        return discrim_inherit and isinstance(discrim_inherit, list | tuple) and len(discrim_inherit) == 2
    def get_dataloader(self):
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        if not self.params.efficient_multi_res or not self.params.rand_downscale_options:
            return super().get_dataloader()
        print("Using rand res sampler")
        batch_sampler = RandResBatchSampler(
            sampler=RandomSampler(self.dataset, generator=None),
            num_batches=self.params.batch_size, drop_last=self.params.drop_last,
            resolutions=self.params.rand_downscale_options
        )
        return DataLoader(
            self.dataset, batch_sampler=batch_sampler,
            multiprocessing_context=new_multiprocess_ctx if self.params.num_workers > 0 else None,
            num_workers=self.params.num_workers
        )
    def run_batch(self, idx, overall_epoch, batch):
        images = self.preprocess(batch)
        # defs
        epoch = overall_epoch - self.params.resume_from
        use_amp = self.params.use_amp
        watch_loss_funcs = self.params.watch_loss_funcs
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, pred, expected, to_plot, *inference_intermediates = self.gen.calc_loss(
                self.gen.preprocess(images)
            )
            real, fake, discrim_real, discrim_fake = self.discrim.preprocess(
                images, pred, expected, inference_intermediates
            )
            discrim_loss = self.discrim.calc_loss(discrim_real, discrim_fake)
            discrim_loss += self.discrim.calc_loss_modifier()

        if self.params.empty_cache_pre_step:
            gc.collect()
            torch.cuda.empty_cache()
        self.discrim.loss_step(idx, discrim_loss)
        if self.params.empty_cache_post_step:
            gc.collect()
            torch.cuda.empty_cache()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            self.discrim.eval()
            discrim_loss = self.discrim.calc_loss(real, fake, calc_metrics=True)
            loss -= self.calc_adv_loss(overall_epoch, loss, discrim_loss)
            loss += self.gen.calc_loss_modifier()

        if self.params.empty_cache_pre_step:
            gc.collect()
            torch.cuda.empty_cache()
        self.gen.loss_step(loss)
        if self.params.empty_cache_post_step:
            gc.collect()
            torch.cuda.empty_cache()

        watching = {}
        with torch.no_grad():
            self.gen.eval()
            with torch.cuda.amp.autocast(enabled=use_amp):
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

            if idx == 0 and epoch % self.params.update_rate == 0:
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

    
class ModelTrainer(ParamManager):
    def set_default_params(self):
        default_params = {
            "name": None,
            "learn_rate": 1e-3,
            "device": "cuda",
            "optim_type": torch.optim.Adam,
            "use_checkpointing": False,
            "grad_acc_size": 1,
            "dtype": torch.float32,
            "reg_func": lambda model: 0,
            "sparsity": 0,
            "sparsity_dict": None,
            "scheduler": None,
            "gpu_id": 0,
            "trainable_loss": False,
            "strict_loading": True,
            "new_save_format": True,
            "batch_norm_acc": False,
            "load_optim": True,
            "diversity_loss_func": None,
            "show_true_loss": False,
            "data_parallel": False
        }
        for param in default_params:
            if param in self.params:
                continue
            self.params[param] = default_params[param]
        super().set_default_params()
    def __init__(self, env=None, **kwargs):
        super().__init__(**kwargs)
        self._env = None
        self.set_env(env)
        self.batch_state = {}
        self.name = self.params.name
        self.dtype = self.params.dtype
        self.device = torch.device(
            self.params.device if ("cuda" in self.params.device and torch.cuda.is_available())
            else "cpu"
        )
        self.log("loading model...")
        self.model = self.load_model(self.params.model_type, self.params.model_kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        
        sparsity = self.params.sparsity
        sparsity_dict = self.params.sparsity_dict
        pruner_type = FineGrainedPruner if (sparsity or sparsity_dict) else DummyPruner
        self.pruner = pruner_type(self.model, sparsity, sparsity_dict=sparsity_dict)
        self.log("model loading finished.")
        if self.params.trainable_loss:
            self.load_loss_weights()
        self.log("loading optimizer...")
        self.optimizer = self.load_optimizer()
        self.scheduler = None
        if self.params.scheduler:
            self.scheduler = self.params.scheduler(self.optimizer)
        self.log("optimizer loading finished.")
        self.log(f"Model size: {calc_model_size(self.model)},  Num parameters: {count_parameters(self.model)}")

        self.bn_accumulator = None
        if self.params.batch_norm_acc and self.params.grad_acc_size > 1:
            self.log("Using BatchNormAccumulator")
            self.bn_accumulator = BatchNormAccumulator(
                self.model,
                num_accumulation_steps=self.params.grad_acc_size
            )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.get_env_param("use_emp", False))
        self.diversity_loss_man = DummyFDL()
        if self.params.diversity_loss_func:
            div_func = self.params.diversity_loss_func
            if not callable(div_func):
                div_func = DIVERSITY_FUNCS[div_func]
            self.diversity_loss_man = FeatureDiversityLoss(div_func, alpha=0.1)
            self.diversity_loss_man.register_hooks(self.model)
            self.log(f"Set diversity_loss_func to {div_func}")

    def log(self, *args, **kwargs):
        if self.disable_logging:
            return
        return print(f"{self.name}:", *args, **kwargs)
    
    def get_env_param(self, param_name, default_val=None):
        if not self._env or param_name not in self.env.params:
            return default_val
        return self.env.params[param_name]

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

    def load_optimizer(self, save_folder=None):
        learn_rate = self.params.learn_rate
        optim_type = self.params.optim_type
        resume_from = self.params.resume_from
        optim_params = self.model.parameters()
        if self.params.trainable_loss:
            optim_params = list(optim_params) + list(self.params.loss_func.parameters())
        
        optimizer = optim_type(
            optim_params, lr=0.01 if learn_rate is None else learn_rate,
            eps=1e-4 if self.params.dtype == torch.float16 else 1e-8
        )
        weights_dir = (
            self.env.weights_dir if save_folder is None else f"{save_folder}/weights"
        ).rstrip('/')
        if self.params.load_optim:
            try:
                optim_state = torch.load(
                    f'{weights_dir}/{resume_from}_{self.full_name()}_optim.torch',
                    map_location='cpu'
                )
                optimizer.load_state_dict(optim_state)
                for g in optimizer.param_groups:
                    if learn_rate is not None:
                        g['lr'] = learn_rate
            except FileNotFoundError:
                self.log("optimizer loading failed, using new optimizers.")
        else:
            self.log("Using new optimizers.")
        gc.collect()
        torch.cuda.empty_cache()
        return optimizer
        
    def load_model(self, model_type, model_kwargs, save_folder=None):
        resume_from = self.params.resume_from
        model = model_type if model_kwargs is None else model_type(
            **model_kwargs, device=self.device
        )
        if self.params.data_parallel:
            model = DataParallel(model)
        weights_dir = (
            self.env.weights_dir if save_folder is None else f"{save_folder}/weights"
        ).rstrip('/')
        if resume_from:
            try:
                state_dict = torch.load(
                    f'{weights_dir}/{resume_from}_{self.full_name()}.torch',
                    map_location='cpu'
                )
                model.load_state_dict(state_dict, strict=self.params.strict_loading)
                del state_dict
            except Exception as e:
                self.log(f"Could not load from epoch {resume_from}")
                raise e

        return model.to(device=self.device, dtype=self.params.dtype)

    def load_loss_weights(self, save_folder=None):
        resume_from = self.params.resume_from
        if not self.params.resume_from:
            return
        weights_dir = (
            self.env.weights_dir if save_folder is None else f"{save_folder}/weights"
        ).rstrip('/')
        base_path = f"{weights_dir}/{resume_from}_{self.full_name()}"
        try:
            state_dict = torch.load(f"{base_path}_loss.torch", map_location='cpu')
            self.params.loss_func.load_state_dict(state_dict)
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
                    _use_new_zipfile_serialization=self.params.new_save_format
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
        if self.params.gpu_id != 0:
            return
        weights_dir = (
            self.env.weights_dir if save_folder is None else f"{save_folder}/weights"
        ).rstrip('/')
        base_path = f"{weights_dir}/{epoch + 1}_{self.full_name()}"
        for save_type, save_obj in enumerate([("", self.model), ("_optim", self.optimizer)]):
            if save_error := self.safe_save_weights(save_obj, f"{base_path}{save_type}.torch", bool(save_type)):
                raise Exception(f"Failed to save {'optimizer' if save_type else 'model'} weights") from save_error
        if self.params.trainable_loss:
            torch.save(
                self.params.loss_func.state_dict(),
                f"{base_path}_loss.torch",
                _use_new_zipfile_serialization=self.params.new_save_format
            )
    def eval(self):
        return self.model.eval()
    def train(self):
        return self.model.train()
    def prune(self):
        return self.pruner.prune(self.model)
    def zero_grad(self):
        return self.optimizer.zero_grad()
    def scheduler_step(self):
        if self.scheduler:
            self.scheduler.step()
    def acc_loss(self, pred, expected, loss_tracker, loss_func=None, always_acc_indivs=False):
        loss_func = self.params.loss_func if loss_func is None else loss_func
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
        self.model.eval()
        self.model.requires_grad_(False)
    @property
    def last_loss_components(self):
        if StateKeys.LOSS_COMPS not in self.batch_state:
            self.batch_state[StateKeys.LOSS_COMPS] = {}
        return self.batch_state[StateKeys.LOSS_COMPS]
    def calc_loss_modifier(self):
        reg_loss = self.params.reg_func(self.model)
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
        if self.params.show_true_loss:
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
        did_step = self.do_optimizer_step(idx, self.params.grad_acc_size)
        if did_step:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.diversity_loss_man.reset_loss()
        self.prune()
        return did_step
    def update_loss_components(self, loss_name, loss_val=None):
        if self.model.training:
            return
        assert (isinstance(loss_name, str) and not_none(loss_val)) or isinstance(loss_name, dict)
        if isinstance(loss_name, dict):
            self.last_loss_components.update(loss_name)
            return
        self.last_loss_components[loss_name] = loss_val
    def call_loss_func(self, *args, **kwargs):
        result = self.params.loss_func(*args, **kwargs)
        if len(result) >= 2 and not self.model.training:
            self.update_loss_components(result[1])
        return result[0]
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


