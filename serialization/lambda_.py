import ast
import functools
import inspect
import json
import re

from typing import (
    Any, Callable, Dict, Generic, Optional, Sequence, Type, TypeVar, Union,
    get_args, get_origin, get_type_hints, overload
)
from pydantic import GetCoreSchemaHandler, Field
from pydantic_core import core_schema
from torch import nn

from utils import not_none
from .core import (
    PARAM_MAN_SER_PREFIX, ParamManager, SafeEvaluator,
    eval_obj_name, get_module_name, is_serialized_param_man,
    parse_serialized_param_man
)


def is_param_man_json(val):
    return (
        type(val) is str
        and PARAM_MAN_SER_PREFIX in val
        and PARAM_MAN_SER_PREFIX in json.loads(val)
    )


def get_arity(func):
    """
    Get the minimum and maximum number of positional arguments
    for any callable, including functools.partial and lambdas.

    Returns:
        (min_args, max_args)
        max_args == -1 means unbounded (*args present)
    """
    # Unwrap functools.partial
    while isinstance(func, functools.partial):
        func = func.func

    # If it's a callable object, get its __call__
    if not inspect.isfunction(func) and not inspect.ismethod(func):
        if isinstance(func, Lambda):
            return func.arity
        if isinstance(func, nn.Module):
            func = func.forward
        elif hasattr(func, "__call__"):
            func = func.__call__
        else:
            raise TypeError(f"{func} is not a callable")

    sig = inspect.signature(func)
    min_args = 0
    max_args = 0
    has_var_args = False

    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            min_args += int(param.default is param.empty)
            max_args += 1
        elif param.kind == param.VAR_POSITIONAL:
            has_var_args = True

    if has_var_args:
        max_args = -1  # Indicates unbounded args
    return min_args, max_args


def eval_str(string):
    try:
        return ast.literal_eval(string)
    except:
        return globals()[string]


class Lambda(ParamManager):
    # TODO: Remove arg_arity (make arity property base on passed kwargs and func kwargs)
    # TODO: Add limited support for passing literal lambda objects (dill.source.getsource / inspect.getsource)
    # TODO: Change AutoLambda to validate func result type using function annotation (if there enforce, else do nothing)

    """A ParamManager-wrapped Lambda that supports arbitrary call signatures."""

    func_name: str | Callable[..., Any] = Field(
        ..., description="Function name as import path or callable."
    )
    func_caching_: bool = Field(
        True, description="Cache the resolved function after first evaluation."
    )
    ignore_args_: bool = Field(
        False, description="Ignore *args when called."
    )
    parent_kwargs_: Optional[Sequence[Dict]] = Field(
        None, description="List of keyword arguments (dicts) used to evaluate function(s) returned by "
        "the top-level function given (func_name(**parent_kwargs[0])...(**parent_kwargs[n])...(**kwargs))"
    )

    def __init__(self, func_name, func_caching_=True, ignore_args_=False, parent_kwargs_=None, **kwargs):
        func_name_callable = callable(func_name)
        assert func_name_callable or isinstance(func_name, str), f"'{func_name}' must be str or callable."
        if func_name_callable and ("<function <lambda>" in str(func_name) or ".<locals>." in str(func_name)):
            raise ValueError(
                "Callables like lambda functions or locally defined functions are not supported. "
                "Use a fully-qualified importable function or class (e.g., 'torch.nn.ReLU')."
            )
        
        # Check if str is lambda str that calls a function with args/kwargs
        if (arg_extract := self.extract_init_args(func_name)):
            func_name, base_kwargs = arg_extract
            kwargs = dict(base_kwargs, **kwargs)

        cached_func = None
        if func_name_callable:
            cached_func = None if not_none(parent_kwargs_) else func_name
            func_name = (
                cached_func.model_dump_json()
                if isinstance(cached_func, ParamManager)
                else get_module_name(func_name)
            )

        if isinstance(parent_kwargs_, dict):
            parent_kwargs_ = [parent_kwargs_]

        super().__init__(
            func_name=func_name,
            func_caching_=func_caching_,
            ignore_args_=ignore_args_,
            parent_kwargs_=parent_kwargs_
        )
        self._base_kwargs = kwargs
        self._func = cached_func if func_caching_ else None

    def param_model_dump(self, *args, **kwargs):
        dump = super().param_model_dump(*args, **kwargs)
        dump.update(self._base_kwargs)
        return dump

    def eval_func_name(self, allowed_funcs=None):
        if self._func is not None:
            return self._func
        if self.func_name.startswith("lambda"):
            def run_evaluator(*args, **kwargs):
                param_list, body_str = self.split_lambda_str(self.func_name)
                assert len(param_list) == len(args), f"Expected {len(param_list)} arguments but got {len(args)}"
                context = {var_name: arg for var_name, arg in zip(param_list, args)}
                context.update({"nn": nn})
                context.update(kwargs)
                return SafeEvaluator(allowed_funcs=allowed_funcs).eval_expr(body_str, context=context)
            func = run_evaluator
        elif is_param_man_json(self.func_name):
            func = ParamManager.load_json(self.func_name)
        else:
            func = eval_obj_name(self.func_name)
        assert callable(func), f"'{func}' is not callable."
        if not_none(self.parent_kwargs_):
            for parent_kwargs in self.parent_kwargs_:
                func = func(**parent_kwargs)
        if self.func_caching_:
            self._func = func
        return func

    def get_func(self):
        return self.eval_func_name()
    
    @property
    def arity(self):
        return get_arity(self.get_func())
 
    @classmethod
    def split_lambda_str(cls, lambda_str):
        arg_str, body_str = lambda_str[len("lambda"):].strip().split(':')
        return [
            stripped_arg for arg in arg_str.split(',')
            if (stripped_arg := arg.strip())
        ], body_str.strip()

    @classmethod
    def extract_init_args(cls, lambda_str: str):
        try:
            assert type(lambda_str) is str
            cls_name = cls.__name__
            lambda_tag_len = len(cls_name)
            assert lambda_str[:lambda_tag_len].lower() == "lambda"
            _, body_str = cls.split_lambda_str(lambda_str)
            func_name, param_str = body_str.split('(')
            base_kwargs = {
                arg_name: eval_str(val_str)
                for arg_name, val_str in re.findall(r"(\w+)\s*=\s*([^,)]+)", param_str)
            }
            return func_name.strip(), base_kwargs
        except:
            return None

    def __str__(self):
        min_args, max_args = self.arity
        req_args = ', '.join([f"a{i}" for i in range(min_args)])
        star_args = ""
        if max_args == -1:
            if req_args:
                star_args += ", "
            star_args += "*args"
        args_str = req_args + star_args
        str_rep = f"{self.__class__.__name__} {args_str}"
        if args_str:
            str_rep += ", "
        str_rep += "**kwargs"
        str_rep += f": {self.func_name}({args_str}"
        if args_str:
            str_rep += ", "
        kwargs_str = ", ".join([
            f"{arg_name}={val}"
            for arg_name, val in self._base_kwargs.items()
        ])
        if kwargs_str:
            kwargs_str += ", "
        str_rep += "**kwargs)"
        return str_rep

    def __call__(self, *args, **kwargs):
        if self.ignore_args_:
            args = []
        func = self.eval_func_name()
        # TODO: Remove args passed through *args from _base_kwargs before calling ?
        return func(*args, **self._base_kwargs, **kwargs)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):

        def validate(v):
            if is_serialized_param_man(v):
                _, v = parse_serialized_param_man(v)
            if isinstance(v, cls):
                lam = v
            elif callable(v) or isinstance(v, str) or isinstance(v, dict):
                try:
                    lam = cls(**v) if type(v) is dict else cls(v)
                except Exception as e:
                    raise ValueError(f"Casting '{v}' to {cls.__name__} failed") from e
            else:
                raise ValueError(f"Cannot convert {v} to {cls.__name__}")

            return lam

        return core_schema.no_info_plain_validator_function(validate)

    @classmethod
    def _PM_auto_cast(cls, obj):
        return cls(obj)


class FuncWrapper(Lambda):
    def __init__(self, func):
        super().__init__(
            func_name=func,
            func_caching_=True,
            ignore_args_=False
        )
    def __call__(self, *args, **kwargs):
        return self.get_func()(*args, **kwargs)
    def param_model_dump(self, *args, **kwargs):
        return {"func_name": self.func_name}
    def eval_func_name(self, allowed_funcs=None):
        return self._func


class ModuleWrapper(FuncWrapper):
    func_name: Union[str, Type[nn.Module]] = Field(
        ..., description="nn.Module class to wrap"
    )
    def __init__(self, func_name):
        module = func_name
        if isinstance(func_name, str):
            module = eval_obj_name(func_name)
        assert issubclass(module, nn.Module), f"{module} is not nn.Module"
        super().__init__(module)
    @property
    def module(self):
        return self._func


Args = TypeVar("Args")
Return = TypeVar("Return")

class AutoLambda(Generic[Args, Return]):
    """
    Generic wrapper for Lambdas that auto-casts from string or callable.
    AutoLambda[T] means:
    - The callable itself must be T (if T is a type)
    - Or it must return T (if T is not a type)
    AutoLambda[[Args], Return] means:
    - Callable taking Args, returning Return
    """

    @overload
    def __class_getitem__(cls, item: Return) -> "AutoLambda[Args, Return]": ...
    
    @overload
    def __class_getitem__(cls, item: tuple[Args, Return]) -> "AutoLambda[Args, Return]": ...

    def __class_getitem__(cls, item):
        if isinstance(item, tuple) and len(item) == 2:
            return super().__class_getitem__(item)
        else:
            # Treat single argument as Return type
            return super().__class_getitem__(((), item))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        expected_args_typ = None
        expected_return = None

        if get_origin(source_type) is AutoLambda:
            type_args = get_args(source_type)
            if len(type_args) == 1:
                expected_return = type_args[0]
            elif len(type_args) == 2:
                expected_args_typ, expected_return = type_args
            elif len(type_args) > 2:
                raise TypeError(f"Unsupported AutoLambda type form: {type_args}")

        def validate(v):
            lam = cls._PM_auto_cast(v)
            func = None
            if expected_args_typ or expected_return:
                func = lam.get_func()

            if expected_args_typ and not lam.ignore_args_:
                sig_params = inspect.signature(func).parameters.values()
                expected_args = get_args(expected_args_typ)
                if len(sig_params) != len(expected_args):
                    raise TypeError(
                        f"{func} takes {len(sig_params)} args, "
                        f"but {len(expected_args)} expected"
                    )
                for _, (param, expected_type) in enumerate(zip(sig_params, expected_args)):
                    annotated = hints.get(param.name, None)
                    if not annotated or annotated == expected_type:
                        continue
                    raise TypeError(
                        f"Parameter '{param.name}' of {func} is annotated as {annotated}, "
                        f"but {expected_type} is expected"
                    )
            if expected_return:
                # Use func annotation for return type checking
                hints = get_type_hints(func)
                return_type = hints.get("return", None)
                if return_type and return_type != expected_return:
                    raise TypeError(
                        f"{func} is annotated to return {return_type}, "
                        f"but {expected_return} is expected"
                    )
            return lam

        return core_schema.no_info_plain_validator_function(validate)
    
    @classmethod
    def _PM_auto_cast(cls, v):
        if isinstance(v, Lambda):
            return v
        cast_type = Lambda
        if is_serialized_param_man(v):
            cast_type, v = parse_serialized_param_man(v)
        if isinstance(v, type) and issubclass(v, nn.Module):
            cast_type = ModuleWrapper
        elif callable(v):
            cast_type = FuncWrapper
        cast_result = v
        if callable(v) or isinstance(v, str) or isinstance(v, dict):
            try:
                cast_result = cast_type(**v) if type(v) is dict else cast_type(v)
            except Exception as e:
                raise ValueError(f"Casting '{v}' to {cast_type} failed") from e
        else:
            raise ValueError(f"Cannot convert {v} to {cast_type}")

        return cast_result

LambdaModuleT = AutoLambda[Type[nn.Module]]
