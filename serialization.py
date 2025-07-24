import re
import ast
import importlib

import operator as op

from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import core_schema
from torch import nn
from typing import get_args, get_origin

from typing import Generic, Callable, Any, Union, Tuple, Type, TypeVar, overload, get_args, get_origin
from utils import write_json, load_json, not_none

PARAM_MAN_SER_PREFIX = "_ParamManager"


def is_serialized_param_man(val) -> bool:
    return type(val) is dict and PARAM_MAN_SER_PREFIX in val


def parse_serialized_param_man(val) -> dict:
    return val["config"]


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
    def __class_getitem__(cls, item: Return) -> 'AutoLambda[Args, Return]': ...
    
    @overload
    def __class_getitem__(cls, item: tuple[Args, Return]) -> 'AutoLambda[Args, Return]': ...

    def __class_getitem__(cls, item):
        if isinstance(item, tuple) and len(item) == 2:
            return super().__class_getitem__(item)
        else:
            # Treat single argument as Return type
            return super().__class_getitem__(((), item))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        expected_args = None
        expected_return = None

        if get_origin(source_type) is AutoLambda:
            type_args = get_args(source_type)
            if len(type_args) == 1:
                expected_return = type_args[0]
            elif len(type_args) == 2:
                expected_args, expected_return = type_args
            else:
                raise TypeError(f"Unsupported AutoLambda type form: {type_args}")

        def validate(v):
            if is_serialized_param_man(v):
                v = parse_serialized_param_man(v)
            if isinstance(v, Lambda):
                lam = v
            elif callable(v) or isinstance(v, str) or isinstance(v, dict):
                try:
                    lam = Lambda(**v) if type(v) is dict else Lambda(v)
                except Exception as e:
                    raise ValueError(f"Casting '{v}' to Lambda failed") from e
            else:
                raise ValueError(f"Cannot convert {v} to Lambda")
            
            func = lam.get_func()
            if expected_return:
                if isinstance(func, type):
                    args = get_args(expected_return)
                    if args and isinstance(args[0], type):
                        target_cls = args[0]
                    else:
                        target_cls = expected_return


                    if not isinstance(target_cls, type):
                        raise TypeError(f"Expected a concrete type, got {target_cls}")
                    if not issubclass(func, target_cls):
                        raise ValueError(f"{func} must be a subclass of {target_cls}")
                else:
                    # func is callable, check return type
                    dummy_args = [None] * (len(expected_args) if expected_args else 0)
                    result = func(*dummy_args)
                    if not isinstance(result, expected_return):
                        raise ValueError(f"{func} must return {expected_return}, got {type(result)}")

            return lam

        return core_schema.no_info_plain_validator_function(validate)
    
    @classmethod
    def _PM_auto_cast(self, obj):
        if isinstance(obj, Lambda):
            print("AutoLambda-_PM_auto_cast-nvm")
            return obj
        return Lambda(obj)



def get_module_name(cls, shortest=True):
    module_path = cls.__module__
    class_name = cls.__name__

    # Try resolving shorter module names
    module_parts = iter(module_path.split('.'))
    shortest_path = [next(module_parts)]
    while shortest:
        try:
            mod = importlib.import_module('.'.join(shortest_path))
            if hasattr(mod, class_name):
                module_path = '.'.join(shortest_path)
                break
        except ImportError:
            pass
        shortest_path.append(next(module_parts))

    return f"{module_path}.{class_name}"


def eval_obj_name(obj_name):
    name_split = obj_name.rsplit('.', 1)
    if len(name_split) == 1:
        return globals()[name_split[0]]
    module_path, *attrs, = name_split
    while module_path:
        try:
            module = importlib.import_module(module_path)
            break
        except ModuleNotFoundError:
            module_split = module_path.rsplit('.', 1)
            module_path = module_split[0] if len(module_split) == 2 else ""
            attrs.insert(0, module_split[-1])
    if not module_path:
        module = globals()[attrs[0]]
        attrs = attrs[1:]
    obj = module
    for module_name in attrs:
        obj = getattr(obj, module_name)
    return obj


class ParamManager:
    """Auto-validating parameter manager without BaseModel inheritance."""

    def __init_subclass__(cls, inherit_fields: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)

        def is_generic_type(typ):
            return get_origin(typ) is not None

        def get_auto_caster(typ):
            # Directly from typ or its origin
            origin = get_origin(typ)
            for candidate in [typ, origin]:
                if candidate and hasattr(candidate, "_PM_auto_cast"):
                    return False, candidate

            # If Union (like Optional[T]), check the arguments
            if origin is Union:
                args = get_args(typ)
                for arg in args:
                    if (auto_caster := get_auto_caster(arg)):
                        return type(None) in args, auto_caster[-1]

            return False, None

        # Collect annotated fields and defaults
        annotations = {}
        defaults = {}
        inherit_order = reversed(cls._PM_inherit_order()) if inherit_fields else [cls.__mro__[0]]

        for base in inherit_order:
            base_annotations = getattr(base, '__annotations__', {})
            base_defaults = {
                k: getattr(base, k)
                for k in base_annotations
                if hasattr(base, k)
            }
            annotations.update(base_annotations)
            defaults.update(base_defaults)

        fields = {}
        for name, typ in annotations.items():
            default = defaults.get(name, ...)
            print("field_name:", name)

            if isinstance(default, FieldInfo):
                if default.default_factory is not None:
                    # Call default_factory to get value
                    default = default.default_factory()
                elif default.default is not ...:
                    default = default.default
                else:
                    default = ...

            if default is not ...:
                optional, auto_caster = get_auto_caster(typ)
                if (
                    auto_caster
                    and (not optional or not_none(default))
                    and (is_generic_type(typ) or not isinstance(default, auto_caster))
                ):
                    try:
                        print(f"Wrapping '{default}' using {auto_caster}")
                        default = auto_caster._PM_auto_cast(default)
                    except Exception as e:
                        get_class_name = lambda obj: f"{obj.__module__}.{obj.__name__}"
                        raise TypeError(
                            f"Error casting default '{default}' of Field '{name}' using "
                            f"{get_class_name(auto_caster)} in class '{get_class_name(cls)}"
                        ) from e

            fields[name] = (typ, default)

        # Auto-generate a Pydantic BaseModel for validation
        cls._schema = create_model(
            f"{cls.__name__}Schema",
            __base__=BaseModel,
            __config__=ConfigDict(extra="forbid", frozen=True),
            **fields
        )

    def __init__(self, **kwargs):
        # Validate kwargs using the auto-generated Pydantic model
        validated = self._schema(**kwargs)
        for k in self._schema.model_fields:
            object.__setattr__(self, k, getattr(validated, k))
        self._params = validated

    def __dict__(self):
        return self.as_dict()
    
    def __str__(self):
        return self.model_dump_json()
        
    @property
    def params(self):
        return self._params
    
    def param_model_dump(self, *args, **kwargs):
        return self.params.model_dump(*args, **kwargs)

    def model_dump(self, *args, explicit=True, **kwargs):
        dump = {}
        for k, v in self.param_model_dump(*args, **kwargs).items():
            dump[k] = (
                v.model_dump(*args, explicit=explicit, **kwargs)
                if isinstance(v, ParamManager) else v
            )
        return dump if not explicit else {
            "_ParamManager": get_module_name(self.__class__),
            "config": dump
        }

    def model_dump_json(self, *args, explicit=True, **kwargs):
        return json.dumps(self.model_dump(
            *args, explicit=explicit, **kwargs
        ))

    def as_dict(self, *args, **kwargs) -> dict:
        return self.model_dump(*args, **kwargs)
    
    def save(self, save_path):
        return write_json(save_path, self.as_dict())
    
    @classmethod
    def load(cls, path):
        return cls.load_dict(load_json(path))
    
    @classmethod
    def load_dict(cls, inp):
        return eval_obj_name(inp[PARAM_MAN_SER_PREFIX])(**{
            k: cls.load_dict(v)
            if is_serialized_param_man(v) else v
            for k, v in inp["config"].items()
        })

    @classmethod
    def load_json(cls, inp):
        return cls.load_dict(json.loads(inp))
    
    @classmethod
    def _PM_inherit_order(cls):
        return tuple(c for c in cls.__mro__ if issubclass(c, ParamManager))



class SerializableModule(nn.Module, ParamManager):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        ParamManager.__init__(self, **kwargs)


class ModelComponent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def rec_call(self, call_attr):
        evaled_args = []
        for arg in self.args:
            if isinstance(arg, ModelComponent):
                arg = getattr(arg, call_attr)()
            evaled_args.append(arg)
        evaled_kwargs = {}
        for attr, val in self.kwargs.items():
            if isinstance(val, ModelComponent):
                val = getattr(val, call_attr)()
            evaled_kwargs[attr] = val
        return evaled_args, evaled_kwargs

    @classmethod
    def eval_obj_name(cls, obj_name):
        return eval_obj_name(obj_name)

    @classmethod
    def from_json(cls, json_el):
        def is_mc_json(el): return type(el) is dict and "__MCOBJECT__" in el
        if is_mc_json(json_el):
            obj_class = cls.eval_obj_name(json_el["__MCOBJECT__"])
            parsed_params = {param: cls.from_json(
                val) for param, val in json_el["params"].items()}
            args = []
            if "__ARGS__" in parsed_params:
                args = parsed_params.pop("__ARGS__")
            return obj_class(*args, **parsed_params)
        elif isinstance(json_el, dict):
            return {param: cls.from_json(val) for param, val in json_el.items()}
        elif isinstance(json_el, list):
            return [cls.from_json(el) for el in json_el]
        return json_el


class SafeEvaluator:
    """
    A safe expression evaluator that supports basic math and whitelisted function calls.
    """

    def __init__(self, allowed_funcs=None):
        # Operators to allow
        self.operators = {
            ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
            ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
            ast.Mod: op.mod
        }
        # Whitelisted functions (e.g., torch.relu, torch.mean)
        self.allowed_funcs = allowed_funcs or {}

    def eval_expr(self, expr, context=None):
        """
        Safely evaluate an expression string using the provided context.
        """
        tree = ast.parse(expr, mode='eval')
        return self._eval(tree.body, context or {})

    def _eval(self, node, context):
        if isinstance(node, ast.Constant):  # Literal number
            return node.n
        elif isinstance(node, ast.BinOp):  # Binary operation
            left = self._eval(node.left, context)
            right = self._eval(node.right, context)
            return self.operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operation
            operand = self._eval(node.operand, context)
            return self.operators[type(node.op)](operand)
        elif isinstance(node, ast.Name):  # Variable
            if node.id in context:
                return context[node.id]
            raise NameError(f"Undefined variable: {node.id}")
        elif isinstance(node, ast.Call):  # Function call
            func = self._eval(node.func, context)
            args = [self._eval(arg, context) for arg in node.args]
            return func(*args)
        elif isinstance(node, ast.Attribute):  # Attribute access (e.g., torch.mean)
            value = self._eval(node.value, context)
            return getattr(value, node.attr)
        elif isinstance(node, ast.Expr):
            return self._eval(node.value, context)
        else:
            raise TypeError(f"Unsupported expression: {ast.dump(node)}")

    def _eval_func(self, name):
        """Resolve a whitelisted function."""
        if name in self.allowed_funcs:
            return self.allowed_funcs[name]
        raise NameError(f"Function '{name}' is not allowed")
    

def is_param_man_json(val):
    return (
        type(val) is str
        and PARAM_MAN_SER_PREFIX in val
        and PARAM_MAN_SER_PREFIX in json.loads(val)
    )


# TODO: Remove arg_arity (make arity property base on passed kwargs and func kwargs)
# TODO: Add limited support for passing literal lambda objects (dill.source.getsource / inspect.getsource)
# TODO: Allow nested calling (maybe provide sets of args, base_func(*args_set1)(*args_set2)...(*args_setn))
# TODO: Refactor serialization into submodules with (_base/utils/_core/common).py for common functions
class Lambda(ParamManager, ModelComponent):
    # """A ParamManager-wrapped Lambda that supports arbitrary call signatures."""

    func_name: str | Callable[..., Any] = Field(
        ..., description="Function name as import path or callable."
    )
    arg_arity: Tuple[int, int] | int = Field(
        (0, -1), description="Min and max number of arguments."
    )
    func_caching_: bool = Field(
        True, description="Cache the resolved function after first evaluation."
    )
    call_on_eval_: bool = Field(
        False, description="Call the function on eval() instead of returning self."
    )
    ignore_args_: bool = Field(
        False, description="Ignore *args when called."
    )

    def __init__(self, func_name, arg_arity=(0, -1), func_caching_=True,
                 call_on_eval_=False, ignore_args_=False, **kwargs):
        func_name_callable = callable(func_name)
        assert func_name_callable or isinstance(func_name, str), f"'{func_name}' must be str or callable."
        if func_name_callable and ("<function <lambda>" in str(func_name) or ".<locals>." in str(func_name)):
            raise ValueError(
                "Callables like lambda functions or locally defined functions are not supported. "
                "Use a fully-qualified importable function or class (e.g., 'torch.nn.ReLU')."
            )
        cached_func = None
        if func_name_callable:
            cached_func = func_name
            func_name = (
                cached_func.model_dump_json()
                if isinstance(cached_func, ParamManager)
                else get_module_name(func_name)
            )

        if isinstance(arg_arity, int):
            arg_arity = (0, -1) if arg_arity == -1 else (arg_arity, arg_arity)

        super().__init__(
            func_name=func_name,
            arg_arity=arg_arity,
            func_caching_=func_caching_,
            call_on_eval_=call_on_eval_,
            ignore_args_=ignore_args_,
        )
        self._base_kwargs = kwargs
        self._func = cached_func if func_caching_ else None
        assert self._func is None or func_name_callable, f"'{func_name}' is not callable."

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
            func = self.eval_obj_name(self.func_name)
        assert callable(func), f"'{func}' is not callable."
        if self.func_caching_:
            self._func = func
        return func
    
    def get_func(self):
        return self.eval_func_name()

    def to_json(self):
        params = {
            "func_name": self.func_name,
            "arg_arity": self.arg_arity,
            "func_caching_": self.func_caching_,
            "call_on_eval_": self.call_on_eval_,
            "ignore_args_": self.ignore_args_
        }
        params.update(self._base_kwargs)
        return {
            "__MCOBJECT__": self.__class__.__name__,
            "params": params
        }

    @classmethod
    def eval_str(cls, string):
        try:
            return ast.literal_eval(string)
        except:
            return globals()[string]
        
    @classmethod
    def split_lambda_str(cls, lambda_str):
        arg_str, body_str = lambda_str[len("lambda"):].strip(' ').split(':')
        return [arg.strip(' ') for arg in arg_str.split(',') if arg.strip(' ')], body_str.strip(' ')

    @classmethod
    def parse(cls, lambda_str):
        cls_name = cls.__name__
        lambda_tag_len = len(cls_name)
        assert lambda_str[:lambda_tag_len] == cls_name.lower()
        arg_str_list, body_str = cls.split_lambda_str(lambda_str)
        if arg_str_list and arg_str_list[-1] == "**kwargs":
            arg_str_list.pop(-1)
        arbitrary_max = False
        if arg_str_list and arg_str_list[-1] == "*args":
            arg_str_list.pop(-1)
            arbitrary_max = True
        min_args = len(arg_str_list)
        max_args = -1 if arbitrary_max else min_args
        func_name, param_str = body_str.split('(')
        base_kwargs = {}
        for arg_pair in re.findall(r"[a-zA-Z.0-9]+=[a-zA-Z.0-9]+", param_str):
            arg_name, val_str = arg_pair.split('=', 1)
            base_kwargs[arg_name] = cls.eval_str(val_str)
        return Lambda(func_name.strip(' '), (min_args, max_args), **base_kwargs)

    def __str__(self):
        min_args, max_args = self.arg_arity
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

    def eval(self):
        return self() if self.call_on_eval_ else self

    def __call__(self, *args, **kwargs):
        if self.ignore_args_:
            args = []
        else:
            assert len(args) >= self.arg_arity[0], (
                f"Not enough args: expected at least {self.arg_arity[0]}, got {len(args)}"
            )
            assert (self.arg_arity[1] == -1 or len(args) <= self.arg_arity[1]), (
                f"Too many args: expected at most {self.arg_arity[1]}, got {len(args)}"
            )
        func = self.eval_func_name()
        return func(*args, **self._base_kwargs, **kwargs)
