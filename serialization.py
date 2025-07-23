import re
import ast
import importlib

import operator as op

from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, Field, PrivateAttr, create_model, model_serializer
from pydantic_core import core_schema
from torch import nn
from typing import Any, Tuple, Callable
from typing import Generic, TypeVar, Tuple, Callable, Any, Union

ArgsT = TypeVar('ArgsT')
ReturnT = TypeVar('ReturnT')

PARAM_MAN_SER_PREFIX = "_ParamManager"

def is_serialized_param_man(val) -> bool:
    return type(val) is dict and PARAM_MAN_SER_PREFIX in val
def parse_serialized_param_man(val) -> dict:
    return val["config"]
class AutoLambda(Generic[ArgsT, ReturnT]):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        def validate(v):
            if v is None or isinstance(v, Lambda):
                return v
            try:
                if is_serialized_param_man(v):
                    v = parse_serialized_param_man(v)
                lam = Lambda(**v) if type(v) is dict else Lambda(v)
            except Exception as e:
                raise ValueError(f"'{v}' cannot be cast to Lambda") from e
            return lam

        return core_schema.no_info_plain_validator_function(validate)


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


class ParamManager:
    """Auto-validating parameter manager without BaseModel inheritance."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Collect annotated fields and defaults
        annotations = getattr(cls, '__annotations__', {})
        defaults = {
            k: getattr(cls, k)
            for k in annotations.keys()
            if hasattr(cls, k)
        }

        # Auto-generate a Pydantic BaseModel for validation
        cls._schema = create_model(
            f"{cls.__name__}Schema",
            __base__=BaseModel,
            __config__=ConfigDict(extra="forbid", frozen=True),
            **{
                name: (typ, defaults.get(name, ...))
                for name, typ in annotations.items()
            }
        )

    def __init__(self, **kwargs):
        # Validate kwargs using the auto-generated Pydantic model
        validated = self._schema(**kwargs)
        for k in validated.__fields__:
            object.__setattr__(self, k, getattr(validated, k))
        self._params = validated

    @property
    def params(self):
        return self._params

    def as_dict(self, *args, **kwargs) -> dict:
        """Dump only the validated parameter fields."""
        return self.params.model_dump(*args, **kwargs)


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
        # Convert positional args to keyword args for Pydantic
        func_name = func_name if isinstance(func_name, str) else get_module_name(func_name)
        func_name_callable = callable(func_name)
        assert func_name_callable or isinstance(func_name, str), f"'{func_name}' must be str or callable."
        if func_name_callable and ("<function <lambda>" in str(func_name) or ".<locals>." in str(func_name)):
            raise ValueError(
                "Callables like lambda functions or locally defined functions are not supported. "
                "Use a fully-qualified importable function or class (e.g., 'torch.nn.ReLU')."
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
        self._func = None if isinstance(func_name, str) or not func_caching_ else func_name
        assert self._func is None or func_name_callable, f"'{func_name}' is not callable."

    def model_dump(self, *args, **kwargs) -> dict:
        """Flatten base_kwargs into the top level of the dict."""
        data = super().model_dump(*args, **kwargs)
        data.update(self._base_kwargs)
        return data

    def eval_func_name(self, allowed_funcs=None):
        if self._func is not None:
            return self._func
        if self.func_name.startswith("lambda"):
            def run_evaluator(*args, **kwargs):
                param_list, body_str = self.split_lambda_str(self.func_name)
                assert len(param_list) == len(args), f"Expected {len(param_list)} arguments but got {len(args)}"
                context = {var_name: arg for var_name, arg in zip(param_list, args)}
                context.update({"nn": nn})
                return SafeEvaluator(allowed_funcs=allowed_funcs).eval_expr(body_str, context=context)
            func = run_evaluator
        else:
            func = self.eval_obj_name(self.func_name)
        assert callable(func), f"'{func}' is not callable."
        if self.func_caching_:
            self._func = func
        return func

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
