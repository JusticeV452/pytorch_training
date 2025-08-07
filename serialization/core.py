import ast
import importlib
import json
import operator as op

from typing import get_args, get_origin
from pydantic import BaseModel, ConfigDict, create_model
from pydantic.fields import FieldInfo

from typing import Union, get_args, get_origin
from utils import write_json, load_json, not_none

PARAM_MAN_SER_PREFIX = "_ParamManager"


def is_serialized_param_man(val) -> bool:
    return type(val) is dict and PARAM_MAN_SER_PREFIX in val


def parse_serialized_param_man(val) -> dict:
    return eval_obj_name(val[PARAM_MAN_SER_PREFIX][0]), val[PARAM_MAN_SER_PREFIX][1]


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
    if module_path == "nn":
        module_path = "torch.nn"
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
            base_annotations = getattr(base, "__annotations__", {})
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
                        # default = getattr(auto_caster, "_PM_auto_cast", auto_caster)(default)
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
            PARAM_MAN_SER_PREFIX: (get_module_name(self.__class__), dump)
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
        obj_cls_name, dump = inp[PARAM_MAN_SER_PREFIX]
        return eval_obj_name(obj_cls_name)(**{
            k: cls.load_dict(v)
            if is_serialized_param_man(v) else v
            for k, v in dump.items()
        })

    @classmethod
    def load_json(cls, inp):
        return cls.load_dict(json.loads(inp))
    
    @classmethod
    def _PM_inherit_order(cls):
        return tuple(c for c in cls.__mro__ if issubclass(c, ParamManager))


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
