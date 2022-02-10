import copy
import importlib
import json
import os
import re
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Optional, TypeVar, Union

import yaml


def with_lock(func: Callable):
    """Use as decorator (@withlock) around object methods that need locking.

    Note: The object must have a self._lock = threading.Lock() property.
    Locking thus works on the object level (no two locked methods of the same
    object can be called asynchronously).

    Args:
        func (Callable): The function to decorate/wrap.

    Returns:
        Callable: The wrapped (object-level locked) function.
    """

    def wrapper(self, *a, **k):
        try:
            with self._lock:
                return func(self, *a, **k)
        except AttributeError as e:
            if "has no attribute '_lock'" in e.args[0]:
                raise AttributeError(
                    "Object {} must have a `self._lock` property (assigned "
                    "to a threading.RLock() object in its "
                    "constructor)!".format(self))
            raise e

    return wrapper

def deep_update(
        original: dict,
        new_dict: dict,
        new_keys_allowed: str = False,
        allow_new_subkey_list: Optional[List[str]] = None,
        override_all_if_type_changes: Optional[List[str]] = None) -> dict:
    """Updates original dict with values from new_dict recursively.

    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the allow_new_subkey_list, then new subkeys can be introduced.

    Args:
        original (dict): Dictionary with default values.
        new_dict (dict): Dictionary with values to be updated
        new_keys_allowed (bool): Whether new keys are allowed.
        allow_new_subkey_list (Optional[List[str]]): List of keys that
            correspond to dict values where new subkeys can be introduced.
            This is only at the top level.
        override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (dict), iff the "type" key in that value dict changes.
    """
    allow_new_subkey_list = allow_new_subkey_list or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}` ".format(k))

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Allowed key -> ok to add new subkeys.
            elif k in allow_new_subkey_list:
                deep_update(original[k], value, True)
            # Non-allowed key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original

def merge_dicts(d1: dict, d2: dict) -> dict:
    """
    Args:
        d1 (dict): Dict 1.
        d2 (dict): Dict 2.

    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged



def force_list(elements=None, to_tuple=False):
    """
    Makes sure `elements` is returned as a list, whether `elements` is a single
    item, already a list, or a tuple.

    Args:
        elements (Optional[any]): The inputs as single item, list, or tuple to
            be converted into a list/tuple. If None, returns empty list/tuple.
        to_tuple (bool): Whether to use tuple (instead of list).

    Returns:
        Union[list,tuple]: All given elements in a list/tuple depending on
            `to_tuple`'s value. If elements is None,
            returns an empty list/tuple.
    """
    ctor = list
    if to_tuple is True:
        ctor = tuple
    return ctor() if elements is None else ctor(elements) \
        if type(elements) in [list, tuple] else ctor([elements])


def from_config(cls, config=None, **kwargs):
    """Uses the given config to create an object.

    If `config` is a dict, an optional "type" key can be used as a
    "constructor hint" to specify a certain class of the object.
    If `config` is not a dict, `config`'s value is used directly as this
    "constructor hint".

    The rest of `config` (if it's a dict) will be used as kwargs for the
    constructor. Additional keys in **kwargs will always have precedence
    (overwrite keys in `config` (if a dict)).
    Also, if the config-dict or **kwargs contains the special key "_args",
    it will be popped from the dict and used as *args list to be passed
    separately to the constructor.

    The following constructor hints are valid:
    - None: Use `cls` as constructor.
    - An already instantiated object: Will be returned as is; no
        constructor call.
    - A string or an object that is a key in `cls`'s `__type_registry__`
        dict: The value in `__type_registry__` for that key will be used
        as the constructor.
    - A python callable: Use that very callable as constructor.
    - A string: Either a json/yaml filename or the name of a python
        module+class (e.g. "ray.rllib. [...] .[some class name]")

    Args:
        cls (class): The class to build an instance for (from `config`).
        config (Optional[dict, str]): The config dict or type-string or
            filename.

    Keyword Args:
        kwargs (any): Optional possibility to pass the constructor arguments in
            here and use `config` as the type-only info. Then we can call
            this like: from_config([type]?, [**kwargs for constructor])
            If `config` is already a dict, then `kwargs` will be merged
            with `config` (overwriting keys in `config`) after "type" has
            been popped out of `config`.
            If a constructor of a Configurable needs *args, the special
            key `_args` can be passed inside `kwargs` with a list value
            (e.g. kwargs={"_args": [arg1, arg2, arg3]}).

    Returns:
        any: The object generated from the config.
    """
    # `cls` is the config (config is None).
    if config is None and isinstance(cls, (dict, str)):
        config = cls
        cls = None
    # `config` is already a created object of this class ->
    # Take it as is.
    elif isinstance(cls, type) and isinstance(config, cls):
        return config

    # `type_`: Indicator for the Configurable's constructor.
    # `ctor_args`: *args arguments for the constructor.
    # `ctor_kwargs`: **kwargs arguments for the constructor.
    # Try to copy, so caller can reuse safely.
    try:
        config = deepcopy(config)
    except Exception:
        pass
    if isinstance(config, dict):
        type_ = config.pop("type", None)
        if type_ is None and isinstance(cls, str):
            type_ = cls
        ctor_kwargs = config
        # Give kwargs priority over things defined in config dict.
        # This way, one can pass a generic `spec` and then override single
        # constructor parameters via the kwargs in the call to `from_config`.
        ctor_kwargs.update(kwargs)
    else:
        type_ = config
        if type_ is None and "type" in kwargs:
            type_ = kwargs.pop("type")
        ctor_kwargs = kwargs
    # Special `_args` field in kwargs for *args-utilizing constructors.
    ctor_args = force_list(ctor_kwargs.pop("_args", []))

    # Figure out the actual constructor (class) from `type_`.
    # None: Try __default__object (if no args/kwargs), only then
    # constructor of cls (using args/kwargs).
    if type_ is None:
        # We have a default constructor that was defined directly by cls
        # (not by its children).
        if cls is not None and hasattr(cls, "__default_constructor__") and \
                cls.__default_constructor__ is not None and \
                ctor_args == [] and \
                (
                        not hasattr(cls.__bases__[0],
                                    "__default_constructor__")
                        or
                        cls.__bases__[0].__default_constructor__ is None or
                        cls.__bases__[0].__default_constructor__ is not
                        cls.__default_constructor__
                ):
            constructor = cls.__default_constructor__
            # Default constructor's keywords into ctor_kwargs.
            if isinstance(constructor, partial):
                kwargs = merge_dicts(ctor_kwargs, constructor.keywords)
                constructor = partial(constructor.func, **kwargs)
                ctor_kwargs = {}  # erase to avoid duplicate kwarg error
        # No default constructor -> Try cls itself as constructor.
        else:
            constructor = cls
    # Try the __type_registry__ of this class.
    else:
        constructor = lookup_type(cls, type_)

        # Found in cls.__type_registry__.
        if constructor is not None:
            pass
        # type_ is False or None (and this value is not registered) ->
        # return value of type_.
        elif type_ is False or type_ is None:
            return type_
        # Python callable.
        elif callable(type_):
            constructor = type_
        # A string: Filename or a python module+class or a json/yaml str.
        elif isinstance(type_, str):
            if re.search("\\.(yaml|yml|json)$", type_):
                return from_file(cls, type_, *ctor_args, **ctor_kwargs)
            # Try un-json/un-yaml'ing the string into a dict.
            obj = yaml.safe_load(type_)
            if isinstance(obj, dict):
                return from_config(cls, obj)
            try:
                obj = from_config(cls, json.loads(type_))
            except json.JSONDecodeError:
                pass
            else:
                return obj

            # Test for absolute module.class path specifier.
            if type_.find(".") != -1:
                module_name, function_name = type_.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_name)
                    constructor = getattr(module, function_name)
                # Module not found.
                except (ModuleNotFoundError, ImportError, AttributeError):
                    pass

            # If constructor still not found, try attaching cls' module,
            # then look for type_ in there.
            if constructor is None:
                if isinstance(cls, str):
                    # Module found, but doesn't have the specified
                    # c'tor/function.
                    raise ValueError(
                        f"Full classpath specifier ({type_}) must be a valid "
                        "full [module].[class] string! E.g.: "
                        "`my.cool.module.MyCoolClass`.")

                try:
                    module = importlib.import_module(cls.__module__)
                    constructor = getattr(module, type_)
                except (ModuleNotFoundError, ImportError, AttributeError):
                    # Try the package as well.
                    try:
                        package_name = importlib.import_module(
                            cls.__module__).__package__
                        module = __import__(package_name, fromlist=[type_])
                        constructor = getattr(module, type_)
                    except (ModuleNotFoundError, ImportError, AttributeError):
                        pass

            if constructor is None:
                raise ValueError(
                    f"String specifier ({type_}) must be a valid filename, "
                    f"a [module].[class], a class within '{cls.__module__}', "
                    f"or a key into {cls.__name__}.__type_registry__!")

    if not constructor:
        raise TypeError(
            "Invalid type '{}'. Cannot create `from_config`.".format(type_))

    # Create object with inferred constructor.
    try:
        object_ = constructor(*ctor_args, **ctor_kwargs)
    # Catch attempts to construct from an abstract class and return None.
    except TypeError as e:
        if re.match("Can't instantiate abstract class", e.args[0]):
            return None
        raise e  # Re-raise
    # No sanity check for fake (lambda)-"constructors".
    if type(constructor).__name__ != "function":
        assert isinstance(
            object_, constructor.func
            if isinstance(constructor, partial) else constructor)

    return object_


def from_file(cls, filename, *args, **kwargs):
    """
    Create object from config saved in filename. Expects json or yaml file.

    Args:
        filename (str): File containing the config (json or yaml).

    Returns:
        any: The object generated from the file.
    """
    path = os.path.join(os.getcwd(), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError("File '{}' not found!".format(filename))

    with open(path, "rt") as fp:
        if path.endswith(".yaml") or path.endswith(".yml"):
            config = yaml.safe_load(fp)
        else:
            config = json.load(fp)

    # Add possible *args.
    config["_args"] = args
    return from_config(cls, config=config, **kwargs)


def lookup_type(cls, type_):
    if cls is not None and hasattr(cls, "__type_registry__") and \
            isinstance(cls.__type_registry__, dict) and (
            type_ in cls.__type_registry__ or (
            isinstance(type_, str) and
            re.sub("[\\W_]", "", type_.lower()) in cls.__type_registry__)):
        available_class_for_type = cls.__type_registry__.get(type_)
        if available_class_for_type is None:
            available_class_for_type = \
                cls.__type_registry__[re.sub("[\\W_]", "", type_.lower())]
        return available_class_for_type
    return None
