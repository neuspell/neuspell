import importlib


def is_module_available(module_name: str):
    """
    checks if a module is available or not (eg. _is_module_available("sentence_transformers"))
    Args:
        module_name (str): name of the model to check
    Returns:
        bool, if or not the given module exists
    """
    try:
        importlib.import_module(module_name)
    except Exception as e:
        # print(e)
        return False

    return True


def get_module_or_attr(module_name: str, func_name: str = None):
    """
    Loads an attribute from a module or a module itself
    (check if the module exists before calling this function)
    """
    m = importlib.import_module(module_name)
    if not func_name:
        return m
    if func_name not in dir(m):
        raise ImportError(f"Cannot import {func_name} from {module_name}")
    return getattr(m, func_name)
