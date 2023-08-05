import copy

from collections import OrderedDict


def etk_deep_copy(x, verbose=False):
    if type(x).__module__ == "numpy":
        return x.copy()
    elif type(x).__module__ == "torch":
        return x.clone()
    elif isinstance(x, list):
        return [etk_deep_copy(item) for item in x]
    elif isinstance(x, tuple):
        return tuple(etk_deep_copy(item) for item in x)
    elif isinstance(x, dict):
        return {k: etk_deep_copy(v) for k, v in x.items()}
    elif isinstance(x, set):
        return set(etk_deep_copy(item) for item in x)
    elif isinstance(x, OrderedDict):
        return OrderedDict((k, etk_deep_copy(v)) for k, v in x.items())
    else:
        if verbose:
            print(f"copying {x}, of type {type(x)}")

        return copy.deepcopy(x)
