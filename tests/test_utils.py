import torch
import numpy as np
from custom_nodes.EternalKernelLiteGraphNodes.utils import etk_deep_copy


def test_etk_deep_copy():
    original = {
        "list": [1, 2, 3, [4, 5, 6], {"inner_list": [7, 8, 9]}],
        "dict": {"inner_dict": {"key": "value"}}
    }
    torch_test = {"torch_tensor": torch.tensor([1, 2, 3])}
    np_test = {"np_array": np.array([1, 2, 3])}

    test(original)
    test(torch_test)
    test(np_test)


def test(original):
    # Creating a deep copy
    copy = etk_deep_copy(original)

    # Modifying the original structure
    if "list" in original:
        original["list"][0] = 100
        if isinstance(original["list"][3], list):
            original["list"][3][0] = 200
        if "inner_list" in original["list"][4]:
            original["list"][4]["inner_list"][0] = 300

    if "dict" in original and "inner_dict" in original["dict"]:
        original["dict"]["inner_dict"]["key"] = "modified"

    if "torch_tensor" in original:
        original["torch_tensor"][0] = 100

    if "np_array" in original:
        original["np_array"][0] = 200

    # Returning both the original and the copy for inspection
    return original, copy
