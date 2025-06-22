try:
    from .config import config_settings
except ImportError as e:
    from config import config_settings

import functools
import torch
import torchvision

#NODE_CLASS_MAPPINGS = {}
#NODE_DISPLAY_NAME_MAPPINGS = {}
from . import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def ETK_torchvision_base(cls):
    cls.CATEGORY = "ETK/torchvision"
    # Add spaces to the camel case class name
    pretty_name = cls.__name__
    for i in range(1, len(pretty_name)):
        if pretty_name[i].isupper():
            pretty_name = pretty_name[:i] + " " + pretty_name[i:]
    cls.DISPLAY_NAME = pretty_name
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    NODE_DISPLAY_NAME_MAPPINGS[cls.DISPLAY_NAME] = pretty_name

    # Wrap the function defined in the FUNCTION attribute
    func_name = getattr(cls, "FUNCTION", None)
    if func_name and hasattr(cls, func_name):
        original_func = getattr(cls, func_name)

        @functools.wraps(original_func)
        def wrapped_func(*args, **kwargs):
            with torch.inference_mode(False):
                return original_func(*args, **kwargs)

        setattr(cls, func_name, wrapped_func)

    return cls


@ETK_torchvision_base
class TorchVisionTransformCompositionList:
    """Provides the base list that helps define a composition pipeline for torchvision transforms"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "t_comp_00"}),
            },
        }

    RETURN_TYPES = ("TV_COMPOSITION_LIST", "STRING")
    RETURN_NAMES = ("transforms_list", "name")
    FUNCTION = "compose"

    def compose(self, name):
        transforms = []
        return (transforms, name,)


@ETK_torchvision_base
class TorchVisionTransformStringParser:
    """Parses a newline separated list of torchvision transform calls.

    Each line should be either ``TransformName`` or ``TransformName(arg1, arg2)``.
    This node helps users quickly define a composition pipeline without having
    to add a separate node for each transform.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import inspect
        import torchvision.transforms as transforms
        import torch.nn as nn

        # build default definition text listing all available transform
        # classes and their constructor parameters for quick reference
        lines = []
        for name, obj in inspect.getmembers(transforms, inspect.isclass):
            if obj.__module__.startswith("torchvision.transforms") and issubclass(obj, nn.Module):
                sig = inspect.signature(obj.__init__)
                params = [p for p in sig.parameters.values() if p.name != "self"]
                p_names = ", ".join(p.name for p in params)
                lines.append(f"{name}({p_names})")

        default_def = "\n".join(lines)

        return {
            "required": {
                "definition": ("STRING", {"default": default_def, "multiline": True}),
                "name": ("STRING", {"default": "t_comp_00"}),
            },
        }

    RETURN_TYPES = ("TV_COMPOSITION_LIST", "STRING")
    RETURN_NAMES = ("transforms_list", "name")
    FUNCTION = "compose"

    def compose(self, definition, name):
        import re

        lines = [l.strip() for l in definition.splitlines() if l.strip()]
        transforms = []
        for line in lines:
            m = re.match(r"(\w+)(?:\((.*)\))?", line)
            if not m:
                continue
            t_name = m.group(1)
            args_str = m.group(2)
            if args_str:
                args = [a.strip() for a in args_str.split(',') if a.strip()]
            else:
                args = None
            transforms.append([t_name, args])

        return (transforms, name,)


@ETK_torchvision_base
class TorchVisionTransformNode:
    """Provides the base class for torchvision transforms"""

    @classmethod
    def INPUT_TYPES(cls):
        import torchvision.transforms as transforms
        import inspect
        import torch.nn as nn

        # gather torchvision transform classes for dropdown selection
        t_list = []
        c_list = []
        for name, obj in inspect.getmembers(transforms, inspect.isclass):
            if obj.__module__.startswith("torchvision.transforms") and issubclass(obj, nn.Module):
                t_list.append([name, obj])
                c_list.append(name)

        return {
            "required": {
                "composition": ("TV_COMPOSITION_LIST", {"default": None}),
                "transform": (c_list,),
            }, "optional": {
                "args": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("TV_COMPOSITION_LIST",)

    FUNCTION = "compose"

    def compose(self, composition: list, transform, args=None):
        composition = composition.copy()

        if args:
            if not "(" in args[0] and not ")" in args[-1]:
                args = args.split(",")
            else:
                args = [args]

        composition.append([transform, args])

        return (composition,)


@ETK_torchvision_base
class TorchVisionCallComposition:
    """
    creates the composition object and calls it on the input TORCH_TENSOR or IMAGE
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "composition": ("TV_COMPOSITION_LIST", {"default": None}),
            }, "optional": {
                "tensor": ("TORCH_TENSOR",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("TORCH_TENSOR", "IMAGE", "TV_COMPOSITION")
    FUNCTION = "compose"

    def compose(self, composition, tensor=None, image=None):
        # we assume that both tensor and image are fit for the transform and that the transform is valid
        if tensor == None and image == None:
            raise ValueError("Must provide either a tensor or an image")
        if tensor != None and image != None:
            raise ValueError("Must provide either a tensor or an image, not both")

        if tensor == None and image != None:
            tensor = image

        # transform args that might be ints or floats
        composition = composition.copy()

        for c_name, c_args in composition:
            if c_args:

                if isinstance(c_args, list):
                    for i in range(len(c_args)):
                        try:
                            if "(" in c_args[i][0] and ")" in c_args[i][-1]:
                                c_args[i] = [eval(c_args[i])][0]

                            elif "." in c_args[i]:
                                c_args[i] = float(c_args[i])

                            else:
                                c_args[i] = int(c_args[i])
                        except:
                            pass

        # create the transform composition list that will be used with transforms.Compose
        transform_composition = []
        for l_c_name, l_c_args in composition:
            if l_c_name:
                if l_c_args is None:
                    l_c_args = []
                transform_composition.append(getattr(torchvision.transforms, l_c_name)(*l_c_args))

        # now that the list is composed use it to create the compostion object
        transform_composition = torchvision.transforms.Compose(transform_composition)

        # apply the transform to each item in the batch
        # many torchvision transforms expect a single image tensor of shape C,H,W
        tensor = tensor.permute(0, 3, 1, 2)  # B,C,H,W
        out_list = []
        for img in tensor:
            out_list.append(transform_composition(img))
        tensor = torch.stack(out_list, dim=0)
        tensor = tensor.permute(0, 2, 3, 1)

        return (tensor, tensor, transform_composition,)
