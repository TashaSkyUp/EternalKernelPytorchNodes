try:
    from .config import config_settings
except ImportError as e:
    from config import config_settings

import functools
import torch
import torchvision

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


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
class TorchVisionTransformNode:
    """Provides the base class for torchvision transforms"""

    @classmethod
    def INPUT_TYPES(cls):
        import torchvision.transforms as transforms
        import inspect
        import torch
        t_list = []
        c_list = []
        for i in transforms.__dict__.values():
            if "torchvision.transforms.transforms." in str(i):
                nm = str(i).split(".")[-1][:-2]
                t_list.append([nm, i])
                c_list.append(nm)

        return {
            "required": {
                "composition": ("TV_COMPOSITION_LIST", {"default": None}),
                "transform": (c_list,),
            }, "optional": {
                "args": ("STRING", {"default": None}),
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
            if c_name:
                transform_composition.append(getattr(torchvision.transforms, l_c_name)(*l_c_args))

        # now that the list is composed use it to create the compostion object
        transform_composition = torchvision.transforms.Compose(transform_composition)

        # apply the transform to the tensor
        # these transforms require b,c,h,w
        # so then permute
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = transform_composition(tensor)
        tensor = tensor.permute(0, 2, 3, 1)

        return (tensor, tensor, transform_composition,)
