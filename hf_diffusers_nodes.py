import torch
import folder_paths
import torchvision
import torch.nn as nn
import functools
import torch
import json

try:
    from .config import config_settings
except ImportError as e:
    from config import config_settings

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def ETK_HF_Diffusers_base(cls):
    cls.CATEGORY = "ETK/HF/Diffusers"
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


@ETK_HF_Diffusers_base
class DDPMPipline:
    """A node that loads the specified DDPMPipeline, supports only from_pretrained for now"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "johnowhitaker/ddpm-butterflies-32px"}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("DDPM_PL",)
    FUNCTION = "load_from_pretrained"

    def load_from_pretrained(self, model_name, device):
        from diffusers import DDPMPipeline
        ret = DDPMPipeline.from_pretrained(model_name, device=device)
        return (ret,)


@ETK_HF_Diffusers_base
class SampleDDPMPipline:
    """A node that samples from the specified DDPMPipeline"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DDPM_PL", {}),
                "batch_size": ("INT", {"default": 1}),
                "num_timesteps": ("INT", {"default": 1000}),
            },
        }

    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "sample"

    def sample(
            self,
            model,
            batch_size,
            num_timesteps,
    ):
        ret = model(
            batch_size=batch_size,
            generator=None,
            num_inference_steps=num_timesteps,
            output_type="tensor",
        )
        ret = torch.tensor(ret.images)
        return (ret,)


@ETK_HF_Diffusers_base
class TensorToImage:
    """A node that converts a tensor to an image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TORCH_TENSOR", {}),
                "incoming_format": (["BHWC", "BCHW","HWC"], {"default": "BHWC"}
                                    ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tensor_to_image"

    def tensor_to_image(self, tensor, incoming_format):
        # outgoing format is BHWC torch tensor float 32 0-1

        if incoming_format == "BCHW":
            tensor = tensor.permute(0, 2, 3, 1)
        elif incoming_format == "HWC":
            tensor = tensor.unsqueeze(0)
        tensor = tensor.clamp(0, 1)

        return (tensor,)

@ETK_HF_Diffusers_base
class LoadHFDataset:
    """A node that loads the specified HFDataset"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": ("STRING", {"default": "huggan/smithsonian_butterflies_subset"}),
            },
        }

    RETURN_TYPES = ("HF_DATASET",)
    FUNCTION = "load_from_hf"

    def load_from_hf(self, dataset_name):
        from datasets import load_dataset
        ret = load_dataset(dataset_name,
                           split="train",
                           )
        return (ret,)