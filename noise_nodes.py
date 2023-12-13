import torch
import folder_paths
import torchvision
import torch.nn as nn
import functools
import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def ETK_noise_base(cls):
    cls.CATEGORY = "ETK/noise"
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


@ETK_noise_base
class PerlinFractalImage:
    """
    use noise.perlin_fractal_image to generate a torch tensor (b,h,w,c=3)
    """

    # noise = torch.cat((noise, noise, noise), dim=3)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 256}),
                "height": ("INT", {"default": 256}),
                "scale": ("FLOAT", {"default": 0.01}),
                "octaves": ("INT", {"default": 6}),
                "lacunarity": ("FLOAT", {"default": 3}),
                "gain": ("FLOAT", {"default": 0.5}),
                "seed": ("INT", {"default": 1}),
                "offset": ("FLOAT", {"default": 0}),
                "workers": ("INT", {"default": 4}),
                "norm_zero_to_1": ([True, False], {"default": True}),

            }
        }

    # Define the return types
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("Image", "Grayscale Image", "Mask",)
    FUNCTION = "handler"

    # Method to download the dataset
    def handler(self, **kwargs):
        from .modules import noise
        width = kwargs["width"]
        height = kwargs["height"]
        scale = kwargs["scale"]
        octaves = kwargs["octaves"]
        lacunarity = kwargs["lacunarity"]
        gain = kwargs["gain"]
        seed = kwargs["seed"]
        offset = kwargs["offset"]
        workers = kwargs["workers"]
        norm_zero_to_1 = kwargs["norm_zero_to_1"]

        noise = noise.get_perlin_2d_as_torch_image_stack(width, height, scale, octaves, lacunarity, gain, seed, offset,
                                                         workers, norm_zero_to_1)

        noise_image = torch.cat((noise, noise, noise), dim=3)

        # format for this according to comfyui is unknown atm
        noise_grayscale = noise

        # format for this according to comfyui is unknown atm
        noise_mask = noise_grayscale > torch.median(noise_grayscale)
        noise_mask = noise_mask.float()

        return (noise_image, noise_grayscale, noise_mask,)
