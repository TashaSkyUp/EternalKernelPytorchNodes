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

    RETURN_TYPES = ("TORCH_TENSOR",)
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
                "incoming_format": (["BHWC", "BCHW", "HWC"], {"default": "BHWC"}
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


@ETK_HF_Diffusers_base
class CreateUnet2dModel:
    """A node that creates a unet2d model"""

    @classmethod
    def INPUT_TYPES(cls):
        down_blocks = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        mid_blocks = ('UNetMidBlock2D', 'UnCLIPUNetMidBlock2D')
        up_blocks = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")

        return {
            "required": {
                "name": ("STRING", {"default": "UNet2D"}),
            },
            "optional": {
                "sample_size": ("TUPLE", {"default": (32, 32, 3,)}),
                "in_channels": ("INT", {"default": 3}),
                "out_channels": ("INT", {"default": 3}),
                "norm_num_groups": ("INT", {"default": 32}),
                "layers_per_block": ("INT", {"default": 2}),
                "down_block_types": ("TUPLE", {"default": (down_blocks[0],)}),
                "mid_block_scale_factor": ("FLOAT", {"default": 1.0}),
                "up_block_types": ("TUPLE", {"default": (up_blocks[0],)}),
                "block_out_channels": ("TUPLE", {"default": (64, 128, 256, 512,)}),
            },
        }

    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "create_unet2d_model"

    def create_unet2d_model(self, name, sample_size, in_channels, out_channels, norm_num_groups, layers_per_block,
                            down_block_types=("AttnDownBlock2D",),
                            mid_block_scale_factor=("UNetMidBlock2D",),
                            up_block_types=("AttnUpBlock2D",),
                            block_out_channels=(64, 128, 256, 512,)
                            ):
        from diffusers import UNet2DModel
        with torch.inference_mode(False):
            ret = UNet2DModel(
                sample_size=sample_size,
                in_channels=in_channels,
                out_channels=out_channels,
                norm_num_groups=norm_num_groups,
                layers_per_block=layers_per_block,
                down_block_types=down_block_types,
                mid_block_scale_factor=mid_block_scale_factor,
                up_block_types=up_block_types,
                block_out_channels=block_out_channels,

            )
        for param in ret.parameters():
            param.requires_grad = True
            #param.retains_grad = True

        return (ret,)

@ETK_HF_Diffusers_base
class TrainUnet2dModel:
    """A node that trains a unet2d model"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TORCH_MODEL", {}),
                "dataset": ("HF_DATASET", {}),
                "batch_size": ("INT", {"default": 1}),
                "num_epochs": ("INT", {"default": 1}),
                "learning_rate": ("FLOAT", {"default": 1e-4}),
            },
        }

    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "train_unet2d_model"

    def train_unet2d_model(self, model, dataset, batch_size, num_epochs, learning_rate):
        from torch.utils.data import DataLoader
        # import mse loss
        from torch.nn import MSELoss
        from torch.optim import Adam
        from diffusers import UNet2DModel


        model = model.train()
        model = model.cuda()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss_fn = MSELoss()

        for epoch in range(num_epochs):
            for batch in dataloader:
                batch = batch["image"]
                batch = batch.cuda()
                optimizer.zero_grad()
                loss = loss_fn(model, batch)
                loss.backward()
                optimizer.step()
                print(loss.item())

        return (model,)

# @ETK_HF_Diffusers_base
# class LoadModelFromHF:
#     """A node that loads the specified model from huggingface"""
#
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "model_name": ("STRING", {"default": "johnowhitaker/ddpm-butterflies-32px"}),
#             },
#         }
#
#     RETURN_TYPES = ("TORCH_MODEL",)
#     FUNCTION = "load_model_from_hf"
#
#     def load_model_from_hf(self, model_name):
#         from huggingface_hub import hf_hub_url
#         from diffusers import DDPMPipeline
#         ret = DDPMPipeline.from_pretrained(model_name, device="cuda")
#         return (ret,)




