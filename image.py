import torch
from PIL import Image
import numpy as np
import comfy
import folder_paths
import torch
import torchvision
import os
import hashlib

# info for code completion AI ###
"""
all of these classes are plugins for comfyui and follow the same pattern
all of the images are torch tensors and it is unknown and unimportant if they are on the cpu or gpu
all image inputs are (B,W,H,C)

avoid numpy and PIL as much as possible
"""

try:
    from . import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


# NODE_CLASS_MAPPINGS = {}
# NODE_DISPLAY_NAME_MAPPINGS = {}


# def ETK_image_base(cls):
#     # cls.FUNCTION = "func"
#     cls.CATEGORY = "ETK/Image"
#     NODE_CLASS_MAPPINGS[cls.__name__] = cls
#     return cls

def ETK_image_base(cls):
    cls.CATEGORY = "ETK/Image"
    # Add spaces to the camel case class name
    pretty_name = cls.__name__
    for i in range(1, len(pretty_name)):
        if pretty_name[i].isupper():
            pretty_name = pretty_name[:i] + " " + pretty_name[i:]
    cls.DISPLAY_NAME = pretty_name
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    NODE_DISPLAY_NAME_MAPPINGS[cls.DISPLAY_NAME] = pretty_name
    return cls


def get_fonts():
    # Directory paths for fonts in Windows and Linux
    windows_fonts_dir = 'C:\\Windows\\Fonts\\'
    linux_fonts_dir = '/usr/share/fonts/'

    # Use the appropriate directory based on the current platform
    if os.name == 'nt':  # Windows
        fonts_dir = windows_fonts_dir
    else:  # Linux and others
        fonts_dir = linux_fonts_dir

    # Initialize the list of fonts
    fonts = []

    # Check if the directory exists
    if os.path.exists(fonts_dir):
        try:
            # Traverse the directory recursively and add all .ttf and .otf files to the list
            for dirpath, dirnames, filenames in os.walk(fonts_dir):
                for filename in filenames:
                    if filename.endswith('.ttf') or filename.endswith('.otf'):
                        fonts.append(os.path.join(dirpath, filename))
        except PermissionError:
            print(f"Permission denied to access the directory: {fonts_dir}")
        except Exception as e:
            print(f"An error occurred while accessing the directory: {fonts_dir}")
            print(e)
    else:
        print(f"The directory does not exist: {fonts_dir}")

    return fonts


fonts = get_fonts()


def rgba_per_channel_norm(image):
    """ normalize per channel """
    print("normalizing per channel")

    if image[:, :, :, 0].min() != image[:, :, :, 0].max():
        image[:, :, :, 0] = image[:, :, :, 0] - image[:, :, :, 0].min()
        image[:, :, :, 0] = image[:, :, :, 0] / image[:, :, :, 0].max()

    if image[:, :, :, 1].min() != image[:, :, :, 1].max():
        image[:, :, :, 1] = image[:, :, :, 1] - image[:, :, :, 1].min()
        image[:, :, :, 1] = image[:, :, :, 1] / image[:, :, :, 1].max()

    if image[:, :, :, 2].min() != image[:, :, :, 2].max():
        image[:, :, :, 2] = image[:, :, :, 2] - image[:, :, :, 2].min()
        image[:, :, :, 2] = image[:, :, :, 2] / image[:, :, :, 2].max()

    if image[:, :, :, 3].min() != image[:, :, :, 3].max():
        image[:, :, :, 3] = image[:, :, :, 3] - image[:, :, :, 3].min()
        image[:, :, :, 3] = image[:, :, :, 3] / image[:, :, :, 3].max()
    return image


def torch_image_show(image):
    """Show a torch image"""
    from PIL import Image

    if image.size().__len__() == 4:
        image = image[0]
    if image.dtype != torch.uint8:
        image = torch.mul(image, 255, )
        image = image.to(torch.uint8)  # Convert tensor to uint8 data type

    # must have at least 3 channels
    if image.shape[-1] == 1:
        image = torch.cat((image, image, image), dim=-1)

    image = image.numpy()
    try:
        image = Image.fromarray(image)  # Create a PIL image
    except TypeError as e:
        # this is a np array not a torch tensor
        # try to rearrange the channels using np
        image = np.moveaxis(image, 0, -1)

        # image.permute(1, 2, 0)
        image = Image.fromarray(image)  # Create a PIL image

    image.show()


@ETK_image_base
class LoadImageFromPath:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "output/image.png"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, image_path: str):
        import torch
        from PIL import Image
        import numpy as np
        from pathlib import Path

        path = Path(image_path)
        loaded_image = Image.open(path)
        loaded_image = np.array(loaded_image).astype(np.float32) / 255.0
        loaded_image = torch.from_numpy(loaded_image)[None,]

        # torch tensor mus be (B,W,H,C)
        if len(loaded_image.shape) == 3:
            loaded_image = loaded_image.unsqueeze(0)
        return (loaded_image,)

    @classmethod
    def IS_CHANGED(s, image_path):
        import hashlib
        from pathlib import Path
        # use the file path and modified date + file size (all as a string) to determine if the file has changed
        path = Path(image_path)
        m = hashlib.sha256()
        m.update(str(path.stat().st_mtime).encode())
        m.update(str(path.stat().st_size).encode())
        return m.digest().hex()






@ETK_image_base
class SaveImageToPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "path": ("STRING", {"default": "output/image.png"}),
                "save_sequence": ("BOOLEAN", {"default": False}),
                "image_format": (["png", "jpg", "bmp"], {"default": "png"}),
                "jpg_quality": ("INT", {
                    "default": 70,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }),
                "png_compression": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 9,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("saved_paths",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, image: torch.Tensor, path: str, save_sequence: bool, image_format: str, jpg_quality: int,
                png_compression: int):
        import numpy as np
        from PIL import Image
        from pathlib import Path

        save_paths = []
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(image):
            # Convert torch tensor to PIL Image
            pil_image = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8))

            # Determine file path
            if save_sequence:
                file_path = path.with_stem(f"{path.stem}_{i:06d}")
            elif image.shape[0] > 1:
                file_path = path.with_stem(f"{path.stem}_{i}")
            else:
                file_path = path

            # Ensure correct file extension
            file_path = file_path.with_suffix(f".{image_format}")

            # Save image
            if image_format == "jpg":
                pil_image.save(file_path, quality=jpg_quality, optimize=True)
            elif image_format == "png":
                pil_image.save(file_path, compress_level=png_compression)
            else:  # bmp
                pil_image.save(file_path)

            save_paths.append(str(file_path))

        return (save_paths,)


@ETK_image_base
class TinyTxtToImg:
    """small text to image generator"""
    share_clip = None
    share_mdl = None
    share_vae = None

    def __init__(self):
        print("ETK> TinyTxtToImg init")
        self.mdl = None
        self.clp = None
        self.vae = None

        self.vision = None
        self.steps = 10
        self.cfg = 8
        self.sampler_name = comfy.samplers.KSampler.SAMPLERS[0]
        self.scheduler = comfy.samplers.KSampler.SCHEDULERS[0]
        self.positive = None
        self.negative = None
        self.width = 512
        self.height = 512
        self.batch_size = 1
        self.denoise = 1.0
        self.latent_image = None
        self.samples = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "prompt": ("STRING", {"multiline": True}),
                "neg_prompt": ("STRING", {"multiline": True}),
                "clip_encoder": (["comfy -ignore below", "advanced"],),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++"],),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "one_seed": ([True, False], {"default": False}),
                "unload_model": ([True, False], {"default": True}),

            },
            "optional": {
                "name": ("STRING", {"multiline": False,
                                    "default": "tinytxt2img"}
                         ),
                "overrides": ("STRING", {"multiline": True,
                                         "default": '{"width":768,"height":768,"steps":20}'}),
                "latent": ("LATENT",),
                "pos_cond": ("CONDITIONING",),
                "neg_cond": ("CONDITIONING",),
                "image": ("IMAGE",),
            }
        }

    CATEGORY = "ETK"

    RETURN_TYPES = ("IMAGE", "FUNC", "LATENT", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("image", "FUNC", "latent", "positive", "negative",)
    FUNCTION = "tinytxt2img"

    def tinytxt2img(self, **kwargs):
        """ use the imports from nodes to generate an image from text """
        from nodes import CLIPTextEncode
        import comfy.model_management as mm
        import random
        import json
        import gc
        torch.cuda.empty_cache()

        unload_model = kwargs.get("unload_model", True)
        prompt = kwargs.get("prompt", "")
        neg_prompt = kwargs.get("neg_prompt", "")
        clip_encoder = kwargs.get("clip_encoder", "comfy -ignore below")
        token_normalization = kwargs.get("token_normalization", "length+mean")
        weight_interpretation = kwargs.get("weight_interpretation", "comfy++")
        ckpt_name = kwargs.get("ckpt_name", None)

        model_ref = kwargs.get("model", None)
        vae = kwargs.get("vae", None)
        clip = kwargs.get("clip", None)

        overrides = kwargs.get("overrides", "")
        name = kwargs.get("name", "tinytxt2img")
        FUNC = kwargs.get("FUNC", None)
        render_what = kwargs.get("render_what", "image")

        self.latent_image = kwargs.get("latent", None)
        self.pos_cond = kwargs.get("pos_cond", None)
        self.neg_cond = kwargs.get("neg_cond", None)
        self.seed = kwargs.get("seed", random.randint(0, 2 ** 32 - 1))
        self.one_seed_per_batch = kwargs.get("one_seed", False)
        self.denoise = kwargs.get("denoise", 1.0)
        self.steps = kwargs.get("steps", 10)

        if model_ref and vae and clip:
            self.mdl = model_ref
            self.vae = vae
            self.clp = clip
        else:
            from nodes import CheckpointLoaderSimple
            self.mdl, self.clp, self.vae = \
                CheckpointLoaderSimple.load_checkpoint(None,
                                                       ckpt_name=ckpt_name
                                                       )
        # Check if an input image is provided
        input_image = kwargs.get("image", None)
        if input_image is not None:
            # Encode the input image using the model's VAE
            with torch.no_grad():
                self.vae.device = torch.device("cuda:0")
                self.vae.first_stage_model = self.vae.first_stage_model.to("cuda:0")
                # self.vae.first_stage_model.decoder  = self.vae.first_stage_model.decoder.to("cuda:0")
                # self.vae.first_stage_model = self.vae.first_stage_model.to("cuda:0")
                input_image = input_image.to("cuda")
                q = self.vae.encode(input_image)
                self.latent_image = {"samples": q}

        self.cfg = kwargs.get("cfg", 8)

        if not "sampler_name" in kwargs and ckpt_name:
            if "lcm" in ckpt_name.lower():
                self.sampler_name = comfy.samplers.KSampler.SAMPLERS[18]
                self.scheduler = comfy.samplers.KSampler.SCHEDULERS[0]
            else:
                self.sampler_name = comfy.samplers.KSampler.SAMPLERS[0]
                self.scheduler = comfy.samplers.KSampler.SCHEDULERS[0]
                self.scheduler = comfy.samplers.KSampler.SCHEDULERS[0]
        else:
            self.sampler_name = kwargs.get("sampler_name", comfy.samplers.KSampler.SAMPLERS[0])
            self.scheduler = kwargs.get("scheduler", comfy.samplers.KSampler.SCHEDULERS[0])

        self.positive = prompt
        self.negative = neg_prompt
        self.width = 512
        self.height = 512
        self.batch_size = 1
        if not self.denoise:
            self.denoise = 1.0

        if overrides:
            if overrides != "":
                overrides = overrides.replace("'", "\"")
                if overrides[0] != "{":
                    overrides = "{" + overrides + "}"
                valid_dict = json.loads(overrides)
                for k, v in valid_dict.items():
                    if k in self.__dict__:
                        self.__setattr__(k, v)
                    else:
                        print(f"invalid override key: {k}")
        # TODO: probably need to prepare the execution better to make sure that applying the incomming func can
        #  change everything
        if FUNC is not None:
            self.__dict__.update(FUNC(self.__dict__))

        if not self.pos_cond:
            self.clp.cond_stage_model = self.clp.cond_stage_model.to("cuda")
            if clip_encoder == "comfy -ignore below":
                self.clp_encode = lambda x: CLIPTextEncode.encode(None, self.clp, x)[0]
                self.pos_cond = self.clp_encode(self.positive)
                self.neg_cond = self.clp_encode(self.negative)

            elif clip_encoder == "advanced":
                self.clp_encode = lambda x: CLIPTextEncodeAdvanced.encode(None,
                                                                          self.clp,
                                                                          x,
                                                                          token_normalization,
                                                                          weight_interpretation)[0]
                try:
                    self.pos_cond = self.clp_encode(self.positive)
                    self.neg_cond = self.clp_encode(self.negative)

                except Exception as e:
                    print(e)
                    raise ValueError("advanced clip encoder failed")
            self.clp.cond_stage_model = self.clp.cond_stage_model.to("cpu")

            torch.cuda.empty_cache()
            gc.collect()
        if self.latent_image == None:
            from nodes import EmptyLatentImage
            self.latent_image = EmptyLatentImage().generate(self.width, self.height, self.batch_size)[0]

        if "latent" in render_what or "all" in render_what or "image" in render_what:
            self.KSampler = ETKKSampler()
            # self.mdl.offload_device = torch.device("cuda:0")
            # self.mdl.load_device = torch.device("cuda:0")
            self.samples = self.KSampler.sample(
                model=self.mdl,
                seed=self.seed,
                steps=self.steps,
                cfg=self.cfg,
                sampler_name=self.sampler_name,
                scheduler=self.scheduler,
                latent_image=self.latent_image,
                pos_cond=self.pos_cond,
                neg_cond=self.neg_cond,
                positive=self.pos_cond,
                negative=self.neg_cond,
                denoise=self.denoise,
                one_seed_per_batch=self.one_seed_per_batch)[0]

        if "image" in render_what or "all" in render_what:
            # Move the samples to the GPU
            self.samples["samples"] = self.samples["samples"].to("cuda")

            # Move VAE model to the GPU asynchronously
            self.vae.first_stage_model = self.vae.first_stage_model.to("cuda:0")
            self.vae.device = torch.device(torch.device("cuda:0"))
            torch.cuda.empty_cache()

            decoded_images = self.vae.decode(self.samples["samples"])
            image = decoded_images.to("cpu")

            self.vae.first_stage_model.to("cpu")
            self.vae.device = torch.device(torch.device("cpu"))
            torch.cuda.empty_cache()
            gc.collect()
            # Perform VAEDecode using samples in the computation stream
            # with torch.cpu.stream(cpu_stream):
            #    decoded_images = self.vae.decode(self.samples["samples"])

            # Synchronize computation stream before moving model back to CPU
            # torch.cuda.synchronize()

            # Move VAE model back to CPU asynchronously
            # self.vae.first_stage_model.to("cpu", non_blocking=True)

            # Convert the decoded image to CPU

        self.FUNC = self.tinytxt2img
        self.ARGS = kwargs

        self.samples["samples"] = self.samples["samples"].to("cpu")

        ret = (image,
               self,
               self.samples,
               self.pos_cond,
               self.neg_cond
               ,)
        if unload_model:
            del self.mdl
            del self.clp
            del self.vae
            if "vision" in self.__dict__:
                del self.vision
            del self.samples
            del self.FUNC
            del self.ARGS
            del self.pos_cond
            del self.neg_cond
            del self.KSampler
            mm.free_memory(mm.get_total_memory(mm.get_torch_device()), mm.get_torch_device())
            torch.cuda.empty_cache()
            gc.collect()
        return ret


@ETK_image_base
class PromptTemplate:
    """replaces the text in the given text string with a given other text at some key positions"""

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "text": ("STRING", {"multiline": True}),
                "replacement": ("STRING", {"multiline": True}),
                "key1": ("STRING", {"multiline": False}),
            }
        }

    CATEGORY = "text"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "prompt_template_handler"

    def prompt_template_handler(self, text: str, replacement, key1):
        """
        >>> PromptTemplate().prompt_template_handler("hello world", "universe", "world")
        'hello universe'
        """
        return (text.replace(key1, replacement),)


@ETK_image_base
class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        return {"required":
                    {"image": (sorted(os.listdir(input_dir)),)},
                }

    CATEGORY = "ETK/image"

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    FUNCTION = "load_image_handler"

    def load_image_handler(self, image):
        """
        >>> fp = './mouths.tif'
        >>> results = LoadImage().load_image_handler(fp)
        >>> torch_image_show(results[0][0])
        :param image:
        :return:
        """
        if image.endswith(".xcf"):
            return self.load_xcf(image)
        elif image.endswith(".tif"):
            return self.load_tif(image)
        else:
            return self.load_image(image)

    def load_xcf(self, image):
        pass

    def load_tif(self, image):
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image)
        img = Image.open(image_path)
        out = []

        # first find the maximum width and height
        max_width = 0;
        max_height = 0
        for i in range(img.n_frames):
            img.seek(i)
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)

        # now create a new tensor with the max width and height
        out = torch.zeros((img.n_frames, max_height, max_width, 4), dtype=torch.float)

        for i in range(img.n_frames):
            img.seek(i)
            np_arr = np.array(img).astype(np.float32) / 255.0

            padx = max_width - img.width
            pady = max_height - img.height
            np_arr = np.pad(np_arr, ((0, pady), (0, padx), (0, 0)), mode='constant')

            t_arr = torch.tensor(np_arr)
            out[i] = t_arr

        mask = out[:, :, :, 2:3]
        out_rgb = out[:, :, :, :3]
        out_rgba = out

        return (out_rgb, mask, out_rgba,)

    def load_image(self, image):
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image)
        i = Image.open(image_path)

        img_rgba = Image.open(image_path)
        img_rgba = img_rgba.convert("RGBA")
        img_rgba = np.array(img_rgba).astype(np.float32) / 255.0
        img_rgba = torch.from_numpy(img_rgba)[None,]

        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (image, mask, img_rgba)

    @classmethod
    def IS_CHANGED(s, image):
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image)
        m = hashlib.sha256()
        try:
            with open(image_path, 'rb') as f:
                m.update(f.read())
        except FileNotFoundError:
            return True
        return m.digest().hex()


@ETK_image_base
class SelectFromRGBSimilarity:
    """
    selects pixel positions based on similarity to a given color
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "r": ("FLOAT", {
                    "default": 0.500,
                    "min": 0.000,
                    "max": 1.000,
                    "step": 0.001
                }),
                "g": ("FLOAT", {
                    "default": 0.500,
                    "min": 0.000,
                    "max": 1.000,
                    "step": 0.001
                }),
                "b": ("FLOAT", {
                    "default": 0.500,
                    "min": 0.000,
                    "max": 1.000,
                    "step": 0.001
                }),
                "similarity": ("FLOAT", {
                    "default": 0.500,
                    "min": 0.000,
                    "max": 1.000,
                    "step": 0.001
                })
            }
        }

    CATEGORY = "ETK/image"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "select_from_rgb_similarity"

    def select_from_rgb_similarity(self, image: torch.Tensor, r, g, b, similarity: float = 0.5):
        """
        >>> image = torch.rand((1, 64, 64,3))
        >>> color = [0.5, 0.5, 0.5]
        >>> similarity = 0.5
        >>> mask = SelectFromRGBSimilarity.select_from_rgb_similarity(None,image,color[0],color[1],color[2],similarity)[0]
        #>>> torch_image_show(mask)
        >>> mask.shape
        torch.Size([1, 64, 64, 3])
        >>> mask.mean()
        tensor(1.)

        >>> image = torch.rand((2, 64, 64,3))
        >>> color = [0.5, 0.5, 0.5]
        >>> similarity = 0.25
        >>> mask = SelectFromRGBSimilarity.select_from_rgb_similarity(None,image,color[0],color[1],color[2],similarity)[0]
        #>>> torch_image_show(mask)
        >>> mask.shape
        torch.Size([1, 64, 64, 3])
        >>> mask.mean()
        tensor(0.34)

        :param image: torch tensor of shape (B,W,H,C)

        :param similarity: array of 3 floats
        :return: image mask of the positions where the color is similar white=selected black=not selected
        """

        batch_size, height, width, _ = image.shape
        color = [r, g, b]

        result = torch.zeros_like(image)
        # remember that the input image is BWHC
        similarity = torch.tensor(similarity)
        out = []
        for b in range(batch_size):
            # just use the first 3 channels
            use_image = image[b:b + 1, :, :, :3]
            white = torch.ones_like(use_image)

            max_distance = torch.norm(torch.tensor([1.0, 1.0, 1.0]))
            image_vector = torch.norm(use_image - torch.tensor([0, 0, 0]), dim=-1, keepdim=True)

            color_tensor = torch.tensor(color).to(image.device)
            color_tensor = color_tensor[None, None, None, :]  # Expand dimensions to match image shape for broadcasting
            color_distance = torch.linalg.vector_norm(use_image - color_tensor, dim=-1,
                                                      keepdim=True)  # Euclidean distance between color and pixels
            color_distance = color_distance / max_distance

            mask_bool = color_distance < similarity
            mask_white = white * mask_bool
            out.append(mask_white)

        out = torch.cat(out, dim=0)
        return (out,)


@ETK_image_base
class Quantize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {
                    "default": 256,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "quantize"

    CATEGORY = "ETK/image"

    def quantize(self, image: torch.Tensor, colors: int = 256):
        """
        >>> image = torch.rand((2, 512, 512,3))
        >>> colors = 256
        >>> quantized = Quantize.quantize(None,image,colors)[0]
        >>> quantized.shape
        torch.Size([2, 512, 512, 3])
        >>> image = torch.rand((2, 512, 512,4))
        >>> colors = 256
        >>> quantized = Quantize.quantize(None,image,colors)[0]
        >>> quantized.shape
        torch.Size([2, 512, 512, 4])

        """
        batch_size, height, width, _ = image.shape

        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image_start = image[b]
            if tensor_image_start.shape[2] == 4:
                tensor_image = image[b, :, :, :3]
            else:
                tensor_image = image[b]

            img = (tensor_image * 255).to(torch.uint8).numpy()
            pil_image = Image.fromarray(img, mode='RGB')

            palette = pil_image.quantize(
                colors=colors)  # Required as described in https://github.com/python-pillow/Pillow/issues/5836
            quantized_image = pil_image.quantize(colors=colors, palette=palette, dither=0)

            rgb = quantized_image.convert("RGB")
            quantized_array = torchvision.transforms.PILToTensor()(rgb).float() / 255

            # fix the indices to b, w, h, c
            quantized_array = quantized_array.permute(1, 2, 0)

            if tensor_image_start.shape[2] == 4:
                result[b][:, :, :3] = quantized_array.to(image.dtype)
                result[b][:, :, 3] = image[b, :, :, 3]
            else:
                result[b] = quantized_array.to(image.dtype)

        return (result,)


@ETK_image_base
class select_from_batch:
    """selects a single image from a batch"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }), }

            ,
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
            },
        }

    CATEGORY = "ETK/image"

    RETURN_TYPES = ("IMAGE", "LATENT",)

    FUNCTION = "select_from_batch"

    def select_from_batch(self, index=0, image=None, latent=None):
        """
        >>> image = torch.rand((1, 512, 512,3))
        >>> index = 0
        >>> selected = select_from_batch.select_from_batch(None,image,index)
        >>> selected.shape
        torch.Size([1, 512, 512, 3])
        >>> image = torch.rand((2, 512, 512,3))
        >>> selected = select_from_batch.select_from_batch(None,image,1)
        >>> selected.shape
        torch.Size([1, 512, 512, 3])

        """
        olat = None
        oimg = None
        o_lat_dict = {}
        if image is None:
            print(latent["samples"].shape)
            smpls = latent["samples"][index:index + 1, :, :, :].expand(1, -1, -1, -1).clone()
            o_lat_dict["samples"] = smpls
            olat = o_lat_dict
            print(olat["samples"].shape)
        else:
            oimg = image[index:index + 1, :, :, :].clone()
            print(oimg.shape)

        return (oimg, olat,)


@ETK_image_base
class ImageDistanceMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10240,
                    "step": 1
                }),
                "y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10240,
                    "step": 1
                }),

                "Z": ("FLOAT", {
                    "default": 1,
                    "min": -2,
                    "max": 2,
                    "step": .05
                }),
                "aspect_ratio": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 2,
                    "step": .05
                }),

                "falloff_method": (["linear", "spherical"],),

            }
        }

    CATEGORY = "ETK/image"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "distance_mask_image"

    def distance_mask_image(self, image, x, y, falloff_method=None, Z=1, aspect_ratio=1.0):
        B, H, W, C = image.shape

        # Create a meshgrid for the coordinates
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))

        # Adjust aspect ratio
        grid_x = grid_x * aspect_ratio
        x = x * aspect_ratio

        # Ensure grid_x and grid_y are on the same device and dtype as the image
        grid_x, grid_y = grid_x.to(image.device).type(image.dtype), grid_y.to(image.device).type(image.dtype)

        # Calculate distance from (x, y) to each pixel
        dist = torch.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)

        # Normalize the distance to be between 0 and 1
        dist = (dist - dist.min()) / (dist.max() - dist.min())

        if falloff_method:
            if falloff_method == "linear":
                dist = 1 - dist
            elif falloff_method == "spherical":
                dist = torch.sqrt(1 - dist ** 2)

        if Z < 0:
            dist = 1 - dist
            Z = abs(Z)

        dist = dist - (1 - Z)
        dist = torch.clamp(dist, 0, 1)

        # Normalize again after the modifications
        dist = (dist - dist.min()) / (dist.max() - dist.min())

        # Extend dimensionality to match the input tensor (B, H, W, C)
        dist = dist[None, ..., None].expand(B, H, W, C)

        return (dist,)


@ETK_image_base
class rgba_lower_clip:
    """Clip RGBA values below a threshold"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "nrm_after": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1

                }),
                "r": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "g": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "b": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "a": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),

            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rgba_lower_clip"

    CATEGORY = "ETK/image"

    def rgba_lower_clip(self, image: torch.Tensor, nrm_after, r, g, b, a):
        # image is (B,W,H,C)
        image = image.clone()
        # check for alpha channel, add if not present

        if image.shape[3] == 3:
            alpha = torch.ones_like(image[:, :, :, 0])
            image = torch.cat((image, alpha.unsqueeze(3)), dim=3)

        image[:, :, :, 0] = torch.clamp(image[:, :, :, 0], min=r, max=1.0)
        image[:, :, :, 1] = torch.clamp(image[:, :, :, 1], min=g, max=1.0)
        image[:, :, :, 2] = torch.clamp(image[:, :, :, 2], min=b, max=1.0)
        image[:, :, :, 3] = torch.clamp(image[:, :, :, 3], min=a, max=1.0)

        if int(nrm_after) == 1:
            image = rgba_per_channel_norm(image)

        return (image,)


@ETK_image_base
class ImageStackBlendByMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stack a": ("IMAGE",),
                "stack b": ("IMAGE",),
                "img mask": ("IMAGE",),

            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("image", "image rgba")
    FUNCTION = "stack_mask"

    CATEGORY = "ETK/Image"

    def stack_mask(self, **kwargs):
        import torch
        # use torch to blend two images by a mask
        # stacks are already (B,W,H,C)

        stack_a = kwargs["stack a"]
        stack_b = kwargs["stack b"]
        img_mask = kwargs["img mask"]
        # check if they all have alpha
        if stack_a.shape[3] == 3:
            alpha = torch.ones_like(stack_a[:, :, :, 0])
            stack_a = torch.cat((stack_a, alpha.unsqueeze(3)), dim=3)

        if stack_b.shape[3] == 3:
            alpha = torch.ones_like(stack_b[:, :, :, 0])
            stack_b = torch.cat((stack_b, alpha.unsqueeze(3)), dim=3)

        # allocate new image
        new_image = torch.zeros_like(stack_a)

        # blend rgb channels

        for i in range(3):
            new_image[:, :, :, i] = \
                (stack_a[:, :, :, i] * img_mask[:, :, :, 0]) \
                + \
                (stack_b[:, :, :, i] * (1 - img_mask[:, :, :, 0]))

        # alpha should be the mean of the two alphas
        new_image[:, :, :, 3] = (stack_a[:, :, :, 3] + stack_b[:, :, :, 3]) / 2
        rgba = new_image
        rgb = new_image[:, :, :, 0:3]

        return (rgb, rgba,)


@ETK_image_base
class rgba_upper_clip:
    """Clip RGBA values above a threshold"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "nrm_after": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1

                }),
                "r": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "g": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "b": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "a": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),

            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rgba_upper_clip"

    CATEGORY = "ETK/image"

    def rgba_upper_clip(self, image: torch.Tensor, nrm_after, r, g, b, a):
        # image is (B,W,H,C)
        image = image.clone()
        # check for alpha channel, add if not present

        if image.shape[3] == 3:
            alpha = torch.ones_like(image[:, :, :, 0])
            image = torch.cat((image, alpha.unsqueeze(3)), dim=3)

        image[:, :, :, 0] = torch.clamp(image[:, :, :, 0], min=0, max=r)
        image[:, :, :, 1] = torch.clamp(image[:, :, :, 1], min=0, max=g)
        image[:, :, :, 2] = torch.clamp(image[:, :, :, 2], min=0, max=b)
        image[:, :, :, 3] = torch.clamp(image[:, :, :, 3], min=0, max=a)

        if int(nrm_after) == 1:
            image = rgba_per_channel_norm(image)

        return (image,)


@ETK_image_base
class RGBA_MOD:
    """provide sliders to modify RGBA values"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "r": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "g": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "b": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "a": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rgba_mod"

    CATEGORY = "ETK/image"

    def rgba_mod(self, image: torch.Tensor, r, g, b, a):
        """simply multiply the RGBA values by the sliders
        images are (B, W, H, C)
        """
        # check for alpha channel
        if image.shape[3] == 3:
            image = torch.cat((image, torch.ones_like(image[:, :, :, 0:1])), dim=3)

        image = image.clone()
        image[:, :, :, 0] = image[:, :, :, 0] * r
        image[:, :, :, 1] = image[:, :, :, 1] * g
        image[:, :, :, 2] = image[:, :, :, 2] * b
        image[:, :, :, 3] = image[:, :, :, 3] * a

        return (image,)


@ETK_image_base
class ImageBC:
    """Normalize an images brightness,droping lower and higher values of n perctile"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_b_c"

    CATEGORY = "ETK/image"

    def image_b_c(self, image, contrast, brightness):
        """
        Adjusts the contrast of an image, while preserving the alpha channel

        >>> image = (torch.rand(2, 256, 256, 4)/10)+.45
        >>> image[:,0,0,:] = torch.tensor([0.0,0.0,0.0,0.0])
        >>> image[:,1,1,:] = torch.tensor([1.0,1.0,1.0,1.0])
        >>> torch_image_show(image)
        >>> normalized_image = ImageBC.image_b_c(None,image,2,2)
        >>> torch_image_show(normalized_image)
        >>> normalized_image.shape
        torch.Size([2, 256, 256, 4])
        """
        # check for alpha channel
        if image.shape[3] == 3:
            image = torch.cat((image, torch.ones_like(image[:, :, :, 0:1])), dim=3)

        # get the image size
        size = image.shape

        # Separate the alpha channel
        img_a = image[..., -1:]
        image = image[..., :-1]

        # Permute the dimensions to [..., 1 or 3, H, W]
        image = image.permute(0, 3, 1, 2)

        # Adjust the contrast
        image = torchvision.transforms.functional.adjust_brightness(image, brightness)
        image = torchvision.transforms.functional.adjust_contrast(image, contrast)

        # Permute the dimensions back to [Batch, H, W, Channels]
        image = image.permute(0, 2, 3, 1)

        # Concatenate the alpha channel back
        image = torch.cat((image, img_a), dim=-1)

        return (image,)


@ETK_image_base
class PadToMatch:
    """ a node that will pad the smaller of two images to match the size of the larger one """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "im2_x_center": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "im2_y_center": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "padme"

    CATEGORY = "ETK/image"

    def padme(self, image1, image2, im2_x_center, im2_y_center):
        """
        pads the smaller of two images to match the size of the larger one
        x_center and y_center are how far to the right and down the smaller image should be placed
        images are (B, W, H, C)
        >>> PadToMatch.padme(None,torch.ones((1, 2, 2, 3)), torch.ones((1, 3, 3, 3)), 0.5, 0.5)[0].mean(axis=(3))
        tensor([[[1., 1., 0.],
                 [1., 1., 0.],
                 [0., 0., 0.]]])


        """
        target_width = max(image1.shape[1], image2.shape[1])
        target_height = max(image1.shape[2], image2.shape[2])

        padded1 = torch.zeros((image1.shape[0], target_width, target_height, image1.shape[3]))
        padded2 = torch.zeros((image2.shape[0], target_width, target_height, image2.shape[3]))

        pad_w_1 = target_width - image1.shape[1]
        pad_h_1 = target_height - image1.shape[2]

        # img 1 is taken as the centered one
        xs = int(pad_w_1 / 2)
        xe = xs + image1.shape[1]
        ys = int(pad_h_1 / 2)
        ye = ys + image1.shape[2]
        padded1[:, xs:xe, ys:ye, :] = image1

        # img 2 is placed according to the centering parameters
        pad_w_2 = target_width - image2.shape[1]
        pad_h_2 = target_height - image2.shape[2]

        xs = int(pad_w_2 * im2_x_center)
        ys = int(pad_h_2 * im2_y_center)
        xe = xs + image2.shape[1]
        ye = ys + image2.shape[2]

        if xe == 0:
            xe = None
        if ye == 0:
            ye = None
        padded2[:, xs:xe, ys:ye, :] = image2

        return (padded1, padded2)


@ETK_image_base
class StackImages:
    """ a node that will stack two images on top of each other """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "number_of_images": ("INT", {"default": 2, "min": 2, "max": 100}),
                # an optionbox with "all" or "upto", model after "one seed per batch" in the sampler
                "mode": (["all", "one_for_one", "upto"], {"default": "all"}),
                "number_of_repeats": ("INT", {"default": 1, "min": 1, "max": 100}),

            }, }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stackme"

    CATEGORY = "ETK/image"

    def stackme(self, image1, image2=None, number_of_images=2, mode="all", number_of_repeats=1):
        # import modules to help clear memory
        import gc
        import torch.cuda

        """
        stacks two images on top of each other
        images are (B, W, H, C) torch tensors
        """
        if image1 is not None:
            image1 = image1.clone()

        if image2 is not None:
            image2 = image2.clone()

        # gc.collect()
        # torch.cuda.empty_cache()

        del gc.garbage[:]
        gc.collect()
        torch.cuda.empty_cache()

        # apply number of repeats

        if mode == "all" and image2 is not None:
            image2 = image2.clone()
            out = torch.cat((image1, image2), dim=0)
        elif mode == "all" and image2 is None:
            out = image1.clone()
        elif mode == "one_for_one" and image2 is not None:
            # for each image in image1, add each to the output
            out_size = min(image1.shape[0], image2.shape[0])
            holder = torch.zeros((out_size, image1.shape[1], image1.shape[2], image1.shape[3]))
            for i in range(out_size):
                holder[i, :, :, :] = image2[i, :, :, :]
            out = holder
        elif mode == "one_for_one" and image2 is None:
            # this might mean that the user whats each index repeated num times
            out_size = number_of_repeats * image1.shape[0]
            out = torch.zeros((out_size, image1.shape[1], image1.shape[2], image1.shape[3]))
            for i in range(out_size):
                out[i, :, :, :] = image1[i % image1.shape[0], :, :, :]

        elif mode == "upto" and image2 is not None:
            # stack them all together but then only return the first number_of_images
            out = torch.cat((image1, image2), dim=0)
            out = out[:number_of_images, :, :, :]

        elif mode == "upto" and image2 is None:
            out = image1.clone()
            out = out[:number_of_images, :, :, :]

        if number_of_repeats > 1:
            out = out.repeat(number_of_repeats, 1, 1, 1)

        # make sure to apply number of images
        out = out[:number_of_images, :, :, :]

        torch.cuda.empty_cache()
        gc.collect()

        return (out,)


@ETK_image_base
class rgba_merge:
    """merges the channels of an RGBA image"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "r": ("IMAGE",),
                "g": ("IMAGE",),
                "b": ("IMAGE",),
                "a": ("IMAGE",),
            }
        }

    CATEGORY = "ETK/image"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "merge"

    def merge(self, r, g, b, a):
        """
        r,g,b,a all may be 3, 1, or 4 channel images
        >>> image = torch.rand((1, 512, 512,3))
        >>> r, g, b, a = rgba_split.split(None,image)
        >>> rgba_merge.merge(None,r,g,b,a).shape
        torch.Size([1, 512, 512, 4])
        :param r:
        :param g:
        :param b:
        :param a:
        :return:
        """
        # use channel 0 of each channel in the resulting image
        r = r[:, :, :, 0:1]
        g = g[:, :, :, 0:1]
        b = b[:, :, :, 0:1]
        a = a[:, :, :, 0:1]

        # there will always be an alpha channel so just put the channels together

        return (torch.cat((r, g, b, a), dim=3),)


@ETK_image_base
class rgba_split:
    """splits the channels of an RGBA image"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    CATEGORY = "ETK/image"

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")

    FUNCTION = "split"

    def split(self, image):
        """
        >>> image = torch.rand((1, 512, 512,3))
        >>> r, g, b, a = rgba_split.split(None,image)
        >>> r.shape
        torch.Size([1, 1, 512, 512])
        :param image:
        :return:
        """
        # check for alhpa channel
        if image.shape[3] == 3:
            # add alpha channel
            alpha = torch.ones((image.shape[0], image.shape[1], image.shape[2], 1))
            image = torch.cat((image, alpha), 3)
        # for now each channel must be (b,w,h,3) so repeat  the channel 3 times
        r = image[:, :, :, 0].unsqueeze(3).repeat(1, 1, 1, 3)
        g = image[:, :, :, 1].unsqueeze(3).repeat(1, 1, 1, 3)
        b = image[:, :, :, 2].unsqueeze(3).repeat(1, 1, 1, 3)
        a = image[:, :, :, 3].unsqueeze(3).repeat(1, 1, 1, 3)

        return (r, g, b, a,)


def get_image_size(image):
    """ returns the width and height of an image """
    size = image.shape
    width = int(size[0])
    height = int(size[1])
    return (width * height)


@ETK_image_base
class KSamplerCreateDataset:
    """ use KSampler to create a dataset """

    @classmethod
    def INPUT_TYPES(cls):
        inps = ETKKSampler.INPUT_TYPES()
        # make sure to set a default
        inps["required"]["dataset_name"] = ("STRING", {"default": "dataset"})
        return inps

    CATEGORY = "ETK/training"

    RETURN_TYPES = ("LATENT", "STRING",)
    RETURN_NAMES = ("LATENT", "dataset_path",)

    FUNCTION = "create_dataset"

    def create_dataset(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                       denoise, one_seed_per_batch, dataset_name):
        """
        creates a dataset using KSampler
        :param model:
        :param seed:
        :param steps:
        :param cfg:
        :param sampler_name:
        :param scheduler:
        :param positive:
        :param negative:
        :param latent_image:
        :param denoise:
        :param dataset_name:
        :return:
        """
        # create the sampler
        sampler = ETKKSampler()

        x = {"model": model, "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler_name,
             "scheduler": scheduler, "positive": positive, "negative": negative, "latent_image": latent_image,
             "denoise": denoise, "one_seed_per_batch": one_seed_per_batch}

        # now save x
        # just use torch .save?
        import torch
        # create a random name
        import hashlib
        import time

        # create a random name
        name = hashlib.md5(str(time.time()).encode()).hexdigest()[0:8]
        name = dataset_name + "_" + name

        o = folder_paths.get_output_directory()
        import os

        dataset_path = os.path.join(o, name)
        torch.save(x, dataset_path + "_x.pt")

        results = sampler.sample(**x)
        # save the results
        torch.save(results[0], dataset_path + "_y.pt")
        # return the dataset path
        return (results[0], dataset_path,)


@ETK_image_base
class ETKKSampler:
    @classmethod
    def INPUT_TYPES(s):
        from nodes import KSampler
        ret = KSampler.INPUT_TYPES()
        ret["required"]["one_seed_per_batch"] = ([True, False], {"default": False})
        return ret

    RETURN_TYPES = ("LATENT", "FUNC",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, **kwargs):
        from nodes import common_ksampler
        # kwargs = deepcopy(kwargs)

        model = kwargs.get("model", None)
        seed = kwargs.get("seed", None)
        steps = kwargs.get("steps", None)
        cfg = kwargs.get("cfg", None)
        sampler_name = kwargs.get("sampler_name", None)
        scheduler = kwargs.get("scheduler", None)
        positive = kwargs.get("positive", None)
        negative = kwargs.get("negative", None)
        latent_image = kwargs.get("latent_image", None)
        denoise = kwargs.get("denoise", None)
        one_seed_per_batch = kwargs.get("one_seed_per_batch", None)

        # self.FUNC = lambda x: ETKKSampler.sample(self, **x)
        # self.ARGS = kwargs

        try:

            ret = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, one_seed_per_batch=one_seed_per_batch)
        except TypeError as e:
            ret = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise)
        except RuntimeError as e:
            raise e
        ret = (ret[0], self,)
        return ret


@ETK_image_base
class ListKSampler:
    """
    A KSampler that takes a list for every input, if the list has one element it will be repeated for every sample
    """

    @classmethod
    def INPUT_TYPES(cls):
        s1 = ETKKSampler.INPUT_TYPES()["required"]["sampler_name"]
        s2 = ETKKSampler.INPUT_TYPES()["required"]["scheduler"]
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("LIST",),
                "steps": ("LIST",),
                "cfg": ("LIST",),
                "sampler_name": s1,
                "scheduler": s2,
                "positive": ("LIST",),
                "negative": ("LIST",),
                "latent_image": ("LIST",),
                "denoise": ("LIST",),
                "one_seed_per_batch": ("LIST",),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, **kwargs):
        import torch
        import random
        # figure out how many samples we need to generate
        defaults = {
            "seed": [-1],
            "steps": [15],
            "cfg": [1.0],
            "denoise": [1.0],
            "one_seed_per_batch": [False]
        }
        for k in defaults:
            if k not in kwargs:
                kwargs[k] = defaults[k]
        num_samples = 1

        for key in kwargs:
            if key == "model":
                continue
            elif key == "sampler_name" or key == "scheduler":
                kwargs[key] = [kwargs[key]]
            if kwargs[key] is not None:
                num_samples = max(num_samples, len(kwargs[key]))

        # create a list of dictionaries
        samples = []
        for i in range(num_samples):
            sample = {}
            for key, v in kwargs.items():
                if v is None:
                    v = defaults[key]

                if not isinstance(kwargs[key], list) and key != "model":
                    kwargs[key] = [v]
                if key == "model":
                    sample[key] = v
                elif key == "seed":
                    if v in [[-1], [0], [None]]:
                        sample[key] = random.randint(0, int(10e10 + 1))
                    else:
                        sample[key] = v
                elif len(v) == 1:
                    sample[key] = v[0]
                else:
                    sample[key] = v[i]
            samples.append(sample)

        # create a list of results
        results = []
        local_ksampler = ETKKSampler()
        for sample in samples:
            results.append(local_ksampler.sample(**sample)[0])
        torch.cuda.empty_cache()

        return (results,)


@ETK_image_base
class ListCustomSampler:
    """
    A Custom sampler that takes a list for every input, if the list has one element it will be repeated for every sample
    """

    @classmethod
    def INPUT_TYPES(cls):
        s1 = ETKKSampler.INPUT_TYPES()["required"]["sampler_name"]
        s2 = ETKKSampler.INPUT_TYPES()["required"]["scheduler"]
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "noise_seed": ("LIST",),
                "cfg": ("LIST",),
                "positive": ("LIST",),
                "negative": ("LIST",),
                "latent_image": ("LIST",),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, **kwargs):
        import torch
        # figure out how many samples we need to generate
        num_samples = 1
        for key in kwargs:
            if key == "model":
                continue
            elif key == "sampler_name" or key == "scheduler":
                continue
            if isinstance(kwargs[key], list):
                num_samples = max(num_samples, len(kwargs[key]))
            else:
                pass

        # create a list of dictionaries
        samples = []
        for i in range(num_samples):
            sample = {}
            for key in kwargs:
                if not isinstance(kwargs[key], list) and key != "model":
                    kwargs[key] = [kwargs[key]]
                if key == "model":
                    sample[key] = kwargs[key]
                elif kwargs[key] is None:
                    sample[key] = None
                elif len(kwargs[key]) == 1:
                    sample[key] = kwargs[key][0]
                else:
                    sample[key] = kwargs[key][i]
            samples.append(sample)

        # create a list of results
        results = []
        from comfy_extras.nodes_custom_sampler import SamplerCustom
        local_ksampler = SamplerCustom()
        for sample in samples:
            results.append(local_ksampler.sample(**sample)[0])
        torch.cuda.empty_cache()

        return (results,)


@ETK_image_base
class ListCLIPTextEncode:
    """
    A CLIPTextEncode that takes a list for every input, if the list has one element it will be repeated for every sample
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("LIST",),
                "clip": ("LIST",),

            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "encode"

    CATEGORY = "CLIP"

    def encode(self, **kwargs):
        # figure out how many samples we need to generate
        num_samples = 1
        for key in kwargs:
            num_samples = max(num_samples, len(kwargs[key]))

        # create a list of dictionaries
        samples = []
        for i in range(num_samples):
            sample = {}
            for key in kwargs:
                if len(kwargs[key]) == 1:
                    sample[key] = kwargs[key][0]
                else:
                    sample[key] = kwargs[key][i]
            samples.append(sample)

        # create a list of results
        results = []
        for sample in samples:
            from nodes import CLIPTextEncode
            results.append(CLIPTextEncode().encode(**sample)[0])

        return (results,)


@ETK_image_base
class ListVAEDecode:
    """
    A VAEDecode that takes a list for every input, if the list has one element it will be repeated for every sample
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LIST",),

            },
            "optional": {
                "vae_list": ("LIST",),
                "vae": ("VAE",),

            }
        }

    RETURN_TYPES = ("LIST", "IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "VAE"

    def decode(self, **kwargs):
        from nodes import VAEDecode
        import torch
        # figure out if we are using the list of vae or just one
        if kwargs.get("vae_list") is None:
            kwargs["vae"] = [kwargs.get("vae")]
        else:
            kwargs["vae"] = kwargs.pop("vae_list")

        if "vae_list" in kwargs:
            kwargs.pop("vae_list")

        # figure out how many samples we need to generate
        num_samples = 1
        for key in kwargs:
            num_samples = max(num_samples, len(kwargs[key]))

        # create a list of dictionaries
        samples = []
        for i in range(num_samples):
            sample = {}
            for key in kwargs:
                if len(kwargs[key]) == 1:
                    sample[key] = kwargs[key][0]
                else:
                    sample[key] = kwargs[key][i]
            samples.append(sample)

        # create a list of results
        results = []
        for sample in samples:
            sample["samples"] = sample.pop("latent")

            results.append(VAEDecode().decode(**sample)[0])

        # the results are a list of image tensors (BHWC) float 0-1
        # we need to stack them into a single tensor but first pre-allocte the tensor
        # we need to figure out the size of the tensor, we know though that all images are the same size
        size = results[0].shape
        width = size[1]
        height = size[2]
        channels = size[3]
        num_images = len(results)

        # also need to specify to make the final tensor on the CPU

        final_bsize = 0
        # find the bsize
        # sometimes a result will actually be shape (>1, W, H, C)
        # sometimes it will be  (W, H, C)
        # sometimes it will be (1, W, H, C)

        for i in range(num_images):
            if len(results[i].shape) == 3:
                results[i] = results[i].unsqueeze(0)
                final_bsize += 1
            elif len(results[i].shape) == 4:
                if results[i].shape[0] == 1:
                    results[i] = results[i]
                    final_bsize += 1
                else:
                    final_bsize += results[i].shape[0]

        # assign
        final_tensor = torch.zeros((final_bsize, width, height, channels), device="cpu")
        cur_idx = 0

        while True:

            if len(results[i].shape) == 4:
                if results[i].shape[0] == 1:
                    final_tensor[cur_idx] = results[i]
                    cur_idx += 1
                else:
                    for j in range(results[i].shape[0]):
                        res = results[i][j]
                        if len(res.shape) == 3:
                            res = res.unsqueeze(0)
                        final_tensor[cur_idx] = res
                        cur_idx += 1
            if cur_idx >= final_bsize:
                break

        return (results, final_tensor,)


@ETK_image_base
class ListVAEEncode:
    """
    A VAEEncode that takes a list for every input, if the list has one element it will be repeated for every sample
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("LIST",),

            },
            "optional": {
                "vae_list": ("LIST",),
                "vae": ("VAE",),

            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "encode"

    CATEGORY = "VAE"

    def encode(self, **kwargs):
        if kwargs.get("vae_list", None) is None:
            kwargs["vae"] = [kwargs.get("vae")]
        else:
            kwargs["vae"] = kwargs.pop("vae_list")

        if "vae_list" in kwargs:
            kwargs.pop("vae_list")

        # figure out how many samples we need to generate
        num_samples = 1
        for key in kwargs:
            num_samples = max(num_samples, len(kwargs[key]))

        # create a list of dictionaries
        samples = []
        for i in range(num_samples):
            sample = {}
            for key in kwargs:
                if len(kwargs[key]) == 1:
                    sample[key] = kwargs[key][0]
                else:
                    sample[key] = kwargs[key][i]
            samples.append(sample)

        # create a list of results
        results = []
        for sample in samples:
            results.append(VAEEncode().encode(**sample)[0])

        return (results,)


@ETK_image_base
class ImageFromListIdx:
    """
    A node that takes a list of images and an index and returns the image at that index
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("LIST", {"default": []}),
                "index": ("INT", {"default": 0, "min": 0, "max": 10e10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_image"

    CATEGORY = "ETK/image"

    def get_image(self, images, index):
        return (images[index],)


@ETK_image_base
class LatentFromListIdx:
    """
    A node that takes a list of latents and an index and returns the latent at that index
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LIST", {"default": []}),
                "index": ("INT", {"default": 0, "min": -10e10, "max": 10e10}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "get_latent"

    CATEGORY = "ETK/latent"

    def get_latent(self, latents, index):
        return (latents[index],)


@ETK_image_base
class ListifyAnything:
    """
    A node that takes anything and returns it in a list
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": ("*",),
                "repeat": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "listify"

    CATEGORY = "ETK/other"

    def listify(self, anything, repeat):
        return ([anything] * repeat,)


@ETK_image_base
class TextRender:
    """
    renders text to an image
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello World!"}),
                "x": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 1024,
                    "step": 1
                }),
                "y": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 1024,
                    "step": 1
                }),
                "width": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1024,
                    "step": 32
                }),
                "height": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1024,
                    "step": 32
                }),
                "font": (fonts,),
                "size": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "color": ("STRING", {
                    "default": "#000000"
                }),
                "func_only": ([True, False], {"default": False}),
                "stroke fill": ("STRING", {"default": "#000000"}),
                "stroke width": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "font_override": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "ETK/text"

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "FUNC",)
    RETURN_NAMES = ("image", "image rgba", "possible_fonts", "FUNC",)

    FUNCTION = "render_text"

    def render_text(self, text, x, y, width, height, font='Arial', size=16, color='#888888', func_only=False, **kwargs):
        """
        This function renders the provided text at specified location, with the given width and height.
        The text is rendered in the provided font, size, and color.
        """
        from PIL import ImageFont, ImageDraw
        import psutil
        process = psutil.Process();
        memory_info = process.memory_info();
        start_memory = memory_info.rss / (1024 * 1024)
        font_override = kwargs.get("font_override", None)

        if font_override:
            font = font_override

        sw = kwargs.get("stroke width", None)
        sf = kwargs.get("stroke fill", None)

        def _wrap_text(font, line, max_width, d):
            words = line.split(' ')
            new_line = ''
            lines = []
            for word in words:
                temp_line = new_line + word + ' '

                # w, _ = d.textsize(temp_line, font=font)

                # w, h = d.textsize(line, font=font)
                (left, top, right, bottom) = d.textbbox((0, 0), text=temp_line, font=font)
                w = right - left
                h = bottom - top
                previous_height = h  # Update the previous height

                if w > max_width:
                    lines.append(new_line.strip())
                    new_line = word + ' '
                else:
                    new_line = temp_line

            lines.append(new_line.strip())
            wrapped_line = '\n'.join(lines)
            return wrapped_line

        def _render_text(font_name, image, size, text, x, y, allow_wrap=True, allow_shrink=True, sw=None, sf=None):
            width = image.size[0]
            d = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_name, size)
            lines = text.split('\n')

            shrink_threshold = 24  # The minimum font size before we start wrapping text
            line_spacing = 5  # The constant space between lines
            y_offset = y
            previous_height = 0  # Store the height of the previous line

            for i, line in enumerate(lines):
                if line.strip() == '' or line.strip() == '\n':
                    # For blank lines, add a blank line of the previous line's height
                    y_offset += previous_height + line_spacing
                else:
                    # w, h = d.textsize(line, font=font)
                    (left, top, right, bottom) = d.textbbox((0, 0), text=line, font=font)
                    w = right - left
                    h = bottom - top
                    previous_height = h  # Update the previous height

                    if w > width:
                        if allow_shrink:
                            new_font_size = int(size * width / w)
                            if new_font_size >= shrink_threshold:
                                font = ImageFont.truetype(font_name, new_font_size)
                            else:
                                # The font size is too small, start wrapping text
                                wrapped_line = _wrap_text(font, line, width, d)
                                y_offset = _render_text(font_name,
                                                        image,
                                                        size,
                                                        wrapped_line, x, y_offset + line_spacing,
                                                        allow_wrap=False,
                                                        allow_shrink=False,
                                                        sf=sf,
                                                        sw=sw)
                                continue  # Skip to the next line

                        if allow_wrap:
                            # Wrap the text
                            wrapped_line = _wrap_text(font, line, width, d)
                            y_offset = _render_text(font_name,
                                                    image,
                                                    new_font_size,
                                                    wrapped_line,
                                                    x,
                                                    y_offset + line_spacing,
                                                    allow_wrap=False,
                                                    allow_shrink=False,
                                                    sf=sf,
                                                    sw=sw)
                            continue  # Skip to the next line

                    # Draw the line
                    x_offset = x + (width - w) / 2
                    if sw or sw:
                        d.text((x_offset, y_offset), line, font=font, fill="#FFFFFF", stroke_width=sw, stroke_fill=sf)
                    else:
                        d.text((x_offset, y_offset), line, font=font, fill="#FFFFFF")

                    y_offset += h + line_spacing  # Move y_offset to the bottom of the last line drawn plus the line spacing
            # debug print the amount of free memory in torch

            return y_offset

        if func_only:
            def lll(new_kwargs):
                if "font" not in new_kwargs:
                    new_kwargs["font"] = kwargs.get("font_name", font)
                if "color" not in new_kwargs:
                    new_kwargs["color"] = color
                if "size" not in new_kwargs:
                    new_kwargs["size"] = size
                if "text" not in new_kwargs:
                    new_kwargs["text"] = text
                if "x" not in new_kwargs:
                    new_kwargs["x"] = x
                if "y" not in new_kwargs:
                    new_kwargs["y"] = y
                if "width" not in new_kwargs:
                    new_kwargs["width"] = width
                if "height" not in new_kwargs:
                    new_kwargs["height"] = height

                return TextRender().render_text(**new_kwargs)

            return (None, None, None, lll,)

        font_name = kwargs.get("font_name", font)

        # Create an empty image with RGBA channels
        image = Image.new('RGBA', (width, height))

        _render_text(font_name, image, size, text, x, y, allow_shrink=True, sf=sf, sw=sw)

        # Convert the image to numpy array
        image_array = np.array(image)

        # Convert the numpy array to PyTorch tensor and add batch dimension
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).float() / 255.0

        # set the alpha to the first channel to make a mask
        image_tensor[..., 3] = image_tensor[..., 0].to("cpu")

        # now set the text to the correct color
        color_arr = torch.zeros(1, 1, 1, 3).to("cpu")
        color_arr[0, 0, 0, 0] = int(color[1:3], 16) / 255.0
        color_arr[0, 0, 0, 1] = int(color[3:5], 16) / 255.0
        color_arr[0, 0, 0, 2] = int(color[5:7], 16) / 255.0
        # use color_arr to set the color
        image_tensor[..., 0:3] *= color_arr

        # make a copy for the RGB image
        image_rgb = image_tensor[..., :3]
        # print(torch.cuda.memory_stats())
        del image
        del image_array
        del color_arr

        process = psutil.Process();
        memory_info = process.memory_info();
        end_memory = memory_info.rss / (1024 * 1024)
        print("Memory used: {} MB".format(end_memory - start_memory))

        image_rgb = image_rgb.to("cpu")
        image_tensor = image_tensor.to("cpu")
        list_of_possible_fonts_str = "\n".join(get_fonts())
        return (image_rgb, image_tensor, list_of_possible_fonts_str,)


@ETK_image_base
class StripAlphaChannel:
    """
    This function strips the alpha channel from the image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        reqd = {"required": {"image": ("IMAGE",)}}
        return reqd

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "strip_alpha_channel"

    def strip_alpha_channel(self, image):
        """
        This function strips the alpha channel from the image.

        >>> sac = StripAlphaChannel()
        >>> result = sac.strip_alpha_channel(image)
        >>> torch_image_show(result[0])
        """
        from copy import deepcopy

        return (deepcopy(image[..., :3]),)


@ETK_image_base
class FuncImageStackToImageStack:
    """ runs exec on giving the user the input image stack and the ability to define the output image stack"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        req = {"required": {
            "image": ("IMAGE",),
            "code": ("STRING", {"multiline": True, "default": "y=x()"})
        }}
        return req

    CATEGORY = "ETK/func"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"

    def func(self, **kwargs):
        func = kwargs.get("func", None)
        code = kwargs.get("code", None)
        my_globals = globals()
        my_locals = locals()

        image_x = kwargs.get("image", None)
        # gs = image_x.mean(dim=3).repeat(1, 1, 1, 3)

        # my_locals["x"] = func
        my_locals["x_image"] = kwargs.get("image", None)
        my_locals["x"] = kwargs.get("image", None)

        exec(code, my_globals, my_locals)

        y_image = my_locals.get("y_image", None)
        y = my_locals.get("y", None)

        out_y = y_image if y_image is not None else y

        return (out_y,)


@ETK_image_base
class ScaleLatentChannelwise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latent": ("LATENT",),
            "w1": ("FLOAT", {"min": -2, "max": 2, "default": 1, "step": 0.05}),
            "w2": ("FLOAT", {"min": -2, "max": 2, "default": 1, "step": 0.05}),
            "w3": ("FLOAT", {"min": -2, "max": 2, "default": 1, "step": 0.05}),
            "w4": ("FLOAT", {"min": -2, "max": 2, "default": 1, "step": 0.05}),
        }}

    RETURN_TYPES = ("LATENT",)

    CATEGORY = "latent"
    FUNCTION = "scale"

    def scale(self, latent, w1, w2, w3, w4):
        # create a new dictionary to store the scaled latent
        scaled_latent = {}

        # get the samples tensor from the latent
        samples = latent["samples"]

        # apply scaling to each channel of the samples tensor using the corresponding weight tensor
        scaled_samples = samples.clone()
        scaled_samples[:, 0, :, :] *= w1
        scaled_samples[:, 1, :, :] *= w2
        scaled_samples[:, 2, :, :] *= w3
        scaled_samples[:, 3, :, :] *= w4

        # assign the scaled samples tensor to the "samples" key of the scaled_latent dictionary
        scaled_latent["samples"] = scaled_samples

        # return the scaled_latent dictionary
        return (scaled_latent,)


@ETK_image_base
class ImageStackAndMatch:
    # Define properties of the node
    """
    Maintain the aspect ratio of multiple images by adding padding or by cropping,
    then stack the images into a single torch tensor.
    Takes up to 4 image stacks as input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "scale_interp": (
                    ["nearest", "bilinear", "bicubic", "area"], {"default": "area"}),
                "pad_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "scale_type": (["SCL_UP", "SCL_DOWN"], {"default": "SCL_UP"}),
                "pad_or_crop": (["PAD", "CROP"], {"default": "PAD"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "ETK/func"
    FUNCTION = "process_images"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def pad_to_max_size(self, image, max_height, max_width, pad_value):
        import torch.nn.functional as F

        _, h, w, c = image.shape
        padding_left = (max_width - w) // 2
        padding_right = max_width - w - padding_left
        padding_top = (max_height - h) // 2
        padding_bottom = max_height - h - padding_top
        padding = (padding_left, padding_right, padding_top, padding_bottom)
        return F.pad(image, padding, value=pad_value)

    def resize_image(self, image, target_height, target_width, scale_interp):
        import torch.nn.functional as F

        _, h, w, _ = image.shape
        scale_factor_height = target_height / h
        scale_factor_width = target_width / w
        if scale_interp == 'area':
            scale_factor_min = min(scale_factor_height, scale_factor_width)
            image = F.interpolate(image, scale_factor=scale_factor_min, mode='area', recompute_scale_factor=True,
                                  align_corners=None)
        else:
            image = F.interpolate(image, size=(target_height, target_width), mode=scale_interp, align_corners=True)
        return image

    def process_image(self, image, max_height, max_width, scale_type, pad_color, scale_interp, pad_or_crop):
        _, h, w, _ = image.shape
        pad_value = self.hex_to_rgb(pad_color)

        if scale_type == 'SCL_UP':
            target_height = max_height if h < max_height else h
            target_width = max_width if w < max_width else w
            image = self.resize_image(image, target_height, target_width, scale_interp)
            if pad_or_crop == 'PAD':
                image = self.pad_to_max_size(image, max_height, max_width, pad_value)
        else:
            raise NotImplementedError('The scale down functionality is not yet implemented.')

        return image

    def process_images(self, image1, image2, image3=None, image4=None, scale_interp='area',
                       pad_color="#000000", scale_type='SCL_UP', pad_or_crop='PAD'):
        images = [img for img in [image1, image2, image3, image4] if img is not None]

        # Determine max height and width
        max_height = max(img.shape[1] for img in images)
        max_width = max(img.shape[2] for img in images)

        # Process and resize images
        for i, img in enumerate(images):
            images[i] = self.process_image(img, max_height, max_width, scale_type, pad_color, scale_interp, pad_or_crop)

        # Stack images into a single tensor
        stacked_images = torch.stack(images, dim=0)
        return stacked_images


# Define a constant for MAX_RESOLUTION if needed
MAX_RESOLUTION = 4096


@ETK_image_base
class TextInImage:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "detect_text_batch"

    CATEGORY = "OCR"

    def detect_text_batch(self, image, scale_factor):
        # calls the detect_text function for each image in the batch
        results = []
        for i in range(image.shape[0]):
            results.append(self.detect_text(image[i].unsqueeze(0), scale_factor)[0])

        results = '\n\n'.join(results)
        return (results,)

    def detect_text(self, image, scale_factor):
        from .config import config_settings
        tesse = config_settings.get("tesseract_location", None)
        if tesse is None:
            raise Exception("Tesseract location not set in config")

        import pytesseract
        import cv2
        import numpy as np
        pytesseract.pytesseract.tesseract_cmd = tesse

        # incomming tensor is (B, H, W, C) so convert it
        image = image.permute(0, 3, 1, 2)

        # Convert the image to a numpy array
        image = image[0].cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)

        # Resize the image
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the image
        detected_text = pytesseract.image_to_string(gray)

        return (detected_text,)


@ETK_image_base
class DetectTextLines:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "threshold": ("FLOAT", {"default": 0.115, "min": 0.01, "max": 1.0, "step": 0.01})}}

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list of IMAGE",)
    FUNCTION = "detect_lines"

    CATEGORY = "OCR"

    def detect_lines(self, image, threshold=0.115):
        # Convert the image to (b, h, w, c) format for consistency
        # image = image.permute(0, 2, 3, 1)

        # Convert the image to grayscale
        grayscale_image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

        # Sum the pixel values along the width to get a 1D tensor
        sum_along_width = torch.sum(grayscale_image, dim=-1)

        # Normalize the sum to get a clearer indication of lines
        normalized_sum = sum_along_width / torch.max(sum_along_width)

        # Find the y-coordinates where the normalized sum is above the threshold
        line_coordinates = torch.where(normalized_sum >= threshold)[1]

        # Additional code for converting lines to bounding boxes
        bounding_boxes = []
        start_y = None
        end_y = None
        sorted_coordinates = torch.sort(line_coordinates).values

        for i, coord in enumerate(sorted_coordinates):
            if start_y is None:
                start_y = coord
            if i < len(sorted_coordinates) - 1:
                if sorted_coordinates[i + 1] - coord > 1:
                    end_y = coord
            else:
                end_y = coord
            if end_y is not None:
                bounding_boxes.append([start_y.item(), end_y.item()])
                start_y = None
                end_y = None
        # at this point the channels are in 2nd dimension
        # so we need to transpose them to the last dimension
        # image = image.permute(0, 3, 1, 2)

        # Create image tensors for each bounding box
        line_images = [image[:, y1:y2 + 1, :, :] for y1, y2 in bounding_boxes]

        return (line_images,)


@ETK_image_base
class TileImage:
    """repeat an image in a 3x3 grid, useful for checking for tiling artifacts"""

    @classmethod
    def INPUT_TYPES(self):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tile_image"

    CATEGORY = "IMAGE"

    def tile_image(self, image):
        import torch

        # image is already B,H,W,C

        # pad the image to make it a multiple of 3
        # height, width = image.shape[1:3]
        # new_height = height + (3 - height % 3)
        # new_width = width + (3 - width % 3)
        # image = F.pad(image, (0, new_width - width, 0, new_height - height))

        # repeat the image in each dimension
        image = torch.cat([image, image, image], dim=1)
        image = torch.cat([image, image, image], dim=2)

        # now the image is 3x3 with each 1/3 being the original image
        # now the image shape is B,3H,3W,5

        return (image,)


@ETK_image_base
class FloodFillNode:
    node_id = "comfyui.flood_fill"
    node_name = "Flood Fill"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Expecting a NumPy array or tensor
                "start_x": ("INT", {"min": 0, "default": 0}),
                "start_y": ("INT", {"min": 0, "default": 0}),
                "target_color": ("STRING", {"default": "#FFFFFF"}),
                "replacement_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def __init__(self):
        self.num_workers = None  # Can be set to the number of processes to create

    @staticmethod
    def hex_to_rgb(hex_color):
        # Convert hex color string to RGB tuple
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def flood_fill(self, image, start_coords, target_color, replacement_color):
        # The flood fill algorithm assumes 2D image input (single sample from the batch)
        image = image.squeeze(0)  # Remove the batch dimension assuming a single sample
        height, width, _ = image.shape

        target_color = torch.tensor(target_color, dtype=torch.float32)
        replacement_color = torch.tensor(replacement_color, dtype=torch.float32)
        target_color = target_color.view(1, 1, 3)
        replacement_color = replacement_color.view(1, 1, 3)

        start_x, start_y = start_coords
        stack = [(start_x, start_y)]

        # Convert to pixel value range [0, 1] if the max value is greater than 1
        if image.max() > 1.0:
            image = image / 255.0

        visited = torch.zeros(height, width, dtype=torch.bool)

        while len(stack) > 0:
            x, y = stack.pop()
            if not (0 <= x < width and 0 <= y < height) or visited[y, x] or not torch.allclose(image[y, x],
                                                                                               target_color, atol=1e-6):
                continue
            visited[y, x] = True
            image[y, x] = replacement_color
            neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            stack.extend(neighbors)

        image = image.unsqueeze(0)  # Add the batch dimension back
        return image

    def execute(self, images, start_x, start_y, target_color_hex, replacement_color_hex):
        target_color = self.hex_to_rgb(target_color_hex)
        replacement_color = self.hex_to_rgb(replacement_color_hex)

        # Normalize colors to [0, 1] range and convert to the tensor
        target_color = [c / 255.0 for c in target_color]
        replacement_color = [c / 255.0 for c in replacement_color]

        # Apply flood fill to each image in the batch
        for i in range(images.shape[0]):
            images[i] = self.flood_fill(images[i], (start_x, start_y), target_color, replacement_color)

        return images


@ETK_image_base
class DecorateHeightMap:
    """
    Decorate a height map with input images at different heights specified by input list
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height_map": ("IMAGE",),  # Expecting a NumPy array or tensor
                "images": ("IMAGE",),  # Expecting a NumPy array or tensor
                "heights": ("LIST", {"default": [0.5]}),
                "feather": ("LIST", {"default": [0.05]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, height_map, images, heights, feather=0.05):
        """
        Args:
            height_map: float values 0-1 1,h,w,3
            images: z,x,y,c
            heights: list of floats 0-1 length of z-1
            feather: single float or a list of floats specifying the gradient width for feathering effect
        Returns:
            torch image 1,h,w,3
        """
        heights = heights.copy()
        images = images[..., 0:3].clone()
        height_map = height_map[..., 0:3].clone()

        # Ensure heights is a list and add boundary values
        heights = [0.0] + heights + [1.0]

        # If feather is a single float, create a list of the same feather value for each transition
        if isinstance(feather, float):
            feather = [feather] * (len(heights) - 1)

        # so the length of th feather list must be at least 1
        # if there is one height then it should be length 1
        # if there are two heights then it should be length 2
        # if there are three heights then it should be length 3
        # etc
        if len(feather) != len(heights) - 2:
            raise ValueError("feather must be a single float or a list of floats of the same length as heights")

        # Check for dimension match between height_map and images
        if height_map.shape[1:3] != images.shape[1:3]:
            raise ValueError("height_map and images must have the same height and width")

        # Convert height map to 2D
        height_map_2d = height_map.squeeze(0)[..., 0]

        # Initialize output with the first texture
        output = images[0].clone()
        for i in range(1, len(heights) - 1):
            current_feather = feather[i - 1]
            # Calculate the mask for the current segment
            mask = torch.sigmoid((height_map_2d - heights[i - 1]) / current_feather) * (
                    1 - torch.sigmoid((height_map_2d - heights[i]) / current_feather))
            mask = mask.unsqueeze(-1).repeat(1, 1, 3)  # Expand mask to match image dimensions

            # Blend the output with the next texture using the mask
            output = output * (1 - mask) + images[i] * mask

        # Normalize the output to ensure it's within the valid range of [0, 1]
        output = torch.clamp(output, 0, 1).unsqueeze(0)
        return (output,)


@ETK_image_base
class RandomImageFromFolder:
    """
    Randomly select an image from a folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "./output"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, folder):
        """
        Args:
            folder: string
        Returns:
            torch image 1,h,w,3
        """
        import random
        # Get a list of all files in the folder
        files = os.listdir(folder)

        # Filter out non-image files
        files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]

        # Select a random file
        file = random.choice(files)

        # Load the image
        image = Image.open(os.path.join(folder, file))

        # Convert to torch tensor
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.unsqueeze(0)

        return (image,)


@ETK_image_base
class ZoomInImage:
    """
    Zoom in on an image by a specified factor and center point
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "zoom_factor": ("FLOAT", {"default": 2.0, "min": .01, "max": 1000.0, "step": 0.1}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "interpolation": (["nearest", "bilinear", "bicubic", "area"], {"default": "bilinear"}),
                "keep_size": (["True", "False"], {"default": "True"}),
                "animate": ("BOOLEAN", {"default": False}),
                "exponential_zoom": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "zoom_in"

    CATEGORY = "ETK/image"

    def zoom_in(self, image, zoom_factor, center_x, center_y, interpolation, keep_size, animate, exponential_zoom):
        batch_size, height, width, channels = image.shape
        frames = []

        for frame in range(batch_size):
            percent_done = frame / batch_size
            if exponential_zoom:
                percent_done = percent_done ** 2

            # Calculate the zoom factor for this image
            frame_zoom_factor = 1 + (zoom_factor * percent_done)

            # Calculate the new dimensions after zooming
            new_height = int(height / frame_zoom_factor)
            new_width = int(width / frame_zoom_factor)

            # Calculate the coordinates of the top-left corner of the zoomed region
            top = int((height - new_height) * center_y)
            left = int((width - new_width) * center_x)

            # Extract the zoomed region from the image
            zoomed_image = image[frame, top:top + new_height, left:left + new_width, :].unsqueeze(0)

            if keep_size == "True":
                # change order of dims for interpolation
                zoomed_image = zoomed_image.permute(0, 3, 1, 2)
                # Resize the zoomed region back to the original dimensions
                zoomed_image = torch.nn.functional.interpolate(zoomed_image, size=(height, width), mode=interpolation,
                                                               align_corners=False)
                # change order of dims back
                zoomed_image = zoomed_image.permute(0, 2, 3, 1)

            frames.append(zoomed_image)

        # Stack all frames into a 4D tensor
        frames = torch.cat(frames, dim=0)

        return (frames,)


@ETK_image_base
class ListUpscaleLatentBy:
    """uses from nodes import LatentUpscaleBy"""

    @classmethod
    def INPUT_TYPES(cls):
        from nodes import LatentUpscaleBy
        tmp = LatentUpscaleBy.INPUT_TYPES()["required"]["upscale_method"]
        return {
            "required": {
                "latents": ("LIST", {"default": None}),
                "upscale_method": tmp,

            },
            "optional": {
                "factors": ("LIST", {"default": None}),
                "factor": ("FLOAT", {"default": 1.5, "min": 0.0, }),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, latents, upscale_method, factors=None, factor=1.5):
        from nodes import LatentUpscaleBy

        if not factors:
            factors = [factor] * len(latents)

        # Check if latents and factors have the same length
        if len(latents) != len(factors):
            raise ValueError("Latents and factors must have the same length")

        # Apply LatentUpscaleBy to each pair of latent and factor
        upscaled_latents = [LatentUpscaleBy().upscale(latent, upscale_method, factor)[0] for latent, factor in
                            zip(latents, factors)]

        return (upscaled_latents,)


@ETK_image_base
class SBSImage:
    """
    Combine two images side by side
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ETK/image"

    def execute(self, image1, image2):
        """
        Args:
            image1: torch image b,h,w,3
            image2: torch image b,h,w,3
        Returns:
            torch image 1,h,w,3
        """

        # Check if the images have the same height
        if image1.shape[1] != image2.shape[1]:
            raise ValueError("Images must have the same height")

        # Check if the images have the same number of channels
        if image1.shape[3] != image2.shape[3]:
            raise ValueError("Images must have the same number of channels")

        # Stack the images horizontally
        output = torch.cat([image1, image2], dim=2)

        return (output,)


@ETK_image_base
class TTBImage:
    """
    Combine two images top to bottom
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ETK/image"

    def execute(self, image1, image2):
        """
        Args:
            image1: torch image b,h,w,3
            image2: torch image b,h,w,3
        Returns:
            torch image 1,h,w,3
        """

        # Check if the images have the same width
        if image1.shape[2] != image2.shape[2]:
            raise ValueError("Images must have the same width")

        # Check if the images have the same number of channels
        if image1.shape[3] != image2.shape[3]:
            raise ValueError("Images must have the same number of channels")

        # Stack the images vertically
        output = torch.cat([image1, image2], dim=1)

        return (output,)


@ETK_image_base
class ImageGrid:
    """
    Combine multiple images into a grid
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1}),
                "cols": ("INT", {"default": 2, "min": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ETK/image"

    def get_img_idx(self, row, col, rows, cols):
        return row * cols + col

    def get_xy_for_idx(self, idx, rows, cols):
        row = idx % cols
        col = idx // cols
        return row, col

    def execute(self, image, rows, cols):
        """
        Args:
            image: image (r*c),h,w,3
            rows: int
            cols: int
        Returns:
            torch image 1,h*r,w*c,3
        """

        # Check if the number of images is equal to the number of rows times the number of columns
        num_images = rows * cols
        if len(image) < num_images:
            # Calculate how many noise images are needed
            num_noise_images = num_images - len(image)
            # Create the required number of noise images
            noise_shape = image[0].shape
            noise_images = torch.randn((num_noise_images,) + noise_shape, dtype=image.dtype, device=image.device)
            # Concatenate the noise images to the original images
            image = torch.cat((image, noise_images), dim=0)

        # Check if each row has the same height for each image
        for row in range(rows):
            row_idxs = [self.get_img_idx(row, col, rows, cols) for col in range(cols)]
            heights = [image[idx].shape[1] for idx in row_idxs]
            if not all(height == heights[0] for height in heights):
                raise ValueError("Images in the same row must have the same height")

        # Check if all columns of images have the same width
        for col in range(cols):
            col_idxs = [self.get_img_idx(row, col, rows, cols) for row in range(rows)]
            widths = [image[idx].shape[2] for idx in col_idxs]
            if not all(width == widths[0] for width in widths):
                raise ValueError("Images in the same column must have the same width")

        col_widths = []
        for column in range(cols):
            idx = self.get_img_idx(0, column, rows, cols)
            col_width = image[idx].shape[1]
            col_widths.append(col_width)

        col_heights = []
        for row in range(rows):
            idx = self.get_img_idx(row, 0, rows, cols)
            col_height = image[idx].shape[0]
            col_heights.append(col_height)

        col_y_pix = [int(sum(col_heights[:i])) for i in range(len(col_heights))]
        col_x_pix = [int(sum(col_widths[:i])) for i in range(len(col_widths))]

        total_height = sum(col_heights)
        total_width = sum(col_widths)

        out_tensor = torch.zeros((1, total_height, total_width, image.shape[3]), dtype=image.dtype,
                                 device=image.device)

        # check if there is not enough rows and cols to display all images
        max_idx = rows * cols
        tot_images = len(image)
        if max_idx < tot_images:
            raise ValueError("not enough grid space to show all images")

        for idx in range(len(image)):
            x, y = self.get_xy_for_idx(idx, rows, cols)
            x_pix = col_x_pix[x]
            y_pix = col_y_pix[y]
            h, w, _ = image[idx].shape
            out_tensor[0, y_pix:y_pix + h, x_pix:x_pix + w, :] = image[idx]

        image.to("cpu")
        del image
        torch.cuda.empty_cache()

        return (out_tensor,)


@ETK_image_base
class QueryImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",)
            }
        }

    FUNCTION = "execute"

    CATEGORY = "image_to_text"
    RETURN_TYPES = ("STRING",)

    def execute(self, prompt, image):
        from .modules.agent_commands import query_image
        result = query_image.execute({"prompt": prompt, "image": image})

        return (result["result"],)


@ETK_image_base
class CLIPSegAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "", "multiline": False}),
                "use_cuda": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "clipseg_model": ("CLIPSEG_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_transform"

    CATEGORY = "image/transformation"

    def x28_model(self, model, processor, num):
        # takes an initialized model and processor and integrates it into a nn.module to run in paralell
        from torch import nn
        class x28_model(nn.Module):
            def __init__(self, model, processor, overlap=64, slice_size=352):
                super(x28_model, self).__init__()
                self.model = model
                self.processor = processor
                self.mask = None

                # Create a blending mask
                self.mask = np.ones((slice_size, slice_size))
                self.mask[:overlap, :] *= np.linspace(0, 1, overlap)[:, None]
                self.mask[-overlap:, :] *= np.linspace(1, 0, overlap)[:, None]
                self.mask[:, :overlap] *= np.linspace(0, 1, overlap)[None, :]
                self.mask[:, -overlap:] *= np.linspace(1, 0, overlap)[None, :]
                self.mask = torch.tensor(self.mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            def forward(self, image, text, boxs):
                import torch
                image = image.permute(0, 3, 1, 2).to(torch.float32) * 255
                out_image = image.clone() * 0

                for i, bc in enumerate(boxs):
                    with torch.no_grad():
                        # img = image[:, :, bc[0]:bc[1], bc[2]:bc[3]]

                        inputs = processor(text=text, images=image[:, :, bc[0]:bc[1], bc[2]:bc[3]], return_tensors="pt")

                        result = self.model(**inputs)
                        t = torch.sigmoid(result[0])
                        mask = (t - t.min()) / t.max()
                        mask = mask.unsqueeze(0)

                        mask = mask.repeat(1, 3, 1, 1)
                        out_image[:, :, bc[0]:bc[1], bc[2]:bc[3]] += (mask * self.mask)

                return out_image

        mdl = x28_model(model, processor, num)
        return mdl

    def apply_transform(self, image, text, use_cuda, clipseg_model):
        import torch
        import torch.nn.functional as F
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        B, H, W, C = image.shape
        aspect = H / W

        if B != 1:
            raise NotImplementedError("Batch size must be 1")

        # Desired slice size and overlap
        slice_size = 352
        overlap = slice_size // 2
        overlap = 32

        # num_slices_w, slices = self.get_slices(image, overlap, slice_size)
        # just support landscape, square or portrait images
        # 4x7=28, 5x5=25, 7x4=28
        if aspect < 1.25:
            num_slices_w = 3 * 2
            num_slices_h = 2 * 2
        elif aspect > 0.75:
            num_slices_w = 2 * 2
            num_slices_h = 3 * 2
        else:
            num_slices_w = 3 * 2
            num_slices_h = 3 * 2

        model, processor = self.get_models(clipseg_model, use_cuda)

        image_global = image.permute(0, 3, 1, 2)
        image_global = F.interpolate(image_global, size=(num_slices_h * slice_size, num_slices_w * slice_size),
                                     mode='bilinear',
                                     align_corners=False)
        image_global = image_global.permute(0, 2, 3, 1)
        image_global = self.get_global_prediction(image_global, model, processor, slice_size, text, use_cuda)

        # _, slices = self.get_slices(image_global, overlap, slice_size)
        slc_boxs = self.get_slice_boxs(image_global, overlap, slice_size)

        modelx28 = self.x28_model(model, processor, num=len(slc_boxs))

        # Apply the transformation to each slice
        transformed_image = modelx28.forward(image_global, text, slc_boxs)

        transformed_image = transformed_image.permute(0, 2, 3, 1)

        y = transformed_image

        total_power = (y + image_global) / 2
        just_black = image_global < 0.01

        p1 = total_power > .5
        p2 = y > .5
        p3 = image_global > .5

        condition = p1 | p2 | p3
        condition = condition & ~just_black
        y = torch.where(condition, 1.0, 0.0)

        return (y,)

    def get_global_prediction(self, image, model, processor, slice_size, text, use_cuda):
        from torch import nn
        import torch.nn.functional as F
        B, H, W, C = image.shape

        image_global = image.permute(0, 3, 1, 2)
        image_global = F.interpolate(image_global, size=(slice_size, slice_size), mode='bilinear', align_corners=False)
        image_global = image_global.permute(0, 2, 3, 1)
        _, image_global = self.CLIPSeg_image(image_global.float(), text, processor, model, use_cuda)
        image_global = image_global.permute(0, 3, 1, 2)
        image_global = F.interpolate(image_global, size=(H, W), mode='bilinear', align_corners=False)
        image_global = image_global.permute(0, 2, 3, 1)
        return image_global

    def get_models(self, clipseg_model, use_cuda):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        # Initialize CLIPSeg model and processor
        if clipseg_model:
            processor = clipseg_model[0]
            model = clipseg_model[1]
        else:
            processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
            model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        # Move model to CUDA if requested
        if use_cuda and torch.cuda.is_available():
            model = model.to('cuda')
        processor.image_processor.do_rescale = True
        processor.image_processor.do_resize = False
        return model, processor

    def get_slices(self, image, overlap, slice_size):
        B, H, W, C = image.shape

        # Calculate the number of slices needed along each dimension
        num_slices_h = (H - overlap) // (slice_size - overlap) + 1
        num_slices_w = (W - overlap) // (slice_size - overlap) + 1
        # Prepare a list to store the slices
        slices = []
        # Generate the slices
        for i in range(num_slices_h):
            for j in range(num_slices_w):
                start_h = i * (slice_size - overlap)
                start_w = j * (slice_size - overlap)

                end_h = min(start_h + slice_size, H)
                end_w = min(start_w + slice_size, W)

                start_h = max(0, end_h - slice_size)
                start_w = max(0, end_w - slice_size)

                slice_ = image[:, start_h:end_h, start_w:end_w, :]
                print(slice_.shape)
                slices.append(slice_)
        return num_slices_w, slices

    def get_slice_boxs(self, image, overlap, slice_size):
        B, H, W, C = image.shape

        # Calculate the number of slices needed along each dimension
        num_slices_h = (H - overlap) // (slice_size - overlap) + 1
        num_slices_w = (W - overlap) // (slice_size - overlap) + 1
        # Prepare a list to store the slices
        slices_boxs = []
        # Generate the slices
        for i in range(num_slices_h):
            for j in range(num_slices_w):
                start_h = i * (slice_size - overlap)
                start_w = j * (slice_size - overlap)

                end_h = min(start_h + slice_size, H)
                end_w = min(start_w + slice_size, W)

                start_h = max(0, end_h - slice_size)
                start_w = max(0, end_w - slice_size)

                bx = (start_h, end_h, start_w, end_w)
                slices_boxs.append(bx)
                print(bx)
        return slices_boxs

    def CLIPSeg_image(self, image, text, processor, model, use_cuda):
        import torch
        import torchvision.transforms.functional as TF
        B, H, W, C = image.shape

        import torchvision
        with torch.no_grad():
            image = image.permute(0, 3, 1, 2).to(torch.float32) * 255

            inputs = processor(text=[text] * B, images=image, padding=True, return_tensors="pt")

            # Move model and image tensors to CUDA if requested
            if use_cuda and torch.cuda.is_available():
                model = model.to('cuda')
                inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            result = model(**inputs)
            t = torch.sigmoid(result[0])
            mask = (t - t.min()) / t.max()
            mask = torchvision.transforms.functional.resize(mask, (H, W))
            mask = mask.unsqueeze(-1)
            mask_img = mask.repeat(1, 1, 1, 3)

            # Move mask and mask_img back to CPU if they were moved to CUDA
            if use_cuda and torch.cuda.is_available():
                mask = mask.cpu()
                mask_img = mask_img.cpu()

        return (mask, mask_img,)


from typing import Tuple


@ETK_image_base
class PreviewImagePassThrough:  # (PreviewImage):
    @classmethod
    def INPUT_TYPES(s):
        from nodes import PreviewImage
        # get the input types from the parent class
        input_types = PreviewImage.INPUT_TYPES()

        return input_types

    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pass_through"

    def pass_through(self, **kwargs):
        from nodes import PreviewImage
        images = kwargs.pop("images", None)
        # first call the parent class's save_image method
        result = PreviewImage().save_images(images, **kwargs)
        result["result"] = (images,)
        return result


def main():
    t = CLIPSegAdv()
    t.apply_transform(torch.rand(1, 1920, 1080, 3), "a", False, None)


if __name__ == "__main__":
    main()
