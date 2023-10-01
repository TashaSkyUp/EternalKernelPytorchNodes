import os
import comfy

try:
    from line_profiler_pycharm import profile
except ImportError as e:
    print("ETK> line_profiler_pycharm not found, skipping it")
    profile = lambda x: x

try:
    from utils import etk_deep_copy as deepcopy
except ImportError as e:
    from custom_nodes.EternalKernelLiteGraphNodes.utils import etk_deep_copy as deepcopy

testing = os.environ.get("ETERNAL_KERNEL_LITEGRAPH_NODES_TEST", None)
if testing == "True":
    testing = True
elif __name__ == "__main__":
    testing = True
else:
    testing = False

if testing:
    class SaveImage:
        def __init__(self):
            pass

        def __call__(self, *args, **kwargs):
            pass
else:
    try:
        from nodes import VAEDecode, KSampler, CheckpointLoaderSimple, EmptyLatentImage
        from nodes import CLIPTextEncode, VAEEncode, SaveImage

        from nodes import common_ksampler

        import comfy.samplers
    except ImportError as e:
        print("ETK> comfy.samplers not found, skipping comfyui")


        class SaveImage:
            def __init__(self):
                pass

            def __call__(self, *args, **kwargs):
                pass

try:
    from custom_nodes.ComfyUI_ADV_CLIP_emb.nodes import AdvancedCLIPTextEncode as CLIPTextEncodeAdvanced
except ImportError as e:
    print("ETK> advanced clip not found, skipping it")

import folder_paths
import torch
import torchvision
import PIL.Image as Image
import os
import numpy as np
import hashlib

### info for code completion AI ###
"""
all of these classes are plugins for comfyui and follow the same pattern
all of the images are torch tensors and it is unknown and unimportant if they are on the cpu or gpu
all image inputs are (B,W,H,C)

avoid numpy and PIL as much as possible
"""
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def ETK_image_base(cls):
    # cls.FUNCTION = "func"
    cls.CATEGORY = "ETK/Image"
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
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
class TinyTxtToImg:
    """small text to image generator"""
    share_clip = None
    share_mdl = None
    share_vae = None

    def __init__(self):
        print("ETK> TinyTxtToImg init")
        import random
        self.mdl = TinyTxtToImg.share_mdl
        self.clp = TinyTxtToImg.share_clip
        self.vae = TinyTxtToImg.share_vae
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
                                         "default": ""}),
                "FUNC": ("FUNC",)
            }
        }

    CATEGORY = "ETK"

    RETURN_TYPES = ("IMAGE", "FUNC", "LATENT", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("image", "FUNC", "latent", "positive", "negative",)
    FUNCTION = "tinytxt2img"

    @profile
    def tinytxt2img(self, **kwargs):
        """ use the imports from nodes to generate an image from text """

        import random
        import json
        import gc

        # Create a stream for computation
        # gpu_stream = torch.cuda.Stream(device="cuda")
        # cpu_stream = torch.Stream()

        # kwargs = deepcopy(kwargs)

        unload_model = kwargs.get("unload_model", True)
        prompt = kwargs.get("prompt", "")
        neg_prompt = kwargs.get("neg_prompt", "")
        clip_encoder = kwargs.get("clip_encoder", "comfy -ignore below")
        token_normalization = kwargs.get("token_normalization", "length+mean")
        weight_interpretation = kwargs.get("weight_interpretation", "comfy++")
        ckpt_name = kwargs.get("ckpt_name", "v1-5-pruned-emaonly.safetensors")
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

        if not hasattr(TinyTxtToImg, "share_mdl"):
            TinyTxtToImg.share_mdl = None
            TinyTxtToImg.share_mdl_ckpt_name = None

        if TinyTxtToImg.share_mdl is None:
            self.mdl, self.clp, self.vae, self.vision = \
                CheckpointLoaderSimple.load_checkpoint(None,
                                                       ckpt_name=ckpt_name
                                                       )
            TinyTxtToImg.share_mdl_ckpt_name = ckpt_name
            CSL = TinyTxtToImg
            CSL.share_mdl = self.mdl
            CSL.share_clip = self.clp
            CSL.share_vae = self.vae
            CSL.share_vision = self.vision

        else:
            # check if the checkpoint is the same
            if TinyTxtToImg.share_mdl_ckpt_name != ckpt_name:
                self.mdl, self.clp, self.vae, self.vision = \
                    CheckpointLoaderSimple.load_checkpoint(None,
                                                           ckpt_name=ckpt_name
                                                           )
                TinyTxtToImg.share_mdl_ckpt_name = ckpt_name
                CSL = TinyTxtToImg
                CSL.share_mdl = self.mdl
                CSL.share_clip = self.clp
                CSL.share_vae = self.vae
                CSL.share_vision = self.vision
            else:

                self.mdl, self.clp, self.vae = (TinyTxtToImg.share_mdl,
                                                TinyTxtToImg.share_clip,
                                                TinyTxtToImg.share_vae
                                                )

        self.cfg = 8
        self.sampler_name = comfy.samplers.KSampler.SAMPLERS[0]
        self.scheduler = comfy.samplers.KSampler.SCHEDULERS[0]
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
        ## TODO: probably need to prepare the execution better to make sure that applying the incomming func can change everything
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
            import comfy.model_management as mm

            torch.cuda.empty_cache()
            gc.collect()
        if self.latent_image == None:
            self.latent_image = EmptyLatentImage.generate(None, self.width, self.height, self.batch_size)[0]

        if "latent" in render_what or "all" in render_what or "image" in render_what:
            self.KSampler = KSampler()
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
        # ret = deepcopy(ret)
        return ret


@ETK_image_base
class PreviewImageTest(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True}),
                "neg_prompt": ("STRING", {"multiline": True}),
                "clip_encoder": (["comfy -ignore below", "advanced"],),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++"],)
            },
        }

    def myfunc(self, images, prompt=None, extra_pnginfo=None, ret=None):
        return ret

    CATEGORY = "ETK/image"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, **kwargs):
        ret = super().save_images(images, filename_prefix, prompt, extra_pnginfo)
        my_ret = self.myfunc(images, prompt, extra_pnginfo, ret)
        return my_ret


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
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "g": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "b": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "similarity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
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
        import torch
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
        inps = KSampler.INPUT_TYPES()
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
        sampler = KSampler()

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
class KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "one_seed_per_batch": ([True, False], {"default": False}),
                     }}

    RETURN_TYPES = ("LATENT", "FUNC",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, **kwargs):

        kwargs = deepcopy(kwargs)

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

        self.FUNC = lambda x: KSampler.sample(self, **x)
        self.ARGS = kwargs

        try:
            ret = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, one_seed_per_batch=one_seed_per_batch)
        except TypeError as e:
            ret = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise)
        ret = (ret[0], self,)
        return ret


from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np


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
            }
        }

    CATEGORY = "ETK/text"

    RETURN_TYPES = ("IMAGE", "IMAGE", "FUNC",)
    RETURN_NAMES = ("image", "image rgba", "FUNC(**kwargs)",)

    FUNCTION = "render_text"

    @profile
    def render_text(self, text, x, y, width, height, font='Arial', size=16, color='#888888', func_only=False, **kwargs):
        """
        This function renders the provided text at specified location, with the given width and height.
        The text is rendered in the provided font, size, and color.
        """
        from PIL import ImageFont, ImageDraw
        import gc
        import psutil
        process = psutil.Process();
        memory_info = process.memory_info();
        start_memory = memory_info.rss / (1024 * 1024)

        sw = kwargs.get("stroke width", None)
        sf = kwargs.get("stroke fill", None)

        @profile
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

        @profile
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
                                                        allow_shrink=False)
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
                                                    allow_shrink=False)
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

                return TextRender.render_text(None, **new_kwargs)

            return (None, None, lll,)

        font_name = kwargs.get("font_name", font)

        # Create an empty image with RGBA channels
        image = Image.new('RGBA', (width, height))

        _render_text(font_name, image, size, text, x, y, allow_shrink=True, sf=sf, sw=sw)

        # Convert the image to numpy array
        image_array = np.array(image)

        # Convert the numpy array to PyTorch tensor and add batch dimension
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).float() / 255.0

        # set the alpha to the first channel to make a mask
        image_tensor[..., 3] = image_tensor[..., 0]

        # now set the text to the correct color
        color_arr = torch.zeros(1, 1, 1, 3)
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
        return (image_rgb, image_tensor, None,)


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
        gs = image_x.mean(dim=3).repeat(1, 1, 1, 3)

        my_locals["x"] = func
        my_locals["x_image"] = kwargs.get("image", None)

        exec(code, my_globals, my_locals)

        return (my_locals["y_image"],)


@ETK_image_base
class ScaleLatentChannelwise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"latent": ("LATENT",),
                     "w1": ("FLOAT", {"min": -2, "max": 2, "default": 1,"step":0.05}),
                     "w2": ("FLOAT", {"min": -2, "max": 2, "default": 1,"step":0.05}),
                     "w3": ("FLOAT", {"min": -2, "max": 2, "default": 1,"step":0.05}),
                     "w4": ("FLOAT", {"min": -2, "max": 2, "default": 1,"step":0.05}),
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



# Define a constant for MAX_RESOLUTION if needed
MAX_RESOLUTION = 4096



# Define the ComfyUI node class for text detection using CRNN model from Torch Hub
@ETK_image_base
class TextInImage:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "detect_text"

    CATEGORY = "OCR"

    def detect_text(self, image, scale_factor):
        import torch
        from torchvision import transforms
        # the incoming tensor in image is in the format (batch, height, width, channels)
        # we need to convert it to (batch, channels, height, width)

        image = image.permute(0, 3, 1, 2)
        new_size=list(image.shape)
        new_size[2]=int(new_size[2]*scale_factor)
        new_size[3]=int(new_size[3]*scale_factor)
        new_size=tuple(new_size[2:])

        rescaled_image = torch.nn.functional.interpolate(image, size=new_size, mode='bilinear')
        # convert to PIL image

        pil_image = transforms.ToPILImage()(rescaled_image.squeeze(0))
        # Load CRNN model from Torch Hub
        # Use a pipeline as a high-level helper

        from transformers import pipeline

        pipe = pipeline("image-to-text", model="microsoft/trocr-large-printed")

        detected_text = pipe.predict(pil_image)[0]["generated_text"]

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
        #image = image.permute(0, 2, 3, 1)

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
        line_images = [image[:, y1:y2+1, :, :] for y1, y2 in bounding_boxes]

        return (line_images,)


if __name__ == "__main__":
    # test TextRender
    tr = TextRender()
    result = tr.render_text('Hello, world!', 128, 128, 512, 512, "Arial", 16, "#FFFF000")[0]
    torch_image_show(result)
    print(result.shape)


def main():
    pass


if __name__ == "__main__":
    main()
