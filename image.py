try:
    import comfy.samplers
    # import custom_nodes.Derfuu_ComfyUI_ModdedNodes.components.fields as field
    from nodes import CLIPTextEncode, VAEEncode, VAEDecode, KSampler, CheckpointLoaderSimple, EmptyLatentImage, \
        SaveImage
    from custom_nodes.ComfyUI_ADV_CLIP_emb.nodes import AdvancedCLIPTextEncode as CLIPTextEncodeAdvanced
    from nodes import common_ksampler
except ImportError as e:
    print("ETK> comfy.samplers not found, skipping comfyui")
    SaveImage = None

import torch
import torchvision
import PIL.Image as Image

try:
    import folder_paths
except:
    pass

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
    image = torch.mul(image[0], 255, )
    image = image.to(torch.uint8)  # Convert tensor to uint8 data type

    # must have at least 3 channels
    if image.shape[-1] == 1:
        image = torch.cat((image, image, image), dim=-1)

    image = image.numpy()
    image = Image.fromarray(image)  # Create a PIL image
    image.show()


class TinyTxtToImg:
    """small text to image generator"""

    def __init__(self):
        import random
        self.mdl = None
        self.clp = None
        self.vae = None
        self.vision = None
        self.seed = random.randint(0, 2 ** 32 - 1)
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

    RETURN_TYPES = ("IMAGE","FUNC",)
    FUNCTION = "tinytxt2img"

    def tinytxt2img(self, prompt, neg_prompt, name="tinytxt2img", overrides: str = "",
                    clip_encoder="comfy -ignore below",
                    token_normalization="length+mean",
                    weight_interpretation="comfy++",
                    FUNC: callable = None):
        """ use the imports from nodes to generate an image from text """

        import random
        import json
        if CheckpointLoaderSimple.share_mdl is None:
            self.mdl, self.clp, self.vae, self.vision = CheckpointLoaderSimple.load_checkpoint(None,
                                                                                               ckpt_name="v1-5-pruned-emaonly.safetensors"
                                                                                               )
            CSL = CheckpointLoaderSimple
            CSL.share_mdl = self.mdl
            CSL.share_clip = self.clp
            CSL.share_vae = self.vae
            CSL.share_vision = self.vision

        else:
            self.mdl, self.clp, self.vae = (CheckpointLoaderSimple.share_mdl,
                                            CheckpointLoaderSimple.share_clip,
                                            CheckpointLoaderSimple.share_vae
                                            )
        self.seed = random.randint(0, 2 ** 32 - 1)
        self.steps = 10
        self.cfg = 8
        self.sampler_name = comfy.samplers.KSampler.SAMPLERS[0]
        self.scheduler = comfy.samplers.KSampler.SCHEDULERS[0]
        self.positive = prompt
        self.negative = neg_prompt
        self.width = 512
        self.height = 512
        self.batch_size = 1
        self.denoise = 1.0
        self.one_seed_per_batch = False

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
        if clip_encoder == "comfy -ignore below":
            self.pos_cond = CLIPTextEncode.encode(None, self.clp, self.positive)[0]
            self.neg_cond = CLIPTextEncode.encode(None, self.clp, self.negative)[0]
        elif clip_encoder == "advanced":
            self.pos_cond = CLIPTextEncodeAdvanced.encode(None,
                                                          self.clp,
                                                          self.positive,
                                                          token_normalization,
                                                          weight_interpretation)[0]
            self.neg_cond = CLIPTextEncodeAdvanced.encode(None,
                                                          self.clp,
                                                          self.negative,
                                                          token_normalization,
                                                          weight_interpretation)[0]

        self.latent_image = EmptyLatentImage.generate(None, self.width, self.height, self.batch_size)[0]

        samples = KSampler.sample(None,
                                  self.mdl,
                                  self.seed, self.steps, self.cfg, self.sampler_name, self.scheduler,
                                  self.pos_cond, self.neg_cond,
                                  self.latent_image,
                                  denoise=self.denoise,
                                  one_seed_per_batch=self.one_seed_per_batch)[0]

        image = VAEDecode.decode(None, self.vae, samples)[0]
        image = image.detach().cpu()

        return (image,lambda : self.tinytxt2img(prompt,
                                                neg_prompt,
                                                name,
                                                overrides,
                                                clip_encoder,
                                                token_normalization,
                                                weight_interpretation,
                                                FUNC
                                                )
                ,)


if SaveImage != None:
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


class ExecWidget:
    """runs eval on the given text"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional":
            {
                "text_to_eval": ("STRING",
                                 {"multiline": True,
                                  "default":"int_out=int_out\n"
                                            "float_out=float_out\n"
                                            "string_out=string_out\n"
                                            "image_out=image_out\n"
                                  }),
                "image1_in": ("IMAGE", {"multiline": False}),
                "float1_in": ("FLOAT", {"multiline": False}),
                "string1_in": ("STRING", {"multiline": False}),
                "int1_in": ("INT", {"multiline": False}),

            },
            "required": {
                "name": ("STRING", {"multiline": False,
                                    "default": "exec_func"}
                         ),
            }
        }

    CATEGORY = "text"
    RETURN_TYPES = ("STRING", "IMAGE", "FLOAT", "INT")
    FUNCTION = "exec_handler"

    def exec_handler(self, text_to_eval, image1_in: torch.Tensor = None, float1_in: float = 0.0, string1_in: str = "",
                     int1_in: int = 0,name: str = "exec_func"):
        """
        >>> ExecWidget().exec_handler("2 + 3")
        '5'
        """
        result = None
        out_image = None
        out_float = None
        out_string = None

        if image1_in is not None:
            image_in = image1_in.clone()
        else:
            image_in = None

        new_locals = {"float_out": float1_in,
                      "image_out": image_in,
                      "string_out": string1_in,
                      "int_out": int1_in,
                      }
        try:
            if isinstance(text_to_eval, list):
                text_to_eval = text_to_eval[0]

            if "<code>" in text_to_eval and "</code>" in text_to_eval:
                start = text_to_eval.find("<code>")
                end = text_to_eval.find("</code>")
                text_to_eval = text_to_eval[start + 6:end]

            exec(text_to_eval, globals(), new_locals)
            out_string = new_locals.get("string_out", None)
            out_image = new_locals.get("image_out", None)
            out_float = new_locals.get("float_out", None)
            out_int = new_locals.get("int_out", None)


        except Exception as e:
            print(e)
            out_string = str(e)

        return (out_string, out_image, out_float, out_int)

    @classmethod
    def IS_CHANGED(s, text_to_eval, image1_in: torch.Tensor = None, float1_in: float = 0.0, string1_in: str = "",
                   int1_in: int = 0, name: str = "exec_func"):

        m = hashlib.sha256()
        m.update(name.encode("utf-8"))
        return m.digest().hex()


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
            }, }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stackme"

    CATEGORY = "ETK/image"

    def stackme(self, image1, image2=None, number_of_images=2):
        """
        stacks two images on top of each other
        images are (B, W, H, C) torch tensors
        """
        image1 = image1.clone()

        if image2 is not None:
            image2 = image2.clone()
            stacked = torch.cat((image1, image2), dim=0)
        else:
            stacked = torch.cat((image1, image1), dim=0)

        if number_of_images > 2:
            while stacked.shape[0] < number_of_images:
                stacked = torch.cat((stacked, stacked), dim=0)
            print(stacked.shape)
        out = stacked[:number_of_images, :, :, :]
        print(out.shape)
        return (out,)


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

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
               one_seed_per_batch=False):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                               denoise=denoise, one_seed_per_batch=one_seed_per_batch)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
