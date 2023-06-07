NODE_CLASS_MAPPINGS = {}
try:
    import comfy.samplers
    from custom_nodes.EternalKernelLiteGraphNodes.image import PreviewImageTest
    from custom_nodes.EternalKernelLiteGraphNodes.image import TinyTxtToImg
    from custom_nodes.EternalKernelLiteGraphNodes.image import KSampler

    use_generic_nodes = False
except:
    use_generic_nodes = True
    print("ETK> Failed to import comfy.samplers, assuming no comfyui skipping SD related nodes")

from custom_nodes.EternalKernelLiteGraphNodes.latent import LatentInterpolation
from custom_nodes.EternalKernelLiteGraphNodes.image import PadToMatch
from custom_nodes.EternalKernelLiteGraphNodes.image import StackImages
from custom_nodes.EternalKernelLiteGraphNodes.image import ImageBC
from custom_nodes.EternalKernelLiteGraphNodes.image import RGBA_MOD
from custom_nodes.EternalKernelLiteGraphNodes.image import rgba_lower_clip
from custom_nodes.EternalKernelLiteGraphNodes.image import rgba_upper_clip

from custom_nodes.EternalKernelLiteGraphNodes.image import rgba_merge
from custom_nodes.EternalKernelLiteGraphNodes.image import rgba_split
from custom_nodes.EternalKernelLiteGraphNodes.image import select_from_batch
from custom_nodes.EternalKernelLiteGraphNodes.image import Quantize
from custom_nodes.EternalKernelLiteGraphNodes.image import SelectFromRGBSimilarity
from custom_nodes.EternalKernelLiteGraphNodes.image import LoadImage
from custom_nodes.EternalKernelLiteGraphNodes.image import PromptTemplate
from custom_nodes.EternalKernelLiteGraphNodes.image import CodeExecWidget
from custom_nodes.EternalKernelLiteGraphNodes.image import ImageDistanceMask
from custom_nodes.EternalKernelLiteGraphNodes.image import TextRender

from custom_nodes.EternalKernelLiteGraphNodes.functional import FuncBase
from custom_nodes.EternalKernelLiteGraphNodes.functional import FuncRender
from custom_nodes.EternalKernelLiteGraphNodes.functional import FuncRenderImage

if use_generic_nodes:
    NODE_CLASS_MAPPINGS_GENERIC = {
        "Interpolation": LatentInterpolation,
        "Image Pad To Match": PadToMatch,
        "Image Stack": StackImages,
        "Image B/C": ImageBC,
        "Image RGBA Mod": RGBA_MOD,
        "Image RGBA Lower Clip": rgba_lower_clip,
        "Image RGBA Upper Clip": rgba_upper_clip,
        "Image RGBA Merge": rgba_merge,
        "Image RGBA Split": rgba_split,
        "Select From Batch": select_from_batch,
        "Quantize": Quantize,
        "Select From RGB Similarity": SelectFromRGBSimilarity,
        "Load Layered Image": LoadImage,
        "Code Execution Widget": CodeExecWidget,
        "Image Distance Mask": ImageDistanceMask,

        "Image Text": TextRender,

        "function": FuncBase,
        "function render": FuncRender,
        "function render image": FuncRenderImage,

    }

    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_GENERIC)
else:
    NODE_CLASS_MAPPINGS_COMFY = {
        "Interpolation": LatentInterpolation,
        "Image Pad To Match": PadToMatch,
        "Image Stack": StackImages,
        "Image B/C": ImageBC,
        "Image RGBA Mod": RGBA_MOD,
        "Image RGBA Lower Clip": rgba_lower_clip,
        "Image RGBA Upper Clip": rgba_upper_clip,
        "Image RGBA Merge": rgba_merge,
        "Image RGBA Split": rgba_split,
        "Select From Batch": select_from_batch,
        "Quantize": Quantize,
        "Select From RGB Similarity": SelectFromRGBSimilarity,
        "Load Layered Image": LoadImage,
        "Prompt Template": PromptTemplate,
        "Code Execution Widget": CodeExecWidget,
        "Tiny Txt 2 Img": TinyTxtToImg,
        "Image Distance Mask": ImageDistanceMask,
        "Preview Image Test": PreviewImageTest,
        "ETK_KSampler": KSampler,

        "Image Text": TextRender,

        "function": FuncBase,
        "function render": FuncRender,
        "function render image": FuncRenderImage,

    }
    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_COMFY)

from custom_nodes.EternalKernelLiteGraphNodes.video import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_VIDEO

NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_VIDEO)
