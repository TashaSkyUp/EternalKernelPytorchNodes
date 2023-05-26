NODE_CLASS_MAPPINGS = {}
try:
    import comfy.samplers
    from custom_nodes.EternalKernelLiteGraphNodes.image import PreviewImageTest
    from custom_nodes.EternalKernelLiteGraphNodes.image import TinyTxtToImg
    use_generic_nodes = False
except:
    use_generic_nodes = True
    print("ETK> Failed to import comfy.samplers, assuming no comfyui skipping SD related nodes")

from custom_nodes.EternalKernelLiteGraphNodes.latent import LatentInterpolation
from custom_nodes.EternalKernelLiteGraphNodes.image  import PadToMatch
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
from custom_nodes.EternalKernelLiteGraphNodes.image import ExecWidget

from custom_nodes.EternalKernelLiteGraphNodes.image import ImageDistanceMask

from custom_nodes.SWAIN.nlp import NODE_CLASS_MAPPINGS as NLP_NODE_CLASS_MAPPINGS




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
        "Eval Widget": ExecWidget,
        "Image Distance Mask": ImageDistanceMask,
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
        "Eval Widget": ExecWidget,
        "Tiny Txt 2 Img": TinyTxtToImg,
        "Image Distance Mask": ImageDistanceMask,
        "Preview Image Test": PreviewImageTest,
    }
    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_COMFY)

for k, v in NLP_NODE_CLASS_MAPPINGS.items():
    NODE_CLASS_MAPPINGS[k] = v
