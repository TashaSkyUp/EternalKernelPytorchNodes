NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

# In /custom_nodes/ETK/__init__.py

import sys
from types import ModuleType


def patch_validate_inputs():
    # Get the execution module
    execution_module = sys.modules.get('execution')
    if not execution_module:
        print("Warning: 'execution' module not found. Patch not applied.")
        return

    # Store the original validate_inputs function
    original_validate_inputs = execution_module.validate_inputs

    def wrapper_validate_inputs(prompt, item, validated):
        original_result = original_validate_inputs(prompt, item, validated)

        if not original_result[0]:  # If validation failed
            errors = original_result[1]
            filtered_errors = [
                error for error in errors
                if error['type'] != 'return_type_mismatch'
            ]

            # If the only errors were type mismatches, consider it valid
            if not filtered_errors:
                return (True, [], original_result[2])
            else:
                return (False, filtered_errors, original_result[2])

        return original_result

    # Replace the original function with our wrapper
    execution_module.validate_inputs = wrapper_validate_inputs
    print("validate_inputs function has been patched by ETK extension.")


# Apply the patch when this module is imported
patch_validate_inputs()


from . import basic
from . import image
from . import video
from . import functional
from . import audio
from . import git
from . import server_endpoints
from . import torchvision_nodes
from . import utils
from . import pytorch_nodes
from . import youtube_nodes
from . import torchvision_nodes
from . import hf_diffusers_nodes



__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# import os
# import importlib
#
# WEB_DIRECTORY = "./web"
# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
# OLD_NODE_CLASS_MAPPINGS = {
#     "Interpolation": "LatentInterpolation",
#     "Image Pad To Match": "PadToMatch",
#     "Image Stack": "StackImages",
#     "Image B/C": "ImageBC",
#     "Image RGBA Mod": "RGBA_MOD",
#     "Image RGBA Lower Clip": "rgba_lower_clip",
#     "Image RGBA Upper Clip": "rgba_upper_clip",
#     "Image RGBA Merge": "rgba_merge",
#     "Image RGBA Split": "rgba_split",
#     "Select From Batch": "select_from_batch",
#     "Quantize": "Quantize",
#     "Select From RGB Similarity": "SelectFromRGBSimilarity",
#     "Load Layered Image": "LoadImage",
#     "Code Execution Widget": "ExecWidget",
#     "Image Distance Mask": "ImageDistanceMask",
#     "Image Text": "TextRender",
#     "function": "FuncBase",
#     "function render": "FuncRender",
#     "function render image": "FuncRenderImage",
#     "Tiny Txt 2 Img": "TinyTxtToImg",
#     "Preview Image Test": "PreviewImageTest",
#     "ETK_KSampler": "KSampler",
# }
# # reverse the dict
# OLD_NODE_CLASS_MAPPINGS = {v: k for k, v in OLD_NODE_CLASS_MAPPINGS.items()}
#
# NODE_CLASS_MAPPINGS = {}
# NODE_DISPLAY_NAME_MAPPINGS = {}
# # Try importing optional dependencies
#
# import sys
#
# if os.getenv("UNIT_TEST", False) or 'pytest' in sys.modules:
#     pass
# else:  # if not a unit test
#     try:
#         import comfy.samplers
#
#         use_generic_nodes = False
#     except ImportError:
#         use_generic_nodes = True
#         print("ETK> Failed to import comfy.samplers, assuming no comfyui skipping SD related nodes")
#
#     # Load classes from modules in the custom_nodes.EternalKernelLiteGraphNodes package
#     folder = os.path.dirname(os.path.abspath(__file__))
#
#     for file in os.listdir(folder):
#         if file == "nodes.py":
#             raise ValueError("nodes.py is not an allowed name in this folder")
#         if file.endswith(".py") and file[0] != "_":
#             # import the NODE_CLASS_MAPPINGS from the file
#             try:
#                 module = importlib.import_module(f".{file[:-3]}", __name__)
#             except ImportError:
#                 try:
#                     module = importlib.import_module(f"{file[:-3]}")  # needed for testing?
#                 except ModuleNotFoundError:
#                     module = importlib.import_module(f".{file[:-3]}", __name__)
#
#             if hasattr(module, "NODE_CLASS_MAPPINGS"):
#                 mappings = module.NODE_CLASS_MAPPINGS
#                 NODE_CLASS_MAPPINGS.update(mappings)
#                 for k, v in mappings.items():
#                     print(f"ETK: updated with {k}")
#
#             if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
#                 NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
#
#     # use OLD_NODE_CLASS_MAPPINGS to add the old names to the new nodes
#     new_node_class_mappings = {}
#     for k, v in OLD_NODE_CLASS_MAPPINGS.items():
#         if k in NODE_CLASS_MAPPINGS:
#             new_node_class_mappings[v] = NODE_CLASS_MAPPINGS[k]
#             # mark it for removeal
#             NODE_CLASS_MAPPINGS[k] = None
#         else:
#             print(f"ETK> Failed to find node {k} in new nodes, skipping...")
#     # remove the nodes with None
#     NODE_CLASS_MAPPINGS = {k: v for k, v in NODE_CLASS_MAPPINGS.items() if v is not None}
#     NODE_CLASS_MAPPINGS.update(new_node_class_mappings)
