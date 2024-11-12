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

from . import AVF

# Update NODE_CLASS_MAPPINGS with AVF nodes
NODE_CLASS_MAPPINGS.update(AVF.NODE_CLASS_MAPPINGS)

# Update NODE_DISPLAY_NAME_MAPPINGS with AVF nodes
NODE_DISPLAY_NAME_MAPPINGS.update(AVF.NODE_DISPLAY_NAME_MAPPINGS)


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
#from . import audio_video_folder


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
