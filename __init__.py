"""
@title: EternalKernel PyTorch Nodes
@nickname: EternalKernel
@description: Comprehensive PyTorch nodes for ComfyUI - Neural network training, inference, and ML workflows with 35+ specialized nodes
"""

# Original content follows

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

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

# Import PyTorch nodes to populate node mappings
from . import pytorch_nodes


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Expose package under custom_nodes namespace for tests
if 'custom_nodes' not in sys.modules:
    sys.modules['custom_nodes'] = ModuleType('custom_nodes')
sys.modules['custom_nodes'].EternalKernelLiteGraphNodes = sys.modules[__name__]
sys.modules['custom_nodes.EternalKernelLiteGraphNodes'] = sys.modules[__name__]
