import sys
import types

# Minimal mock of the ComfyUI package used in tests.
model_patcher = types.ModuleType('model_patcher')
class ModelPatcher:
    def __init__(self, *args, **kwargs):
        pass
model_patcher.ModelPatcher = ModelPatcher

# Register the submodule under this package name and the comfy namespace
sys.modules[__name__ + '.model_patcher'] = model_patcher
sys.modules['comfy.model_patcher'] = model_patcher
