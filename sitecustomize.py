import sys
import types
from pathlib import Path
import os
import importlib.util

# During unit tests we provide lightweight stubs for ComfyUI and the main package
if os.environ.get('UNIT_TEST'):
    # Stub comfy package
    import tests.mock_comfy as mock_comfy
    sys.modules.setdefault('comfy', mock_comfy)

    repo_root = Path(__file__).resolve().parent
    pkg = types.ModuleType('EternalKernelLiteGraphNodes')
    pkg.__path__ = [str(repo_root)]
    sys.modules.setdefault('EternalKernelLiteGraphNodes', pkg)
    custom_nodes = types.ModuleType('custom_nodes')
    custom_nodes.EternalKernelLiteGraphNodes = pkg
    sys.modules.setdefault('custom_nodes', custom_nodes)
    sys.modules.setdefault('custom_nodes.EternalKernelLiteGraphNodes', pkg)

    # Provide a very small torchvision stub if torchvision is missing
    if importlib.util.find_spec('torchvision') is None:
        tv_stub = types.ModuleType('torchvision')
        tv_stub.datasets = types.ModuleType('datasets')
        tv_stub.models = types.ModuleType('models')
        sys.modules['torchvision'] = tv_stub
