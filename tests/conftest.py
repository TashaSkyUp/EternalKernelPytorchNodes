import os
os.environ.setdefault('UNIT_TEST', 'True')

import sys
import types
import importlib.util

# Provide a minimal torchvision stub only when torchvision is not installed
if importlib.util.find_spec('torchvision') is None:
    tv_stub = types.ModuleType('torchvision')
    tv_stub.datasets = types.ModuleType('datasets')
    tv_stub.models = types.ModuleType('models')
    tv_stub.transforms = types.ModuleType('transforms')
    tv_stub.transforms.ToTensor = lambda: lambda x: x
    sys.modules['torchvision'] = tv_stub

if importlib.util.find_spec('PIL') is None:
    pil_stub = types.ModuleType('PIL')
    pil_stub.Image = types.SimpleNamespace(open=lambda *a, **k: None)
    sys.modules['PIL'] = pil_stub
