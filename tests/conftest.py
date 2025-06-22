import os
import sys
from pathlib import Path

os.environ.setdefault('UNIT_TEST', 'True')

# Ensure optional test stubs are loaded
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
try:
    import sitecustomize  # noqa: F401
except Exception:
    pass
finally:
    if sys.path[0] == str(repo_root):
        sys.path.pop(0)
        sys.path.append(str(repo_root))

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

# Provide folder_paths stub if module unavailable
if 'folder_paths' not in sys.modules:
    fp_stub = types.SimpleNamespace(
        get_filename_list=lambda *a, **k: [],
        get_input_directory=lambda: '',
        get_output_directory=lambda: '',
        get_temp_directory=lambda: ''
    )
    sys.modules['folder_paths'] = fp_stub

# Minimal torchaudio stub to satisfy imports
if importlib.util.find_spec('torchaudio') is None:
    ta_stub = types.ModuleType('torchaudio')
    def _load(*a, **k):
        raise NotImplementedError('torchaudio.load stub')
    ta_stub.load = _load
    sys.modules['torchaudio'] = ta_stub

# Basic OpenCV stub
if importlib.util.find_spec('cv2') is None:
    sys.modules['cv2'] = types.ModuleType('cv2')
