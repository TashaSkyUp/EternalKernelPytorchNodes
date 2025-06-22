import torch
# import folder_paths
import torchvision
import torch.nn as nn
import functools
import torch
import json

try:
    from .config import config_settings
except ImportError as e:
    from config import config_settings

from . import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def ETK_pytorch_base(cls):
    cls.CATEGORY = "ETK/pytorch"
    # Add spaces to the camel case class name
    pretty_name = cls.__name__
    for i in range(1, len(pretty_name)):
        if pretty_name[i].isupper():
            pretty_name = pretty_name[:i] + " " + pretty_name[i:]
    cls.DISPLAY_NAME = pretty_name
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    NODE_DISPLAY_NAME_MAPPINGS[cls.DISPLAY_NAME] = pretty_name

    # Wrap the function defined in the FUNCTION attribute
    func_name = getattr(cls, "FUNCTION", None)
    if func_name and hasattr(cls, func_name):
        original_func = getattr(cls, func_name)

        @functools.wraps(original_func)
        def wrapped_func(*args, **kwargs):
            with torch.inference_mode(False):
                return original_func(*args, **kwargs)

        setattr(cls, func_name, wrapped_func)

    return cls


@ETK_pytorch_base
class PyTorchDatasetDownloader:
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": ("STRING", {"default": "mnist"}),
                "download_path": ("STRING", {"default": "./data"}),
                "mode": (["dataset", "tensors", ],),
            }
        }

    # Define the return types
    RETURN_TYPES = ("TORCH_DATASET", "TORCH_TENSOR", "TORCH_TENSOR",)
    RETURN_NAMES = ("Full Dataset", "Features tensor", "labels tensor",)
    FUNCTION = "download_dataset"

    # Method to download the dataset
    def download_dataset(self, dataset_name, download_path, mode):
        from torchvision import datasets, transforms
        import torch.utils.data as data_utils

        # check if the dataset already exists at the download_path

        # Downloading the dataset
        if dataset_name.lower() == "mnist":
            exists = False
            try:
                dataset = datasets.MNIST(
                    root=download_path,
                    train=True,
                    transform=transforms.ToTensor(),
                    download=False
                )
                exists = True
            except:
                dataset = datasets.MNIST(
                    root=download_path,
                    train=True,
                    transform=transforms.ToTensor(),
                    download=True
                )
        else:
            raise ValueError("Unsupported dataset")

        if mode == "tensors":
            # Split dataset into inputs and labels tensors
            inputs_tensor, labels_tensor = self.split_dataset_if_labels_exist(dataset)

            # Return tensors
            return (None, inputs_tensor, labels_tensor,)
        else:
            return (dataset, None, None,)

    def split_dataset_if_labels_exist(self, dataset):
        try:
            item = dataset[0]
            if isinstance(item, tuple) and len(item) > 1:
                # Extract inputs and labels
                inputs = [data[0] for data in dataset]
                labels = [data[1] for data in dataset]

                return torch.stack(inputs), torch.tensor(labels)
            else:
                return dataset, None
        except (TypeError, IndexError):
            return dataset, None


@ETK_pytorch_base
class DatsetSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("TORCH_DATASET",),
            }
        }

    RETURN_TYPES = ("TORCH_DATASET", "TORCH_DATASET")
    RETURN_NAMES = ("train_ds", "val_ds")
    FUNCTION = "split_dataset"

    def split_dataset(self, dataset):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import random_split

        ds_len = len(dataset)
        val_ratio = 0.2
        # use random_split to create a validation set
        train_ds, val_ds = random_split(dataset, [int(ds_len * (1 - val_ratio)), int(ds_len * val_ratio)])

        # train_ds is now of type torch.utils.data.Dataset throw an error if it is not
        if not isinstance(train_ds, torch.utils.data.Dataset):
            raise TypeError("train_ds is not of type torch.utils.data.Dataset")

        # val_ds is now of type torch.utils.data.Dataset throw an error if it is not
        if not isinstance(val_ds, torch.utils.data.Dataset):
            raise TypeError("val_ds is not of type torch.utils.data.Dataset")

        return (train_ds, val_ds,)


@ETK_pytorch_base
class AddLinearLayerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "in_features": ("INT", {"min": 1, "max": 2 ** 24}),
                "out_features": ("INT", {"min": 1, "max": 1_000_000_000}),
                "bias": ([True, False],),
                "initialization": (["default", "xavier_uniform", "xavier_normal"],),
                "dtype": (["float32", "float64", "float16", "int32", "int64", "int16", "int8", "uint8"],),
            }
        }

    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "add_linear_layer"
    CATEGORY = "model"

    def add_linear_layer(self, model, in_features, out_features, bias=True, initialization="default", dtype="float32"):
        import torch
        # Create the new linear layer
        use_dtype = getattr(torch, dtype)

        new_linear_layer = nn.Linear(in_features, out_features, bias=bias, dtype=use_dtype)
        new_linear_layer.requires_grad_(True)

        # Initialize the weights if required
        if initialization == "xavier_uniform":
            nn.init.xavier_uniform_(new_linear_layer.weight)
        elif initialization == "xavier_normal":
            nn.init.xavier_normal_(new_linear_layer.weight)

        # Reconstruct the nn.Sequential model to include the new layer
        if isinstance(model, nn.Sequential):
            # layers = list(model.children())
            # layers.append(new_linear_layer)
            # new_model = nn.Sequential(*layers)
            model.insert(len(model), new_linear_layer)
        else:
            raise TypeError("The provided model is not a nn.Sequential model.")

        # Returning the modified model
        return (model,)


@ETK_pytorch_base
class AddSoftmaxLayerNode:
    """
    SOFTMAX is a function that turns a vector of K real values into a vector of K real values that sum to 1.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "dim": ("INT", {"min": 0}),
            },
            "optional": {
                "log_softmax": ([False, True],),
            }

        }

    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "add_softmax_layer"
    CATEGORY = "model"

    def add_softmax_layer(self, model, dim, log_softmax=False):
        import torch
        # Create the new softmax layer
        if log_softmax:
            new_softmax_layer = nn.LogSoftmax(dim=dim)
        else:
            new_softmax_layer = nn.Softmax(dim=dim)

        # Reconstruct the nn.Sequential model to include the new layer
        if isinstance(model, nn.Sequential):
            # layers = list(model.children())
            # layers.append(new_linear_layer)
            # new_model = nn.Sequential(*layers)
            model.insert(len(model), new_softmax_layer)
        else:
            raise TypeError("The provided model is not a nn.Sequential model.")

        # Returning the modified model
        return (model,)


@ETK_pytorch_base
class AddConvLayer:
    """
    Convolutional layer
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "one_d": ("BOOLEAN", {"default": False}),
                "in_channels": ("INT", {"min": 1}),
                "out_channels": ("INT", {"min": 1}),
                "kernel_size": ("INT", {"min": 1}),
                "stride": ("INT", {"min": 1}),
                "padding": ("STRING", {"default":"(0,0)"}),
                "bias": ([True, False],),
                "initialization": (["default", "xavier_uniform", "xavier_normal"],),
                "dtype": (["float32", "float64", "float16", "int32", "int64", "int16", "int8", "uint8"],),
            }
        }

    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "add_conv_layer"
    CATEGORY = "model"

    def add_conv_layer(self, model, one_d, in_channels, out_channels, kernel_size, stride, padding, bias=True,
                       initialization="default", dtype="float32"):
        import torch
        # Create the new convolutional layer
        use_dtype = getattr(torch, dtype)
        padding=padding.replace("(", "")
        padding=padding.replace(")", "")
        padding = tuple(map(int, padding.split(",")))
        if not one_d:
            new_conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias,
                                       dtype=use_dtype)
        else:
            new_conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias,
                                       dtype=use_dtype)

        new_conv_layer.requires_grad_(True)

        # Initialize the weights if required
        if initialization == "xavier_uniform":
            nn.init.xavier_uniform_(new_conv_layer.weight)
        elif initialization == "xavier_normal":
            nn.init.xavier_normal_(new_conv_layer.weight)

        # Reconstruct the nn.Sequential model to include the new layer
        if isinstance(model, nn.Sequential):
            # layers = list(model.children())
            # layers.append(new_linear_layer)
            # new_model = nn.Sequential(*layers)
            model.insert(len(model), new_conv_layer)
        else:
            raise TypeError("The provided model is not a nn.Sequential model.")

        # Returning the modified model
        return (model,)


@ETK_pytorch_base
class SequentialModelProvider:
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        # Inputs can be used to customize the default architecture
        return {
            "required": {
                "name": ("STRING", {"default": "sequential_model"}),
            }
        }

    FUNCTION = "provide_default_model"
    # Define the return types
    RETURN_TYPES = ("TORCH_MODEL",)

    # Method to provide the default sequential model
    def provide_default_model(self, name):
        # Logic to create the default sequential model based on input configuration
        model = nn.Sequential()

        return (model,)


@ETK_pytorch_base
class PyTorchInferenceNode:
    """
    allows users to perform inference using a pre-trained PyTorch model
    on a given tensor or dataset
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TORCH_MODEL",),  # The pre-trained PyTorch model
            },
            "optional": {
                "input_data": ("TORCH_TENSOR",),  # The input data to perform inference on
                "dataset": ("TORCH_DATASET",),  # The dataset to perform inference on
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR",)
    FUNCTION = "inference"

    def inference(self, model, input_data=None, dataset=None):
        # perform inference on the input data or dataset using the model
        # and return the output
        if input_data != None and dataset != None:
            raise ValueError("Only one of input_data or dataset can be provided")
        elif input_data is not None:
            x = input_data
        elif dataset is not None:
            x = dataset.dataset
            # prepare x so that it can be passed to the model
            # e.g. convert to torch.Tensor
            # avoid the error: ValueError: only one element tensors can be converted to Python scalars
            if isinstance(x, torch.utils.data.Dataset):
                x = next(iter(x))[0]
            elif isinstance(x, torch.utils.data.DataLoader):
                x = next(iter(x))[0]
            elif isinstance(x, torch.Tensor):
                pass



        else:
            raise ValueError("Either input_data or dataset must be provided")

        mdl_device = next(model.parameters()).device
        # check that x and the model are on the same device
        if x.device != mdl_device:
            x = x.to(mdl_device)
        output = model(x)
        return (output,)


@ETK_pytorch_base
class FlattenTensor:
    """
    allows users to flatten a tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("TORCH_TENSOR",),  # The input data to perform inference on
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR",)
    FUNCTION = "flatten"

    def flatten(self, input_data):
        # perform inference on the input data or dataset using the model
        # and return the output
        output = input_data.flatten()
        return (output,)


@ETK_pytorch_base
class FlattenDataset:
    """
    allows users to flatten a dataset
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("TORCH_DATASET",),  # The input data to perform inference on
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR",)
    FUNCTION = "flatten"

    def flatten(self, dataset):
        # perform inference on the input data or dataset using the model
        # and return the output
        output = torch.flatten(dataset.data)
        return (output,)


@ETK_pytorch_base
class ReshapeTensor:
    """
    allows users to reshape a tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("TORCH_TENSOR",),  # The input data to perform inference on
                "shape": ("STRING", {"default": "1, -1"}),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR",)
    FUNCTION = "reshape"

    def reshape(self, input_data, shape):
        # perform inference on the input data or dataset using the model
        # and return the output
        shape = tuple(map(int, shape.split(",")))
        output = input_data.reshape(shape)
        return (output,)


@ETK_pytorch_base
class ChangeTensorType:
    """
    allows users to change the type of a tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("TORCH_TENSOR",),  # The input data to perform inference on
                "dtype": ("STRING", {"default": "torch.float32"}),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR",)
    FUNCTION = "change_type"

    def change_type(self, input_data, dtype):
        # perform inference on the input data or dataset using the model
        # and return the output
        dtype = eval(dtype)
        output = input_data.type(dtype)
        return (output,)


@ETK_pytorch_base
class TensorToList:
    """
    allows users to convert a tensor to a list
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("TORCH_TENSOR",),  # The input data to perform inference on
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "tolist"

    def tolist(self, input_data):
        # perform inference on the input data or dataset using the model
        # and return the output
        output = input_data.tolist()
        return (output,)


@ETK_pytorch_base
class SliceTensor:
    """
    allows users to slice a tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("TORCH_TENSOR",),  # The input data to perform inference on
                "start": ("STRING", {"default": "0"}),
                "end": ("STRING", {"default": "5"}),
                "step": ("STRING", {"default": "1"}),
            }
        }

    RETURN_TYPES = ("TORCH_TENSOR",)
    FUNCTION = "slice"

    def slice(self, input_data, start, end, step):
        # perform inference on the input data or dataset using the model
        # and return the output
        start = int(start)
        end = int(end)
        step = int(step)
        output = input_data[start:end:step]
        return (output,)


@ETK_pytorch_base
class SaveModel:
    """
    allows users to save a model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TORCH_MODEL",),  # The input data to perform inference on
                "path": ("STRING", {"default": ".data/models/model.pt"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    # define that this is an output
    OUTPUT_NODE = True

    def save(self, model, path):
        import os
        # perform inference on the input data or dataset using the model
        # and return the output
        # create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model, path)
        return (path,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        import hashlib
        data = str(kwargs)
        m = hashlib.sha256()
        m.update(data.encode("utf-8"))
        return m.digest().hex()


@ETK_pytorch_base
class LoadModel:
    """
    allows users to load a model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ".data/models/model.pt"}),
            }
        }

    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "load"

    def load(self, path):
        # perform inference on the input data or dataset using the model
        # and return the output
        print("loaded model: ", path)
        model = torch.load(path)
        return (model,)

    @classmethod
    def IS_CHANGED(s, path):
        # get the date modified of the model at the folder_path
        import os
        import hashlib

        data = str(os.path.getmtime(path))
        data = f"{path} {data}"
        m = hashlib.sha256()
        m.update(data.encode("utf-8"))
        return m.digest().hex()


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


@ETK_pytorch_base
class TrainModel:
    """
    allows users to train a model
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Gather available loss functions
        loss_functions = [attr for attr in dir(torch.nn) if attr.endswith("Loss") and callable(getattr(torch.nn, attr))]
        # Gather available optimizers
        optimizers = [attr for attr in dir(torch.optim) if attr[0].isupper() and callable(getattr(torch.optim, attr))]
        # Gather available metrics (for now, just use loss functions + 'accuracy')
        metrics = ["accuracy"] + loss_functions
        return {
            "required": {
                "model": ("TORCH_MODEL",),

            },
            "optional": {
                "dataset": ("TORCH_DATASET",),
                "features tensor": ("TORCH_TENSOR",),
                "labels tensor": ("TORCH_TENSOR",),
                "epochs": ("INT", {"default": 1, "min": 1, "max": 2 ** 24}),
                "batch_size": ("INT", {"default": 1}),
                "loss_function": (loss_functions, {"default": "CrossEntropyLoss"}),
                "optimizer": (optimizers, {"default": "SGD"}),
                "metrics": (metrics, {"default": "accuracy"}),
                "device": ("STRING", {"default": "torch.device('cpu')"}),
                "create_samples": (["TRUE", "FALSE"], {"default": "FALSE"}),
            }
        }

    RETURN_TYPES = ("TORCH_MODEL", "LIST", "TORCH_MODEL")
    RETURN_NAMES = ("model", "metrics", "best model")
    FUNCTION = "train"

    def reset_optimizer_weights(self, optimizer):
        # Loop over the optimizer state
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                # Reset state dictionary for each parameter
                optimizer.state[param] = torch.optim.Optimizer.StateDict()

    def train(self, model, **kwargs):
        # Consistent boolean handling for create_samples
        create_samples = kwargs.get("create_samples", "FALSE")
        if isinstance(create_samples, str):
            create_samples = create_samples.upper() == "TRUE"
        else:
            create_samples = bool(create_samples)

        with torch.inference_mode(False):
            train_dataset = kwargs.get("dataset", None)
            features_tensor = kwargs.get("features tensor", None)
            labels_tensor = kwargs.get("labels tensor", None)
            if train_dataset is None:
                if features_tensor is None or labels_tensor is None:
                    raise ValueError("Must provide either a dataset or features tensor and labels tensor")
                else:
                    train_dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)

            # Consistent int conversion for epochs and batch_size
            epochs = kwargs.get("epochs", 1)
            batch_size = kwargs.get("batch_size", 1)
            try:
                epochs = int(epochs)
            except Exception:
                epochs = 1
            try:
                batch_size = int(batch_size)
            except Exception:
                batch_size = 1

            # Consistent string handling for loss_function, optimizer, metrics
            loss_function = kwargs.get("loss_function", "CrossEntropyLoss")
            optimizer_str = kwargs.get("optimizer", "SGD")
            metrics_str = kwargs.get("metrics", "CrossEntropyLoss")

            # Evaluate loss function
            if isinstance(loss_function, str):
                if not loss_function.startswith("torch.nn."):
                    loss_function = f"torch.nn.{loss_function}()"
                loss_function = eval(loss_function)

            # Evaluate optimizer
            import inspect
            optimizer = None
            if isinstance(optimizer_str, str):
                if "(" in optimizer_str and ")" in optimizer_str:
                    # Full expression, e.g. torch.optim.Adam(model.parameters(), lr=0.001)
                    optimizer = eval(optimizer_str)
                else:
                    # Simple name, e.g. 'Adam' or 'SGD'
                    opt_class = getattr(torch.optim, optimizer_str)
                    sig = inspect.signature(opt_class.__init__)
                    opt_kwargs = {"lr": 0.001}
                    if "momentum" in sig.parameters:
                        opt_kwargs["momentum"] = 0.9
                    optimizer = opt_class(model.parameters(), **opt_kwargs)
            else:
                optimizer = optimizer_str  # fallback

            # Evaluate metrics
            metrics = None
            if isinstance(metrics_str, str):
                if metrics_str.lower() == "accuracy":
                    def accuracy_fn(outputs, labels):
                        # For classification: outputs are logits or probabilities
                        if outputs.dim() > 1 and outputs.size(1) > 1:
                            preds = outputs.argmax(dim=1)
                        else:
                            preds = (outputs > 0.5).long()
                        if labels.dim() > 1 and labels.size(1) > 1:
                            labels = labels.argmax(dim=1)
                        return (preds == labels).float().mean().item()
                    metrics = accuracy_fn
                elif not metrics_str.startswith("torch.nn."):
                    metrics = eval(f"torch.nn.{metrics_str}()")
                else:
                    metrics = eval(metrics_str)
            else:
                metrics = metrics_str  # fallback

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            for param in model.parameters():
                param.requires_grad = True

            metrics_out = []
            device = kwargs.get("device", "torch.device('cpu')")
            e_device = eval(device) if isinstance(device, str) else device

            optimizer_to(optimizer, e_device)
            model.to(e_device)
            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()
            best_l = 1024 ** 3
            best_epoch = 0

            import uuid
            import os
            tmpdir = config_settings.get("tmp_dir", "./temp")
            try:
                os.makedirs(tmpdir, exist_ok=True)
            except PermissionError:
                tmpdir = "./temp"
                os.makedirs(tmpdir, exist_ok=True)
            tmp = os.path.join(tmpdir, str(uuid.uuid4()))

            for epoch in range(epochs):
                tot_loss = 0
                for batch_features, batch_labels in train_dataloader:
                    batch_features = batch_features.to(e_device)
                    batch_labels = batch_labels.to(e_device)
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = loss_function(outputs, batch_labels)
                    tot_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                l = metrics(outputs, batch_labels)
                metrics_out.append(l)
                if os.path.exists(tmp):
                    os.remove(tmp)
                torch.save(model, tmp, _use_new_zipfile_serialization=True, pickle_protocol=4)

                if tot_loss < best_l:
                    if create_samples:
                        import numpy as np
                        import PIL
                        from PIL import Image
                        best_l = tot_loss
                        out_X = None
                        for i in range(len(outputs)):
                            out_tensor = self.make_image_tensor(outputs, i)
                            if out_X is None:
                                out_X = out_tensor
                            else:
                                out_X = np.concatenate((out_X, out_tensor), axis=1)
                        out_y = None
                        for i in range(len(outputs)):
                            label_Tensor = self.make_image_tensor(batch_labels, i)
                            if out_y is None:
                                out_y = label_Tensor
                            else:
                                out_y = np.concatenate((out_y, label_Tensor), axis=1)
                        out_tensor = np.concatenate((out_X, out_y), axis=0)
                        mse = np.mean((out_X - out_y) ** 2)
                        out_tensor = PIL.Image.fromarray(out_tensor)
                        out_tensor.save(f"out{i}_{mse}_{str(loss.item())}.png")
                if os.path.exists(tmp):
                    best_model = torch.load(tmp, weights_only=False)
            # Clean return, no debug prints
            return (model, metrics_out, best_model,)


# --- BEGIN RESTORED NODES FROM 5486f39 ---

@ETK_pytorch_base
class TensorsToDataset:
    """
    allows users to convert a tensor to a dataset
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("TORCH_TENSOR",),
                "labels": ("TORCH_TENSOR",),
            },
            "optional": {
                "move to device": (["cuda", "cpu"], {"default": "cpu"}),
            }
        }
    RETURN_TYPES = ("TORCH_DATASET",)
    FUNCTION = "to_dataset"
    def to_dataset(self, input_data, labels, **kwargs):
        if input_data.is_leaf:
            input_data.requires_grad = True
        try:
            if labels.is_leaf:
                labels.requires_grad = True
        except RuntimeError as e:
            if "floating point and complex dtype" not in str(e):
                raise e
        move_to_device = kwargs.get("move to device", "cpu")
        device_to_use = torch.device(move_to_device)
        input_data = input_data.to(device_to_use)
        labels = labels.to(device_to_use)
        output = torch.utils.data.TensorDataset(input_data, labels)
        return (output,)

@ETK_pytorch_base
class DatasetToDataloader:
    """
    allows users to convert a dataset to a dataloader
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("TORCH_DATASET",),
            }
        }
    RETURN_TYPES = ("TORCH_DATALOADER",)
    FUNCTION = "to_dataloader"
    def to_dataloader(self, dataset):
        output = torch.utils.data.DataLoader(dataset)
        return (output,)

@ETK_pytorch_base
class ComfyUIImageToPytorchTENSOR:
    """
    this just renames the object for compatibility with comfyui->pytorch
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("TORCH_TENSOR", "TORCH_TENSOR",)
    RETURN_NAMES = ("b,h,w,c", "b,h,w,mean(c)",)
    FUNCTION = "to_tensor"
    def to_tensor(self, image):
        out1 = torch.tensor(image)
        out2 = torch.tensor(image).mean(dim=3)
        return (out1, out2,)

@ETK_pytorch_base
class ListToTensor:
    """
    this just renames the object for compatibility with comfyui->pytorch
    """
    @classmethod
    def INPUT_TYPES(cls):
        valid_dtypes_str = [str(dt) for k, dt in torch.__dict__.items() if isinstance(dt, torch.dtype)]
        return {
            "required": {
                "list": ("LIST",),
            },
            "optional": {
                "dtype": (valid_dtypes_str,),
            }
        }
    RETURN_TYPES = ("TORCH_TENSOR",)
    FUNCTION = "to_tensor"
    def to_tensor(self, list, dtype=str(torch.float32)):
        valid_dtypes_str = [str(dt) for k, dt in torch.__dict__.items() if isinstance(dt, torch.dtype)]
        valid_dtypes = [dt for k, dt in torch.__dict__.items() if isinstance(dt, torch.dtype)]
        if dtype not in valid_dtypes_str:
            raise ValueError(f"Unsupported dtype: {dtype}")
        dto = valid_dtypes[valid_dtypes_str.index(dtype)]
        output = torch.tensor(list, dtype=dto)
        return (output,)

@ETK_pytorch_base
class Activation:
    """
    uses values in torch.nn.modules.activation to create an activation function
    """
    @classmethod
    def INPUT_TYPES(cls):
        activation_str_names = [i for i in torch.nn.modules.activation.__dict__.keys() if i[0] != "_"]
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "activation": (activation_str_names,),
            }
        }
    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "activation"
    def activation(self, model, activation):
        act = getattr(torch.nn.modules.activation, activation)()
        model = model.insert(len(model), act)
        return (model,)

@ETK_pytorch_base
class AddReshapeLayer:
    """
    Adds a reshape layer to the model.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "shape": ("STRING", {"default": "1, -1"}),
            }
        }
    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "add_reshape_layer"
    CATEGORY = "model"
    def add_reshape_layer(self, model, shape):
        import torch.nn as nn
        t = tuple([int(i) for i in shape.split(",")])
        reshape_layer_f = nn.Flatten(0)
        reshape_layer_u = nn.Unflatten(0, t)
        if isinstance(model, nn.Sequential):
            model.insert(len(model), reshape_layer_f)
            model.insert(len(model), reshape_layer_u)
        else:
            raise TypeError("The provided model is not an nn.Sequential model.")
        return (model,)

@ETK_pytorch_base
class AddTransformerLayer:
    """
    Adds a transformer encoder layer to the model.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "input_features": ("INT", {"min": 1, "max": 2 ** 24}),
                "num_heads": ("INT", {"min": 1}),
                "feedforward_dim": ("INT", {"min": 1}),
                "dropout": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.1}),
            }
        }
    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "add_transformer_layer"
    CATEGORY = "model"
    def add_transformer_layer(self, model, input_features, num_heads, feedforward_dim, dropout=0.1):
        import torch.nn as nn
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout
        )
        if isinstance(model, nn.Sequential):
            model.insert(len(model), transformer_layer)
        else:
            raise TypeError("The provided model is not a nn.Sequential model.")
        return (model,)

@ETK_pytorch_base
class PyTorchToDevice:
    """
    moves a pytorch object to a device
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda", "cpu"], {"default": "cpu"}),
            },
            "optional": {
                "model": ("TORCH_MODEL",),
                "tensor": ("TORCH_TENSOR",),
            }
        }
    RETURN_TYPES = ("TORCH_MODEL", "TORCH_TENSOR",)
    FUNCTION = "to_device"
    def to_device(self, device, model=None, tensor=None):
        device_to_use = torch.device(device)
        model_out = model.to(device_to_use) if model is not None else None
        tensor_out = tensor.to(device_to_use) if tensor is not None else None
        return (model_out, tensor_out,)

@ETK_pytorch_base
class ExtractLayersAsModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "start_idx": ("INT", {"min": 0}),
                "end_idx": ("INT", {"min": 0}),
                "freeze": (["True", "False"], {"default": "False"}),
            }
        }
    RETURN_TYPES = ("TORCH_MODEL", "STRING",)
    RETURN_NAMES = ("model", "shapes_str",)
    FUNCTION = "extract_layers"
    CATEGORY = "model"
    def extract_layers(self, model, start_idx, end_idx, freeze=False):
        if freeze == "True":
            freeze = True
        else:
            freeze = False
        if not isinstance(model, nn.Sequential):
            raise TypeError("The provided model is not a nn.Sequential model.")
        if start_idx < 0 or end_idx < start_idx or end_idx >= len(model):
            raise ValueError("Invalid start or end index.")
        extracted_layers = nn.Sequential(*list(model.children())[start_idx:end_idx + 1])
        shapes = []
        for i in range(start_idx, end_idx + 1):
            if hasattr(model[i], "in_features"):
                shapes.append(model[i].in_features)
            if hasattr(model[i], "out_features"):
                shapes.append(model[i].out_features)
        shapes_str = str(shapes)
        if freeze:
            for param in extracted_layers.parameters():
                param.requires_grad = False
        return (extracted_layers, shapes_str,)

@ETK_pytorch_base
class AddModelAsLayer:
    """
    given a model main and model addition adds the model addition to the end of the model main
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_main": ("TORCH_MODEL",),
                "model_addition": ("TORCH_MODEL",),
            }
        }
    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "add_model"
    CATEGORY = "model"
    def add_model(self, model_main, model_addition):
        if not isinstance(model_main, nn.Sequential):
            raise TypeError("model_main is not a nn.Sequential model.")
        if not isinstance(model_addition, nn.Sequential):
            raise TypeError("model_addition is not a nn.Sequential model.")
        for layer in model_addition.children():
            model_main.add_module(f"added_layer_{len(model_main)}", layer)
        return (model_main,)

@ETK_pytorch_base
class RandomTensor:
    """
    creates a random tensor of a given shape and dtype and initialization method
    """
    @classmethod
    def INPUT_TYPES(s):
        valid_dtypes_str = [str(dt) for k, dt in torch.__dict__.items() if isinstance(dt, torch.dtype)]
        return {
            "required": {
                "shape": ("STRING", {"default": "(1,1)"}),
            },
            "optional": {
                "dtype": (valid_dtypes_str,),
                "init_method": (
                    ["rand", "randn", "randint", "randint_like", "rand_like", "randn_like"], {"default": "rand"}),
            }
        }
    RETURN_TYPES = ("TORCH_TENSOR",)
    FUNCTION = "random_tensor"
    CATEGORY = "tensor"
    def random_tensor(self, shape, dtype=str(torch.float32), init_method="rand"):
        valid_dtypes_str = [str(dt) for k, dt in torch.__dict__.items() if isinstance(dt, torch.dtype)]
        valid_dtypes = [dt for k, dt in torch.__dict__.items() if isinstance(dt, torch.dtype)]
        if dtype not in valid_dtypes_str:
            raise ValueError(f"Unsupported dtype: {dtype}")
        dto = valid_dtypes[valid_dtypes_str.index(dtype)]

        shape = shape[1:-1]
        shape = [int(s) for s in shape.split(",")]
        shape = tuple(shape)

        if init_method == "rand":
            output = torch.rand(shape, dtype=dto)
        elif init_method == "randn":
            output = torch.randn(shape, dtype=dto)
        elif init_method == "randint":
            output = torch.randint(0, 100, shape, dtype=dto)
        elif init_method == "randint_like":
            temp = torch.empty(shape, dtype=dto)
            output = torch.randint_like(temp, 0, 100)
        elif init_method == "rand_like":
            temp = torch.empty(shape, dtype=dto)
            output = torch.rand_like(temp)
        elif init_method == "randn_like":
            temp = torch.empty(shape, dtype=dto)
            output = torch.randn_like(temp)
        else:
            raise ValueError(f"Unsupported init_method: {init_method}")
        return (output,)

@ETK_pytorch_base
class GridSearchTraining:
    """
    Performs grid search training using the TrainModel class.
    """
    @classmethod
    def INPUT_TYPES(cls):
        default_param_grid = json.dumps({
            'epochs': [1, 2],
            'batch_size': [32, 64],
            'loss_function': ['MSELoss', 'CrossEntropyLoss']
        })
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "param_grid": ("STRING", {"default": default_param_grid}),
            },
            "optional": {
                "dataset": ("TORCH_DATASET",),
                "features tensor": ("TORCH_TENSOR",),
                "labels tensor": ("TORCH_TENSOR",),
            }
        }
    RETURN_TYPES = ("TORCH_MODEL", "DICT", "LIST")
    RETURN_NAMES = ("best_model", "best_params", "all_metrics")
    FUNCTION = "grid_search_train"
    def grid_search_train(self, model, param_grid, **kwargs):
        from itertools import product
        param_grid_dict = json.loads(param_grid)
        best_model = None
        best_params = None
        all_metrics = []
        param_combinations = list(product(*param_grid_dict.values()))
        for params in param_combinations:
            train_kwargs = dict(zip(param_grid_dict.keys(), params))
            train_kwargs.update(kwargs)
            trainer = TrainModel()
            trained_model, metrics, _ = trainer.train(model, **train_kwargs)
            all_metrics.append(metrics)
            if best_model is None or self.is_better(metrics, all_metrics):
                best_model = trained_model
                best_params = train_kwargs
        return best_model, best_params, all_metrics
    @staticmethod
    def is_better(current_metrics, all_metrics):
        return current_metrics[-1] < min(all_metrics, key=lambda m: m[-1])[-1]

@ETK_pytorch_base
class SaveTorchTensor:
    """
    Saves a torch tensor to a file using torch.save which has the required parameters of obj and f
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TORCH_TENSOR", {"default": None}),
                "file": ("STRING", {"default": "/somewhere/some.pt"}),
            },
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_torch_tensor"
    OUTPUT_NODE = True
    def save_torch_tensor(self, tensor, file):
        torch.save(tensor, file)
        return (file,)

@ETK_pytorch_base
class LoadTorchTensor:
    """
    Loads a torch tensor from a file using torch.load
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"default": "/somewhere/some.pt"}),
            },
        }
    RETURN_TYPES = ("TORCH_TENSOR",)
    FUNCTION = "load_torch_tensor"
    def load_torch_tensor(self, file):
        return (torch.load(file),)

@ETK_pytorch_base
class FuncModifyModel:
    """
    allows users to modify a model using a function
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "function": ("STRING", {"default": "model", "multiline": True}),
            }
        }
    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "modify_model"
    def modify_model(self, model, function):
        from copy import deepcopy
        old_model = deepcopy(model)
        exec(function)
        new_model = model
        if old_model == new_model:
            raise ValueError("The model was not modified.")
        return (new_model,)

@ETK_pytorch_base
class PlotSeriesString:
    """
    Plots a series of strings
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "series": ("STRING", {"default": "1\n2\n3\n4\n5\n6\n7\n8\n9\n10"}),
            },
            "optional": {
                "title": ("STRING", {"default": "Series Plot"}),
                "xlabel": ("STRING", {"default": "X-axis"}),
                "ylabel": ("STRING", {"default": "Y-axis"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "plot_series_string"
    OUTPUT_NODE = True
    def plot_series_string(self, series, title="Series Plot", xlabel="X-axis", ylabel="Y-axis"):
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        import PIL
        from PIL import Image
        if isinstance(series, str):
            series = series.split("\n")
        elif isinstance(series, list):
            pass
        else:
            raise ValueError("series must be a string or a list")
        series = [float(s) for s in series]
        plt.plot(series)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        im = np.array(im)
        im = torch.tensor(im)
        im = im.unsqueeze(0)
        im = im.float()
        im = im / 255.0
        plt.close()
        buf.close()
        return (im,)


@ETK_pytorch_base
class AddBatchNormLayer:
    """Adds a BatchNorm layer to an ``nn.Sequential`` model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "num_features": ("INT", {"min": 1}),
                "one_d": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "add_batchnorm_layer"
    CATEGORY = "model"

    def add_batchnorm_layer(self, model, num_features, one_d=True):
        layer = nn.BatchNorm1d(num_features) if one_d else nn.BatchNorm2d(num_features)
        if isinstance(model, nn.Sequential):
            model.insert(len(model), layer)
        else:
            raise TypeError("The provided model is not a nn.Sequential model.")
        return (model,)


@ETK_pytorch_base
class AddDropoutLayer:
    """Adds a Dropout layer to an ``nn.Sequential`` model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "p": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.5}),
            },
            "optional": {
                "inplace": (["True", "False"], {"default": "False"}),
            },
        }

    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "add_dropout_layer"
    CATEGORY = "model"

    def add_dropout_layer(self, model, p=0.5, inplace="False"):
        inplace = True if inplace in (True, "True") else False
        layer = nn.Dropout(p=p, inplace=inplace)
        if isinstance(model, nn.Sequential):
            model.insert(len(model), layer)
        else:
            raise TypeError("The provided model is not a nn.Sequential model.")
        return (model,)


@ETK_pytorch_base
class SetModelTrainable:
    """Sets ``requires_grad`` for all parameters of a model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TORCH_MODEL",),
                "requires_grad": (["True", "False"], {"default": "False"}),
            }
        }

    RETURN_TYPES = ("TORCH_MODEL",)
    FUNCTION = "set_trainable"
    CATEGORY = "model"

    def set_trainable(self, model, requires_grad="False"):
        flag = True if requires_grad in (True, "True") else False
        for param in model.parameters():
            param.requires_grad = flag
        return (model,)

