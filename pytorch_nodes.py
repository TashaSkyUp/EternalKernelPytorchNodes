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

# from pytorch_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
# from . import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


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
            optimizer = None
            if isinstance(optimizer_str, str):
                if "(" in optimizer_str and ")" in optimizer_str:
                    # Full expression, e.g. torch.optim.Adam(model.parameters(), lr=0.001)
                    optimizer = eval(optimizer_str)
                else:
                    # Simple name, e.g. 'Adam' or 'SGD'
                    if not optimizer_str.startswith("torch.optim."):
                        optimizer_str = f"torch.optim.{optimizer_str}(model.parameters(), lr=0.001, momentum=0.9)"
                    optimizer = eval(optimizer_str)
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
