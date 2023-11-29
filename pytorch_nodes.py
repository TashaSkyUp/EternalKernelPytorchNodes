import torch
import folder_paths
import torchvision
import torch.nn as nn
import functools
import torch

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
                "in_features": ("INT", {"min": 1}),
                "out_features": ("INT", {"min": 1}),
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
        model = torch.load(path)
        return (model,)

    def IS_CHANGED(s, path):
        # get the date modified of the model at the folder_path
        import os
        import hashlib

        data = str(os.path.getmtime(path))
        data = f"{path} {data}"
        m = hashlib.sha256()
        m.update(data.encode("utf-8"))
        return m.digest().hex()


@ETK_pytorch_base
class TrainModel:
    """
    allows users to train a model
    """

    @classmethod
    def INPUT_TYPES(cls):
        # iterate over tourch.nn tp fomd amy attributes that end in "Loss" and add them to the list
        loss_functions = []
        for attr in dir(torch.nn):
            if attr.endswith("Loss"):
                loss_functions.append(attr)

        return {

            "required": {
                "model": ("TORCH_MODEL",),

            },
            "optional": {
                "dataset": ("TORCH_DATASET",),
                "features tensor": ("TORCH_DATASET",),
                "labels tensor": ("TORCH_DATASET",),
                "epochs": ("INT", {"default": 1}),
                "batch_size": ("INT", {"default": 1}),
                "loss_function": (loss_functions,),
                "optimizer": ("STRING", {"default": "torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)"}),
                "metrics": ("STRING", {"default": "torch.nn.CrossEntropyLoss()"}),
                "device": ("STRING", {"default": "torch.device('cpu')"}),
            }
        }

    RETURN_TYPES = ("TORCH_MODEL", "LIST",)
    RETURN_NAMES = ("model", "metrics")
    FUNCTION = "train"

    def train(self, model, **kwargs):
        with torch.inference_mode(False):
            train_dataset = kwargs.get("dataset", None)
            features_tensor = kwargs.get("features tensor", None)
            labels_tensor = kwargs.get("labels tensor", None)
            if train_dataset is None:
                if features_tensor is None or labels_tensor is None:
                    raise ValueError("Must provide either a dataset or features tensor and labels tensor")
                else:
                    train_dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)

            epochs = kwargs.get("epochs", "1")
            batch_size = kwargs.get("batch_size", "1")
            loss_function = kwargs.get("loss_function", "torch.nn.CrossEntropyLoss()")
            optimizer = kwargs.get("optimizer", "torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)")
            metrics = kwargs.get("metrics", "torch.nn.CrossEntropyLoss()")

            loss_function = eval(f"torch.nn.{loss_function}()")
            optimizer = eval(optimizer)
            metrics = eval(metrics)
            epochs = int(epochs)

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True,
                                                           num_workers=0)

            for param in model.parameters():
                param.requires_grad = True

            metrics_out = []
            device = kwargs.get("device", "torch.device('cpu')")
            e_device = eval(device)
            model.to(e_device)
            # train_dataloader.to(eval(device))
            model.train()
            torch.set_grad_enabled(True)
            # clear the cache do other things to free up memory
            torch.cuda.empty_cache()

            for epoch in range(epochs):
                for batch_features, batch_labels in train_dataloader:
                    batch_features = batch_features.to(e_device)
                    batch_labels = batch_labels.to(e_device)
                    # batch_features.requires_grad = True
                    # batch_labels.requires_grad = True
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = loss_function(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch} complete")
                metrics_out.append(metrics(outputs, batch_labels))

            # convert metrics to a list of floats
            metrics_out = [float(metric) for metric in metrics_out]
            model.to(torch.device("cpu"))
            del train_dataloader
            torch.cuda.empty_cache()
            return (model, metrics_out,)


@ETK_pytorch_base
class TensorsToDataset:
    """
    allows users to convert a tensor to a dataset
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("TORCH_TENSOR",),  # The input data to perform inference on
                "labels": ("TORCH_TENSOR",),  # The input data to perform inference on
            },
            "optional": {
                "move to device": (["cuda", "cpu"], {"default": "cpu"}),

            }}

    RETURN_TYPES = ("TORCH_DATASET",)
    FUNCTION = "to_dataset"

    def to_dataset(self, input_data, labels, **kwargs):
        # perform inference on the input data or dataset using the model
        # and return the output
        input_data.requires_grad = True
        try:
            labels.requires_grad = True
        except RuntimeError as e:
            if "floating point and complex dtype" in str(e):
                pass
            else:
                raise e

        move_to_device = kwargs.get("move to device", "cpu")
        if move_to_device == "cuda":
            device_to_use = torch.device("cuda")
        else:
            device_to_use = torch.device("cpu")

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
                "dataset": ("TORCH_DATASET",),  # The input data to perform inference on
            }
        }

    RETURN_TYPES = ("TORCH_DATALOADER",)
    FUNCTION = "to_dataloader"

    def to_dataloader(self, dataset):
        # perform inference on the input data or dataset using the model
        # and return the output
        output = torch.utils.data.DataLoader(dataset)
        return (output,)
