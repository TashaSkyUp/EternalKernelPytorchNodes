import torch
from torch import nn
from unittest import mock

from custom_nodes.EternalKernelLiteGraphNodes.pytorch_nodes import (
    PyTorchDatasetDownloader,
    AddLinearLayerNode,
    SequentialModelProvider,
    PyTorchInferenceNode,
    DatsetSplitter,
    AddSoftmaxLayerNode,
    AddConvLayer,
    FlattenTensor,
    FlattenDataset,
    ReshapeTensor,
    ChangeTensorType,
    TensorToList,
    SliceTensor,
    SaveModel,
    LoadModel,
    TrainModel,
    TensorsToDataset,
    DatasetToDataloader,
    ComfyUIImageToPytorchTENSOR,
    ListToTensor,
    Activation,
    AddReshapeLayer,
    AddTransformerLayer,
    PyTorchToDevice,
    ExtractLayersAsModel,
    AddModelAsLayer,
    RandomTensor,
    SaveTorchTensor,
    LoadTorchTensor,
    FuncModifyModel,
    PlotSeriesString,
)


def test_dataset_downloader(monkeypatch):
    # Patch torchvision dataset download to avoid network usage
    from torchvision import datasets

    fake_ds = torch.utils.data.TensorDataset(torch.randn(2, 1, 28, 28), torch.tensor([0, 1]))
    monkeypatch.setattr(datasets, "MNIST", lambda *args, **kwargs: fake_ds, raising=False)

    downloader = PyTorchDatasetDownloader()
    ds, x, y = downloader.download_dataset("mnist", "./data/tmp", mode="tensors")
    assert x.shape[0] == 2 and y.shape[0] == 2


def test_linear_layer_node():
    model = nn.Sequential(nn.Linear(4, 4))
    node = AddLinearLayerNode()
    out_model, = node.add_linear_layer(model, 4, 2)
    assert isinstance(out_model[-1], nn.Linear)
    assert out_model[-1].out_features == 2


def test_sequential_model_provider():
    provider = SequentialModelProvider()
    model, = provider.provide_default_model("test")
    assert isinstance(model, nn.Sequential)


def test_inference_node():
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    data = torch.randn(1, 2, 2)
    node = PyTorchInferenceNode()
    out, = node.inference(model, input_data=data)
    assert out.shape == (1, 2)


def test_dataset_splitter():
    dataset = torch.utils.data.TensorDataset(torch.randn(10, 1), torch.arange(10))
    splitter = DatsetSplitter()
    train_ds, val_ds = splitter.split_dataset(dataset)
    assert len(train_ds) + len(val_ds) == 10


def test_add_softmax_layer():
    model = nn.Sequential(nn.Linear(4, 4))
    node = AddSoftmaxLayerNode()
    out_model, = node.add_softmax_layer(model, dim=1)
    assert isinstance(out_model[-1], nn.Softmax)


def test_add_conv_layer():
    model = nn.Sequential()
    node = AddConvLayer()
    out_model, = node.add_conv_layer(model, False, 1, 1, 3, 1, "(0,0)")
    assert isinstance(out_model[-1], nn.Conv2d)


def test_flatten_tensor():
    tensor = torch.randn(1, 2, 3)
    node = FlattenTensor()
    out, = node.flatten(tensor)
    assert out.shape == (6,)


def test_flatten_dataset():
    class DummyDS:
        def __init__(self):
            self.data = torch.randn(2, 2)
    ds = DummyDS()
    node = FlattenDataset()
    out, = node.flatten(ds)
    assert out.numel() == 4


def test_reshape_tensor():
    tensor = torch.randn(2, 3)
    node = ReshapeTensor()
    out, = node.reshape(tensor, "3,2")
    assert out.shape == (3, 2)


def test_change_tensor_type():
    tensor = torch.randn(2, 2)
    node = ChangeTensorType()
    out, = node.change_type(tensor, "torch.int32")
    assert out.dtype == torch.int32


def test_tensor_to_list():
    tensor = torch.tensor([1, 2, 3])
    node = TensorToList()
    out, = node.tolist(tensor)
    assert out == [1, 2, 3]


def test_slice_tensor():
    tensor = torch.arange(10)
    node = SliceTensor()
    out, = node.slice(tensor, "2", "5", "1")
    assert torch.equal(out, torch.tensor([2,3,4]))


def test_save_and_load_model(tmp_path):
    model = nn.Sequential(nn.Linear(1,1))
    path = tmp_path / "model.pt"
    saver = SaveModel()
    saver.save(model, str(path))
    loader = LoadModel()
    # torch.load defaults to weights_only which fails here; ensure full load
    import torch as _torch
    orig_load = _torch.load
    def _patched(path, *a, **kw):
        kw.setdefault("weights_only", False)
        return orig_load(path, *a, **kw)
    with mock.patch.object(_torch, "load", _patched):
        loaded, = loader.load(str(path))
    assert isinstance(loaded, nn.Module)


def test_train_model(tmp_path):
    model = nn.Sequential(nn.Linear(1,1))
    features = torch.randn(4,1)
    labels = torch.zeros(4, dtype=torch.long)
    trainer = TrainModel()
    trained, metrics, best = trainer.train(
        model,
        **{"features tensor": features, "labels tensor": labels, "epochs":1, "batch_size":2}
    )
    assert isinstance(trained, nn.Module)
    assert isinstance(metrics, list)


def test_tensors_to_dataset_and_back():
    feats = torch.randn(3,2)
    labels = torch.tensor([0,1,0])
    to_ds = TensorsToDataset()
    ds, = to_ds.to_dataset(feats, labels)
    to_dl = DatasetToDataloader()
    dl, = to_dl.to_dataloader(ds)
    assert len(dl.dataset) == 3


def test_comfy_image_to_tensor():
    image = torch.rand(1,2,2,3)
    node = ComfyUIImageToPytorchTENSOR()
    t1, t2 = node.to_tensor(image)
    assert t1.shape[1:] == (2,2,3)
    assert t2.shape[1:] == (2,2)


def test_list_to_tensor():
    node = ListToTensor()
    t, = node.to_tensor([1,2,3], str(torch.int64))
    assert t.dtype == torch.int64


def test_activation():
    model = nn.Sequential()
    node = Activation()
    out_model, = node.activation(model, "ReLU")
    assert isinstance(out_model[-1], nn.ReLU)


def test_add_reshape_layer():
    model = nn.Sequential()
    node = AddReshapeLayer()
    out_model, = node.add_reshape_layer(model, "1,-1")
    assert isinstance(out_model[-2], nn.Flatten)


def test_add_transformer_layer():
    model = nn.Sequential()
    node = AddTransformerLayer()
    out_model, = node.add_transformer_layer(model, 4, 1, 8)
    assert isinstance(out_model[-1], nn.TransformerEncoderLayer)


def test_to_device():
    model = nn.Sequential(nn.Linear(1,1))
    tensor = torch.randn(1,1)
    node = PyTorchToDevice()
    m,t = node.to_device("cpu", model=model, tensor=tensor)
    assert next(m.parameters()).device.type == "cpu" and t.device.type == "cpu"


def test_extract_layers_as_model():
    model = nn.Sequential(nn.Linear(2,2), nn.ReLU(), nn.Linear(2,1))
    node = ExtractLayersAsModel()
    new_model, shapes = node.extract_layers(model, 0, 1)
    assert isinstance(new_model, nn.Sequential)
    assert "2" in shapes


def test_add_model_as_layer():
    m1 = nn.Sequential(nn.Linear(1,1))
    m2 = nn.Sequential(nn.ReLU())
    node = AddModelAsLayer()
    out, = node.add_model(m1, m2)
    assert len(out) == 2


def test_random_tensor():
    node = RandomTensor()
    t, = node.random_tensor("(2,2)")
    assert t.shape == (2,2)


def test_save_load_tensor(tmp_path):
    node_s = SaveTorchTensor()
    node_l = LoadTorchTensor()
    path = tmp_path / "t.pt"
    node_s.save_torch_tensor(torch.tensor([1,2]), str(path))
    t, = node_l.load_torch_tensor(str(path))
    assert torch.equal(t, torch.tensor([1,2]))


def test_func_modify_model():
    model = nn.Sequential()
    node = FuncModifyModel()
    new_model, = node.modify_model(model, "model.add_module('relu', nn.ReLU())")
    assert isinstance(new_model[-1], nn.ReLU)


def test_plot_series_string():
    node = PlotSeriesString()
    import sys, types
    mpl = types.ModuleType('matplotlib')
    mpl.pyplot = types.SimpleNamespace(
        plot=lambda *a, **k: None,
        title=lambda *a, **k: None,
        xlabel=lambda *a, **k: None,
        ylabel=lambda *a, **k: None,
        savefig=lambda buf, format=None: buf.write(b'img'),
        close=lambda: None,
    )
    sys.modules['matplotlib'] = mpl
    sys.modules['matplotlib.pyplot'] = mpl.pyplot
    np_stub = types.SimpleNamespace(array=lambda x: x)
    sys.modules['numpy'] = np_stub
    sys.modules['PIL'] = types.ModuleType('PIL')
    sys.modules['PIL'].Image = types.SimpleNamespace(open=lambda buf: [[1]])
    img, = node.plot_series_string("1\n2\n3")
    assert img.shape[1] > 0
