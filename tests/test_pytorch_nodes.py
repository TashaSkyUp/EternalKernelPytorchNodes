import unittest
from torch import nn

from ..pytorch_nodes import PyTorchDatasetDownloader, AddLinearLayerNode, SequentialModelProvider, PyTorchInferenceNode
class TestPyTorchNodes(unittest.TestCase):
    def test_dataset_downloader(self):
        downloader = PyTorchDatasetDownloader()
        dataset_name = "mnist"
        download_path = "./data/tmp"
        result = downloader.download_dataset(dataset_name, download_path)
        self.assertIsNotNone(result)

    def test_linear_layer_node(self):
        # Prepare a toy model for testing
        toy_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

        linear_adder = AddLinearLayerNode()
        in_features = 2
        out_features = 5
        result_model, = linear_adder.add_linear_layer(toy_model, in_features, out_features)
        # Perform assertions to verify the modification to the model

    def test_sequential_model_provider(self):
        provider = SequentialModelProvider()
        input_channels = 3
        output_classes = 10
        result_model, = provider.provide_default_model(input_channels, output_classes)
        # Perform assertions to verify the default architecture of the created model

    def test_inference_node(self):
        toy_data = torch.rand(1, 3, 224, 224)  # Toy data for inference
        toy_model = torchvision.models.resnet18(pretrained=False)  # Toy model for inference
        inference_node = PyTorchInferenceNode()
        result, = inference_node.inference(toy_model, toy_data)
        # Perform assertions to validate the inference operation


if __name__ == "__main__":
    unittest.main()
