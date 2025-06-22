import torch
import unittest
from custom_nodes.EternalKernelLiteGraphNodes.pytorch_nodes import TrainModel

class TestTrainModelResetOptimizer(unittest.TestCase):
    def test_reset_optimizer_weights_attr_error(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = TrainModel()
        with self.assertRaises(AttributeError):
            trainer.reset_optimizer_weights(optimizer)

if __name__ == "__main__":
    unittest.main()
