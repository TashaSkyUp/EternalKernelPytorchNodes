# Runtime errors in PyTorch nodes

The following problems were discovered:

1. **Undefined helper method and variable** (#2)
   - `TrainModel.train` calls `self.make_image_tensor()` which is not defined.
   - Variable `best_model` may be used before assignment.
2. **Dataset handling in inference** (#3)
   - `PyTorchInferenceNode.inference` assumes input dataset has a `.dataset` attribute, raising `AttributeError` for regular datasets.
3. **FlattenDataset assumes `.data`** (#4)
   - `FlattenDataset.flatten` uses `dataset.data` which is not present in generic datasets.

These should be addressed to prevent runtime failures.
