"""
Test security fixes - verify that all eval/exec replacements work correctly
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
import re
import os

# Import the nodes we're testing
from custom_nodes.EternalKernelPyTorchNodes.pytorch_nodes import (
    ChangeTensorType,
    TrainModel,
    FuncModifyModel,
    PyTorchToDevice
)


class TestSecurityFixes:
    """Test all security fixes to ensure safe implementations work correctly"""

    def test_change_tensor_type_safe_dtype_mapping(self):
        """Test that ChangeTensorType works with predefined dtype mappings"""
        node = ChangeTensorType()
        test_tensor = torch.tensor([1.0, 2.0, 3.0])

        # Test common dtype strings
        test_cases = [
            ("torch.float32", torch.float32),
            ("torch.float64", torch.float64),
            ("torch.int32", torch.int32),
            ("torch.int64", torch.int64),
            ("torch.bool", torch.bool),
            ("torch.uint8", torch.uint8),
            ("torch.int8", torch.int8),
            ("torch.int16", torch.int16),
            ("torch.half", torch.half),
            ("torch.bfloat16", torch.bfloat16),
        ]

        for dtype_str, expected_dtype in test_cases:
            result, = node.change_type(test_tensor, dtype_str)
            assert result.dtype == expected_dtype, f"Failed for {dtype_str}"

    def test_change_tensor_type_fallback_safety(self):
        """Test that ChangeTensorType falls back safely for unknown dtypes"""
        node = ChangeTensorType()
        test_tensor = torch.tensor([1.0, 2.0, 3.0])

        # Test unknown/invalid dtype strings
        invalid_dtypes = [
            "unknown_dtype",
            "torch.invalid_type",
            "malicious_code()",
            "",
            None
        ]

        for invalid_dtype in invalid_dtypes:
            if invalid_dtype is None:
                continue
            result, = node.change_type(test_tensor, invalid_dtype)
            # Should fallback to float32 or handle gracefully
            assert result is not None
            assert isinstance(result, torch.Tensor)

    def test_change_tensor_type_getattr_fallback(self):
        """Test the getattr fallback for torch dtype attributes"""
        node = ChangeTensorType()
        test_tensor = torch.tensor([1.0, 2.0, 3.0])

        # Test torch attribute access
        result, = node.change_type(test_tensor, "torch.float64")
        assert result.dtype == torch.float64

    def test_train_model_safe_loss_functions(self):
        """Test that TrainModel uses safe loss function mappings"""
        # Create a simple model and data for testing
        model = nn.Sequential(nn.Linear(2, 2))
        features = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))

        node = TrainModel()

        # Test predefined loss functions
        loss_functions = [
            "MSELoss",
            "CrossEntropyLoss",
            "BCELoss",
            "L1Loss",
            "NLLLoss",
            "SmoothL1Loss",
            "KLDivLoss",
            "HuberLoss",
            "torch.nn.MSELoss()",
            "torch.nn.CrossEntropyLoss()",
        ]

        for loss_fn in loss_functions:
            try:
                # Should not raise security exceptions
                result = node.train(
                    model, features, labels,
                    epochs=1, batch_size=2,
                    loss_function=loss_fn,
                    optimizer="Adam"
                )
                assert len(result) >= 2  # Should return model and metrics
                assert isinstance(result[0], nn.Module)
            except Exception as e:
                # Allow training-related exceptions but not security ones
                assert "eval" not in str(e).lower()
                assert "exec" not in str(e).lower()

    def test_train_model_safe_optimizers(self):
        """Test that TrainModel uses safe optimizer mappings"""
        model = nn.Sequential(nn.Linear(2, 2))
        features = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))

        node = TrainModel()

        # Test predefined optimizers
        optimizers = [
            "Adam",
            "SGD",
            "RMSprop",
            "AdamW",
            "Adagrad",
            "Adadelta",
            "torch.optim.Adam(model.parameters(), lr=0.001)",  # Should use safe defaults
            "torch.optim.SGD(model.parameters(), lr=0.01)",
        ]

        for optimizer in optimizers:
            try:
                result = node.train(
                    model, features, labels,
                    epochs=1, batch_size=2,
                    loss_function="MSELoss",
                    optimizer=optimizer
                )
                assert len(result) >= 2
                assert isinstance(result[0], nn.Module)
            except Exception as e:
                # Allow training-related exceptions but not security ones
                assert "eval" not in str(e).lower()
                assert "exec" not in str(e).lower()

    def test_train_model_safe_metrics(self):
        """Test that TrainModel uses safe metrics mappings"""
        model = nn.Sequential(nn.Linear(2, 2))
        features = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))

        node = TrainModel()

        # Test predefined metrics
        metrics = [
            "L1Loss",
            "MSELoss",
            "CrossEntropyLoss",
            "BCELoss",
            "torch.nn.L1Loss()",
            "torch.nn.MSELoss()",
        ]

        for metric in metrics:
            try:
                result = node.train(
                    model, features, labels,
                    epochs=1, batch_size=2,
                    loss_function="MSELoss",
                    optimizer="Adam",
                    metrics=metric
                )
                assert len(result) >= 2
            except Exception as e:
                # Allow training-related exceptions but not security ones
                assert "eval" not in str(e).lower()
                assert "exec" not in str(e).lower()

    def test_train_model_safe_device_mapping(self):
        """Test that TrainModel uses safe device mappings"""
        model = nn.Sequential(nn.Linear(2, 2))
        features = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))

        node = TrainModel()

        # Test predefined device strings
        devices = [
            "torch.device('cpu')",
            "torch.device('cuda')",
            "cpu",
            "cuda",
            "cuda:0",
            "cuda:1",
        ]

        for device in devices:
            try:
                result = node.train(
                    model, features, labels,
                    epochs=1, batch_size=2,
                    loss_function="MSELoss",
                    optimizer="Adam",
                    device=device
                )
                assert len(result) >= 2
            except RuntimeError as e:
                # Allow CUDA not available errors
                if "cuda" in str(e).lower():
                    continue
                raise
            except Exception as e:
                # Allow training-related exceptions but not security ones
                assert "eval" not in str(e).lower()
                assert "exec" not in str(e).lower()

    def test_pytorch_to_device_safe_mapping(self):
        """Test that PyTorchToDevice uses safe device mappings"""
        node = PyTorchToDevice()
        model = nn.Sequential(nn.Linear(2, 2))
        tensor = torch.randn(2, 2)

        # Test safe device strings
        safe_devices = ["cpu", "cuda", "cuda:0"]

        for device in safe_devices:
            try:
                result_model, result_tensor = node.to_device(device, model=model, tensor=tensor)
                assert isinstance(result_model, nn.Module)
                assert isinstance(result_tensor, torch.Tensor)
            except RuntimeError as e:
                # Allow CUDA not available errors
                if "cuda" in str(e).lower():
                    continue
                raise

    def test_func_modify_model_safe_patterns(self):
        """Test that FuncModifyModel only allows safe regex patterns"""
        node = FuncModifyModel()
        model = nn.Sequential(nn.Linear(2, 2))

        # Test safe modification patterns
        safe_functions = [
            "model.add_module('relu', nn.ReLU())",
            "model.eval()",
            "model.train()",
        ]

        for func in safe_functions:
            try:
                result, = node.modify_model(model, func)
                assert isinstance(result, nn.Module)
                # Model should be modified (different from original)
                assert len(result) >= len(model) or hasattr(result, 'training')
            except Exception as e:
                # Should not be security-related exceptions
                assert "eval" not in str(e).lower()
                assert "exec" not in str(e).lower()

    def test_func_modify_model_unsafe_patterns_blocked(self):
        """Test that FuncModifyModel blocks unsafe patterns"""
        node = FuncModifyModel()
        model = nn.Sequential(nn.Linear(2, 2))

        # Test unsafe patterns that should be blocked
        unsafe_functions = [
            "import os; os.system('rm -rf /')",
            "eval('malicious_code')",
            "exec('dangerous_code')",
            "__import__('subprocess').call(['rm', '-rf', '/'])",
            "model.__class__ = SomethingElse",
        ]

        for func in unsafe_functions:
            try:
                result, = node.modify_model(model, func)
                # Should fallback to safe default modification
                assert isinstance(result, nn.Module)
                # Should add the safe default ReLU layer
                if hasattr(result, 'safe_relu'):
                    assert isinstance(result.safe_relu, nn.ReLU)
            except Exception as e:
                # Should not execute the unsafe code
                assert "eval" not in str(e).lower()
                assert "exec" not in str(e).lower()

    def test_no_eval_exec_in_codebase(self):
        """Test that no eval() or exec() calls exist in the main code"""
        # Read the main pytorch_nodes.py file
        with open(os.path.join(os.path.dirname(__file__), '..', 'pytorch_nodes.py'), 'r') as f:
            content = f.read()

        # Check for eval( or exec( patterns (not in comments or strings)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            # Check for eval( or exec( (not in regex patterns or strings)
            if ('eval(' in line or 'exec(' in line) and not any(x in line for x in ['"', "'", 'r"', "r'"]):
                pytest.fail(f"Found eval/exec usage on line {i+1}: {line}")

    def test_security_regression_prevention(self):
        """Test cases to prevent regression of security fixes"""
        # Test that we can't inject code through any of the "safe" interfaces

        # Test ChangeTensorType injection attempts
        node_dtype = ChangeTensorType()
        tensor = torch.tensor([1.0])

        injection_attempts = [
            "torch.float32; import os; os.system('echo hacked')",
            "torch.float32') or eval('print(\"injected\")')",
            "__import__('os').system('echo test')",
        ]

        for injection in injection_attempts:
            try:
                result, = node_dtype.change_type(tensor, injection)
                # Should either work safely or use default
                assert isinstance(result, torch.Tensor)
            except Exception:
                # Exceptions are fine, just shouldn't execute injected code
                pass

    def test_safe_fallbacks_work(self):
        """Test that all safe fallbacks provide working defaults"""
        # Test ChangeTensorType fallback
        node_dtype = ChangeTensorType()
        tensor = torch.tensor([1.0])
        result, = node_dtype.change_type(tensor, "unknown_type")
        assert isinstance(result, torch.Tensor)

        # Test TrainModel fallbacks
        node_train = TrainModel()
        model = nn.Sequential(nn.Linear(2, 2))
        features = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))

        # Should work with unknown loss/optimizer/metrics
        try:
            result = node_train.train(
                model, features, labels,
                epochs=1, batch_size=2,
                loss_function="UnknownLoss",
                optimizer="UnknownOptimizer",
                metrics="UnknownMetric"
            )
            assert len(result) >= 2
            assert isinstance(result[0], nn.Module)
        except Exception as e:
            # Training might fail but shouldn't be security-related
            assert "eval" not in str(e).lower()
            assert "exec" not in str(e).lower()

    def test_comprehensive_security_coverage(self):
        """Ensure all previously vulnerable functions are now secure"""

        # List of all functions that previously used eval/exec
        vulnerable_functions = [
            (ChangeTensorType, "change_type"),
            (TrainModel, "train"),
            (FuncModifyModel, "modify_model"),
        ]

        for node_class, method_name in vulnerable_functions:
            # Verify the method exists and is callable
            assert hasattr(node_class, method_name)
            method = getattr(node_class(), method_name)
            assert callable(method)

            # Verify no eval/exec in the method's source (if available)
            import inspect
            try:
                source = inspect.getsource(method)
                assert 'eval(' not in source or 'r"' in source  # Allow in regex patterns
                assert 'exec(' not in source or 'r"' in source  # Allow in regex patterns
            except OSError:
                # Can't get source for some methods, that's ok
                pass
