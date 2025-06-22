# EternalKernel PyTorch Nodes

A specialized collection of PyTorch nodes for ComfyUI, focused on machine learning and neural network operations.

## Overview

This repository contains PyTorch-specific nodes that were extracted from the main EternalKernelLiteGraphNodes project. It provides a clean, focused collection of PyTorch functionality for ComfyUI workflows.

## Features

- **PyTorch Integration**: Native PyTorch tensor operations and model handling
- **Neural Network Nodes**: Various neural network layers and operations  
- **Training Support**: Nodes for model training and optimization
- **GPU Acceleration**: CUDA support for high-performance computing

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone <your-repo-url> EternalKernelPyTorchNodes
```

2. Install the required dependencies:
```bash
cd EternalKernelPyTorchNodes
pip install -r requirements.txt
```

## Contents

- `pytorch_nodes.py` - Main PyTorch node implementations
- `tests/` - Comprehensive test suite for all nodes
- `config.py` - Configuration settings

## Requirements  

- Python 3.8+
- PyTorch
- ComfyUI
- Additional dependencies listed in requirements.txt

## License

See LICENSE file for details.
