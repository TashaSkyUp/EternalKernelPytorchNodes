import pytest
from ..torchvision_nodes import TorchVisionTransformCompositionList
from ..torchvision_nodes import TorchVisionTransformNode
from ..torchvision_nodes import TorchVisionCallComposition


# when testing set "UNIT_TEST" to true in the environment to avoid loading all of ComfyUI


def test_TorchVisionTransformCompositionList():
    tvtcl_node = TorchVisionTransformCompositionList()
    ret = tvtcl_node.compose("test")
    ret = ret[0]
    assert ret is not None

def test_TorchVisionTransformNode():
    node1 = TorchVisionTransformCompositionList()
    node2 = TorchVisionTransformNode()

    ret = node1.compose("test")
    ret = ret[0]
    assert ret is not None

    ret = node2.compose(ret,"GaussianBlur","1")
    ret = ret[0]
    assert ret is not None

def test_TorchVisionCallComposition():
    node1 = TorchVisionTransformCompositionList()
    node2 = TorchVisionTransformNode()
    node3 = TorchVisionCallComposition()

    ret = node1.compose("test")
    ret = ret[0]
    assert ret is not None

    ret = node2.compose(ret,"GaussianBlur","1")
    ret = ret[0]
    assert ret is not None

    import torch
    test_tensor = torch.rand(1,256,256,3)
    ret = node3.compose(ret,test_tensor)
    ret = ret[0]
    assert ret is not None

if __name__ == "__main__":
    pytest.main()
