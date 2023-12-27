import pytest
from ..hf_diffusers_nodes import DDPMPipline, SampleDDPMPipline, TensorToImage
from ..hf_diffusers_nodes import LoadHFDataset



# when testing set "UNIT_TEST" to true in the environment to avoid loading all of ComfyUI
def test_DDPMPipline():
    p_node = DDPMPipline()
    ret = p_node.load_from_pretrained("johnowhitaker/ddpm-butterflies-32px",
                                      "cuda")
    assert ret is not None


def test_SampleDDPMPipline():
    p_node = DDPMPipline()
    s_node = SampleDDPMPipline()
    ret = p_node.load_from_pretrained("johnowhitaker/ddpm-butterflies-32px",
                                      "cuda")
    ret=ret[0]
    assert ret is not None
    ret = s_node.sample(ret,
                        1,
                        25
                        )
    assert ret is not None


def test_TensorToImage():
    p_node = DDPMPipline()
    s_node = SampleDDPMPipline()
    t_node = TensorToImage()
    ret = p_node.load_from_pretrained("johnowhitaker/ddpm-butterflies-32px",
                                      "cuda")
    ret = ret[0]
    assert ret is not None
    ret = s_node.sample(ret,
                        1,
                        25
                        )
    ret = ret[0]
    assert ret is not None
    ret = t_node.tensor_to_image(ret,"HWC")
    ret=ret[0]
    assert ret is not None
    # now convert to pillow image
    from PIL import Image
    import numpy as np
    # ret is float32 0-1 torch tensor we need to convert it to numpy and then to a pillow image

    ret = ret.squeeze(0)
    ret = ret.squeeze(0)
    ret = ret.numpy()
    ret = ret*255
    ret = np.uint8(ret)
    ret = Image.fromarray(ret)

    ret.show()



def test_LoadHFDataset():
    p_node = LoadHFDataset()
    ret = p_node.load_from_hf("huggan/smithsonian_butterflies_subset")
    assert ret is not None


if __name__ == "__main__":
    pytest.main()
