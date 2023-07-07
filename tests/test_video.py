from unittest import TestCase
import os
import torch
os.environ["ETERNAL_KERNEL_LITEGRAPH_NODES_TEST"] = "True"

class TestImageStackToVideoFile(TestCase):
    def test_handler(self):
        # test with random frames of noise
        import numpy as np
        from custom_nodes.EternalKernelLiteGraphNodes.video import ImageStackToVideoFile

        # create a random stack of images
        stack = np.random.randint(0, 255, (100, 100, 100, 3), dtype=np.uint8)
        # it expects a tensor
        stack = torch.from_numpy(stack)
        # create the node
        node = ImageStackToVideoFile()
        # run the node
        node.handler(image_stack=stack, video_out="test.mp4")
        # check that the file exists
        # find the actual filepath for the generated file, will be ../../../video/test.mp4
        test_video_file_fullpath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "video",
            "test.mp4")

        self.assertTrue(os.path.exists(test_video_file_fullpath))
        # remove the file
        os.remove(test_video_file_fullpath)


class TestTemporalSpatialSmoothing(TestCase):
    def test_handler(self):
        # test with random frames of noise

        from custom_nodes.EternalKernelLiteGraphNodes.video import TemporalSpatialSmoothing, ImageStackToVideoFile
        from custom_nodes.EternalKernelLiteGraphNodes.image import torch_image_show

        # use only torch
        # create a random stack of images
        # use floats
        stack = torch.rand((48, 512, 512, 3))
        # _,path= ImageStackToVideoFile().handler(image_stack=stack, video_out="test_start.mp4",fps=24)
        # launch the default video player
        # os.system(f"start {path}")

        # create the node
        node = TemporalSpatialSmoothing()
        # run the node
        result = node.handler(image_stack=stack, spatial_kernel_size=3, temporal_kernel_size=3, spatial_sigma=.5,
                              temporal_sigma=.5)
        # check that the file exists
        _, path = ImageStackToVideoFile().handler(image_stack=result[0], video_out="test_result.mp4", fps=24)
        # launch the default video player
        os.system(f"start {path}")

        # torch_image_show(result[0])


class TestTemporalOpticalFlowSmoothing(TestCase):
    def test_handler(self):
        from custom_nodes.EternalKernelLiteGraphNodes.video import TemporalOpticalFlowSmoothing, ImageStackToVideoFile
        import time
        import numpy as np
        # create a random stack of images
        #stack = torch.rand((24*4, 512, 512, 3))
        h=256
        w=512
        fps = 24
        seconds = 2
        stack = torch.rand((fps * seconds, h, w, 3))
        # now add a box that moves from left to right with a constant speed
        # the box will be 20% of the width and 20% of the height
        # the box will move from left to right

        # create the box
        box = torch.zeros((fps * seconds, int(h), int(w), 3))
        # fill the box with ones
        box = box + 1
        # the box width
        bw = min(int(w * .2),int(h*.2))
        # the box height
        bh = int(h * .2)
        # the box x position

        t = stack.shape[0]
        for i in range(t):
            pt = i / t

            # the box x position
            bx = int((w - bw) * pt)
            # now add a jitter
            bx += int(np.random.randint(-10, 10, 1))
            # the box y position
            by = int(h * .333)
            # fill the box with ones
            box[i,:,:,:] = 0
            box[i, by:by + bh, bx:bx + bw, :] = 1

        # create a mask for the box region
        #box_mask = torch.zeros_like(stack)
        #box_mask[:, by:by + bh, bx:bx + bw, :] = 1

        # set the color of the stack to blue where the box mask is 1
        stack = torch.where(box == 1, torch.tensor([.24, .5, .7], dtype=torch.float32), stack)

        nm = "box1"
        _, path = ImageStackToVideoFile().handler(image_stack=stack, video_out=nm, fps=fps)
        assert os.path.exists(path), f"Output video file doesn't exist: {path}"
        os.system(f"start {path}")
        time.sleep(1)

        # create the node
        node = TemporalOpticalFlowSmoothing()
        # run the node
        result = node.handler(image_stack=stack, device="cuda")
        result = result[0]


        #result = torch.cat(result[0], dim=0)
        #print(result.shape)
        # now we can permute the dimensions
        #result = result.permute(0, 2,3, 1)


        _, path = ImageStackToVideoFile().handler(image_stack=result, video_out="test_result_optical_flow.mp4", fps=fps)
        assert os.path.exists(path), f"Output video file doesn't exist: {path}"
        os.system(f"start {path}")

        ## Check the shape of the output
        #assert len(result) == stack.shape[0] - 1

        # Each element in the result should have the shape of [1, 2, H, W]
        #for flow in result:
        #    assert flow.shape == torch.Size([1, 2, 512, 512])

        # check the file generation and video play


