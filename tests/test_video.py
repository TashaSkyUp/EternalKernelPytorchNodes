from unittest import TestCase
import os
import torch


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
