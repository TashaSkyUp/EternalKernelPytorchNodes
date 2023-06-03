import unittest
from PIL import Image
from custom_nodes.EternalKernelLiteGraphNodes.image import TextRender as Renderer
import os
hard_test = """Chuck Norris needs no mouse;

He stares down the computer

THE END.

SchwaaWeb, trained by Chuck!"""
hard_test2 ="""Chuck Norris can catch a data breach before it even happens.

SchwaaWeb, vigilant 

like Chuck!"""
# set an env var to tell so we know we are in test mode
os.environ["ETERNAL_KERNEL_LITEGRAPH_NODES_TEST"] = "True"

class TestRenderText(unittest.TestCase):
    def test_render_text(self):
        # Initialize object that includes _render_text and _wrap_text methods
        renderer = Renderer()

        # Create a blank image
        image = Image.new('RGB', (500, 500), color=(73, 109, 137))

        # Define parameters for _render_text
        font_name = 'arial'
        size = 24
        text = hard_test2
        x, y = 0, 0

        try:
            # Call the function under test
            renderer._render_text(font_name, image, size, text, x, y, allow_wrap=True, allow_shrink=True)

            # Check that the image is not empty (i.e., it has been changed by the drawing operation)
            self.assertNotEqual(image.tobytes(), Image.new('RGB', (500, 500), color=(73, 109, 137)).tobytes())
        finally:
            # Show the image whether the test passed or failed
            image.show()

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
