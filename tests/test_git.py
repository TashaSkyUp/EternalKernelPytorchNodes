import os
import unittest
from custom_nodes.git import CreateRepoNode

# Set the working directory to E:\git\ComfyUI\
os.chdir("E:/git/ComfyUI/")

class TestCreateRepoNode(unittest.TestCase):
    def test_create_repo(self):
        node = CreateRepoNode()
        node.create_repo(subdirectory="my_repo")
        # Add assertions or additional test logic here

if __name__ == "__main__":
    unittest.main()
