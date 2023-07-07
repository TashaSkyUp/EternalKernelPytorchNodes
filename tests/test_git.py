import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
try:
    from custom_nodes.EternalKernelLiteGraphNodes.git import CreateRepoNode
except:
    from ..git import CreateRepoNode



class TestCreateRepoNode(unittest.TestCase):
    def test_create_repo(self):
        node = CreateRepoNode()
        node.create_repo(subdirectory="my_repo")
        # Add assertions or additional test logic here


if __name__ == "__main__":
    unittest.main()
