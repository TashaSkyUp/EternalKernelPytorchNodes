import unittest
from custom_nodes.git import CreateRepoNode

class TestCreateRepoNode(unittest.TestCase):
    def test_create_repo(self):
        node = CreateRepoNode()
        node.create_repo(subdirectory="my_repo")
        # Add assertions or additional test logic here

if __name__ == "__main__":
    unittest.main()
