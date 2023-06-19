from unittest import TestCase


class TestDecodeTranscriptNode(TestCase):
    def test_func(self):
        from custom_nodes.EternalKernelLiteGraphNodes.descript import DecodeTranscriptNode
        node = DecodeTranscriptNode()
        string = """[00:00:00] jackie: Let's talk about a

        [00:00:01] jackie: thing\n\n[00:00:05] jackie: Let's talk about an apple
        
        """
        speakers, durations, paragraphs = node.func(string)
        self.assertEqual(speakers, ["jackie", "jackie", "jackie"])
        self.assertEqual(durations, [1, 4, 2])
        self.assertEqual(paragraphs, ["Let's talk about a", 'thing', "Let's talk about an apple"])