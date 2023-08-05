import os

os.environ["UNIT_TEST"] = "True"
from custom_nodes.EternalKernelLiteGraphNodes.youtube_nodes import YoutubeUploadNode


def test_upload():
    node = YoutubeUploadNode()
    output = node.upload_video(
        file_path=os.path.join("tests", "example.mp4"),
        category=22,
        title="Test Video",
        description="This is a test video for the Youtube Upload Node",
        keywords="test, youtube, node",
        privacy_status="unlisted"
    )
    print (output)
