NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def ETK_3d_base(cls):
    import functools
    cls.CATEGORY = "ETK/3d"
    # Add spaces to the camel case class name
    pretty_name = cls.__name__
    for i in range(1, len(pretty_name)):
        if pretty_name[i].isupper():
            pretty_name = pretty_name[:i] + " " + pretty_name[i:]
    cls.DISPLAY_NAME = pretty_name
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    NODE_DISPLAY_NAME_MAPPINGS[cls.DISPLAY_NAME] = pretty_name
    return cls


@ETK_3d_base
class TripoSRNode:
    def __init__(self):
        import os
        from folder_paths import output_directory
        self.output_dir = output_directory
        self.repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repos", "img23d")
        self.repo_url = "https://github.com/VAST-AI-Research/TripoSR.git"
        self.venv_dir = os.path.join(self.repo_dir, "venv")
        self.python_executable = os.path.join(self.venv_dir, "Scripts", "python")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_dir": ("STRING", {"default": "outputs/TripoSR"}),
                "mc_resolution": ("INT", {"default": 256}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_3d_models"

    CATEGORY = "3D"

    def generate_3d_models(self, image, output_dir=None, mc_resolution=256):
        import asyncio

        if output_dir is not None:
            #self.output_dir = f"{self.output_dir}/{output_dir}"
            self.final_out_dir = f"{self.output_dir}/{output_dir}"

        # Create a new asyncio loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run the async function within the loop
        output_paths = loop.run_until_complete(self.async_generate_3d_models(image, mc_resolution))

        return (output_paths,)

    async def async_generate_3d_models(self, image, mc_resolution):
        import os
        import numpy as np
        from PIL import Image
        import asyncio

        # Check if the TripoSR repository is installed
        if not os.path.exists(self.repo_dir):
            # Clone the repository if it doesn't exist
            await self.clone_repository()
            # Create a virtual environment and install requirements
            await self.create_venv_and_install_requirements()

        # Create a temporary directory for saving input images
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        # Save each image in the batch as a separate file
        # each image is a torch tensor with shape (b,h,w,c)
        image_paths = []
        for i in range(image.shape[0]):
            image_np = image[i].cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            image_path = os.path.join(temp_dir, f"image_{i}.png")
            image_pil.save(image_path)
            image_paths.append(image_path)

        # Run inference using the TripoSR CLI for each image simultaneously
        output_paths = await asyncio.gather(*[self.run_inference(image_path,mc_resolution) for image_path in image_paths])

        # Clean up the temporary image files
        for image_path in image_paths:
            os.remove(image_path)

        return output_paths

    async def run_inference(self, image_path, mc_resolution):
        import asyncio
        import os
        # Create a virtual environment and install requirements if they don't exist
        command = [
            self.python_executable,
            os.path.join(self.repo_dir, "run.py"),
            image_path,
            "--mc-resolution",
            f"{mc_resolution}",
            "--output-dir",
            self.final_out_dir,
        ]
        process = await asyncio.create_subprocess_exec(*command)
        await process.wait()
        output_path = self.get_output_path(image_path)
        return output_path

    async def clone_repository(self):
        import asyncio
        # Clone the TripoSR repository
        command = ["git", "clone", self.repo_url, self.repo_dir]
        process = await asyncio.create_subprocess_exec(*command)
        await process.wait()

    async def create_venv_and_install_requirements(self):
        import asyncio
        import os
        # Create a virtual environment
        command = ["python", "-m", "venv", self.venv_dir]
        # use os to execute the command
        os.system(" ".join(command))

        # install pip "python -m ensurepip"
        command = [self.python_executable, "-m", "ensurepip"]
        os.system(" ".join(command))

        # install wheel
        command = [self.python_executable, "-m", "pip", "install", "wheel"]
        os.system(" ".join(command))

        # install torch
        # TODO: detect what version of torch to install based on what is installed on system
        command = [self.python_executable, "-m", "pip", "install", "torch",
                   "--index-url https://download.pytorch.org/whl/cu121"]

        #command = [self.python_executable, "-m", "pip", "install", "torch",
        #           "--index-url https://download.pytorch.org/whl/cu118"]

        os.system(" ".join(command))

        # Install requirements
        command = [self.python_executable, "-m", "pip", "install", "-r",
                   os.path.join(self.repo_dir, "requirements.txt")]
        os.system(" ".join(command))

    def get_output_path(self, image_path):
        import os
        # Extract the base name of the input image file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Construct the output path based on the base name and output directory
        output_path = os.path.join(self.output_dir, base_name, "obj", "mesh_norm.obj")
        return output_path
