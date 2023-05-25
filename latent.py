#from nodes import MAX_RESOLUTION
import torch

from custom_nodes.EternalKernelLiteGraphNodes.components import fields as field


class LatentInterpolation:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent1": ("LATENT",),
                "latent2": ("LATENT",),
                "t": field.FLOAT,
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "interpolate"
    CATEGORY = "ETK/Latent"

    def interpolate(self, latent1, latent2, t):
        # Ensure that the latents are dictionaries
        if not (isinstance(latent1, dict) and isinstance(latent2, dict)):
            raise ValueError("Latent values must be dictionaries.")

        # Ensure that both latents have the same keys
        if set(latent1.keys()) != set(latent2.keys()):
            raise ValueError("Latent dictionaries must have the same keys.")

        inter_latent = {}
        for key in latent1:
            if isinstance(latent1[key], (list, tuple)):
                inter_latent[key] = [(1 - t) * l1_val + t * l2_val for l1_val, l2_val in
                                     zip(latent1[key], latent2[key])]
            elif isinstance(latent1[key], torch.Tensor):
                inter_latent[key] = (1 - t) * latent1[key] + t * latent2[key]
            else:
                raise ValueError(f"Unsupported value type for key '{key}' in latent dictionaries.")

        return (inter_latent,)


def test_latent_interpolation():
    import torch
    # Create an instance of the LatentInterpolation class
    li = LatentInterpolation()

    # Define test cases for the interpolate method
    # latents are at least 4 diminsional
    # latents are of the same length
    test_cases = [
        {
            "input": {
                "latent1": torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
                "latent2": torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]]),
                "t": 0.5,

            },
            "expected": torch.tensor([[[[3.0, 4.0], [5.0, 6.0]]]]),
        },
    ]

    # Run the test cases
    for test_case in test_cases:
        result = li.interpolate(**test_case["input"])
        if not (result[0][0] == test_case["expected"]).all().item():
            print(f"Expected {test_case['expected']} but got {result}")




if __name__ == "__main__":
    test_latent_interpolation()