import pyfastnoisesimd as fns


def get_perlin_2d(width, height, scale, octaves=6, lacunarity=3, gain=.5, seed=0, offset=0, workers=4) -> any:
    """
    Returns a 2D fractal perlin noise array of size (width, height) with the given parameters.
    """

    noise = fns.Noise(seed=seed, numWorkers=workers)
    noise.frequency = scale
    # noise.fractal.fractalType = fns.FractalType.RigidMulti
    noise.fractal.octaves = octaves
    noise.fractal.lacunarity = lacunarity
    noise.fractal.gain = gain
    noise.noiseType = fns.NoiseType.PerlinFractal
    noise.fractal.offset = offset

    return noise.genAsGrid([width, height])


def get_perlin_2d_as_torch_image_stack(width, height, scale, octaves=6, lacunarity=3, gain=.5, seed=0, offset=0,
                                       workers=4, norm_zero_to_1=True) -> any:
    """
    just uses get_perlin_2d and converts it to a torch image stack

    (B,H,W,C=3) format
    """

    import torch
    import numpy as np

    noise = get_perlin_2d(width, height, scale, octaves, lacunarity, gain, seed, offset, workers)
    # noise = np.transpose(noise, (2, 0, 1))
    noise = torch.from_numpy(noise)
    # noise = noise.unsqueeze(0)
    noise = torch.reshape(noise, (1, height, width, 1))
    # now stack it 3 times
    # noise = torch.normal(noise)
    # use torch to put it from 0-1
    # noise = torch.sigmoid(noise)



    # normalize it using torch

    if norm_zero_to_1:
        # noise = noise.view(noise.size(0), -1)
        noise -= noise.min()
        noise /= noise.max()

        # noise = noise + 0.5
        # noise = noise / 1.0

    return noise


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    for i in range(10):
        start = time.time()
        # noise = get_perlin_2d(1024, 1024, 0.001, i + 1, 3, .5, 1, 0)
        # noise = get_perlin_2d(1024*10, 1024*10, i/5, i + 1, 3, .5, i, 0)
        # print(type(noise))
        # print(f"Time taken: {time.time() - start}")
        # print the min and max
        # print(f"Min: {noise.min()} Max: {noise.max()}")
        # plt.imshow(noise, cmap='gray')
        # plt.show()

    start = time.time()
    for i in range(10):
        # now test the torch version
        noise = get_perlin_2d_as_torch_image_stack(1024,
                                                   1024,
                                                   0.001,
                                                   6,
                                                   3,
                                                   .5,
                                                   1,
                                                   0,
                                                   4,
                                                   2)

    print(f"Time taken: {time.time() - start}")
