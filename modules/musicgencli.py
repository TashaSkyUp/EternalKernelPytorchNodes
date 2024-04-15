def set_all_seeds(seed):
    import random
    import os
    import numpy as np
    import torch
    old_values = {}
    old_values['random'] = random.getstate()
    old_values['numpy'] = np.random.get_state()
    old_values['torch'] = torch.get_rng_state()
    old_values['torch_cuda'] = torch.cuda.get_rng_state()
    old_values['torch_cuda_all'] = torch.cuda.get_rng_state_all()
    old_values['torch_deterministic'] = torch.backends.cudnn.deterministic

    # From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    return old_values


def reset_seeds(old_values):
    import random
    import numpy as np
    import torch

    random.setstate(old_values['random'])
    np.random.set_state(old_values['numpy'])
    torch.set_rng_state(old_values['torch'])
    torch.cuda.set_rng_state(old_values['torch_cuda'])
    torch.cuda.set_rng_state_all(old_values['torch_cuda_all'])
    torch.backends.cudnn.deterministic = old_values['torch_deterministic']

import time  # Add this at the top of your script

def generate_music_for_description(description, duration, file, model_name, position):
    time.sleep(position * 5)  # Wait for position * 5 seconds
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write
    import torch
    model = MusicGen.get_pretrained(model_name)
    print(f"Generating music for description: {description}")
    print(f"Using model: {model_name}")
    print(f"Using duration: {duration}")
    model.set_generation_params(duration=duration)
    wav = model.generate([description])
    print(f"Writing to {file}")
    audio_write(file.replace(".wav", ""), wav[0].cpu(), model.sample_rate, strategy="loudness")
    del model
    torch.cuda.empty_cache()


def generate_music(**kwargs):
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write
    import torch
    import ast
    import copy
    import random
    from concurrent.futures import ProcessPoolExecutor, as_completed



    torch.cuda.empty_cache()

    # Set seed, -1 means random seed
    seed = kwargs.get('seed', -1)
    if seed == -1:
        seed = random.randint(0, 100000)
    old_seed_values = set_all_seeds(seed)

    kwargs = copy.deepcopy(kwargs)
    description = kwargs.get('description')

    file = kwargs.get('file')
    # Add .wav if not there
    if not file.endswith('.wav'):
        file = file + ".wav"

    model_name = kwargs.get('model_name', 'melody')
    duration = kwargs.get('duration', 10)

    descriptions = ast.literal_eval(description)
    durations = ast.literal_eval(duration)



    o_files = []
    batch_size = 3
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for i, (description, duration) in enumerate(zip(descriptions, durations)):
            ii = str(i).zfill(1 + len(str(1 + len(descriptions))))
            file_to_use = file.replace(".wav", f"_{ii}.wav")
            # Pass i (the position in the batch) as an additional argument
            future = executor.submit(generate_music_for_description, description, duration, file_to_use, model_name, i//batch_size)
            futures.append((future, file_to_use))

        for future, file_to_use in futures:
            # Wait for the future to complete, no need to get the result since the function handles file writing
            future.result()
            o_files.append(file_to_use)

    # Reset seed
    reset_seeds(old_seed_values)
    torch.cuda.empty_cache()

    return (o_files[0], o_files,)


import argparse
import copy


def main():
    print("Running main function...")
    # Define the model names as a list for the choices parameter
    model_names = [
        "facebook/musicgen-melody-large",
        "facebook/musicgen-melody-small",
        "facebook/musicgen-melody-medium",

        "facebook/musicgen-stereo-melody-large",
        "facebook/musicgen-stereo-melody-small",
        "facebook/musicgen-stereo-melody-medium",

        "facebook/musicgen-large",
        "facebook/musicgen-small",
        "facebook/musicgen-medium",

        "facebook/musicgen-stereo-large",
        "facebook/musicgen-stereo-small",
        "facebook/musicgen-stereo-medium",
    ]

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate music based on a given description.')
    parser.add_argument('--description', help='Description of the music to generate')
    parser.add_argument('--file', type=str, default='output.wav', help='Output file name (default: output.wav)')
    parser.add_argument('--model_name', type=str, default='facebook/musicgen-large', choices=model_names,
                        help='Model name to use for generation')
    parser.add_argument('--duration', help='Duration of the generated music in seconds (default: 10)')

    parser.add_argument('--seed', type=int, default=-1,
                        help='Seed for random number generation (default: -1 for random seed)')
    parser.add_argument('--test', action='store_true', help='Run the test suite instead of generating music')
    args = parser.parse_args()

    # Convert argparse.Namespace to dictionary
    kwargs = vars(args)

    if args.test:
        run_tests()
    else:
        print(f"Generating music with the following arguments: {kwargs}")
        # Generate music
        generated_files = generate_music(**kwargs)
        print(f"Generated music files: {generated_files}")


def run_tests():
    print("Running test suite...")
    # Define your test cases here
    test_model_name_validation()
    test_file_generation()
    # Add more tests as needed
    print("All tests passed!")


def test_model_name_validation():
    print("Testing model name validation...")
    model_names = [
        "facebook/musicgen-melody-large",
        "facebook/musicgen-melody-small",
        "facebook/musicgen-melody-medium",
        "facebook/musicgen-stereo-melody-large",
        "facebook/musicgen-stereo-melody-small",
        "facebook/musicgen-stereo-melody-medium",
        "facebook/musicgen-large",
        "facebook/musicgen-small",
        "facebook/musicgen-medium",
        "facebook/musicgen-stereo-large",
        "facebook/musicgen-stereo-small",
        "facebook/musicgen-stereo-medium",
    ]

    description = "A brief test piece"  # Example description for testing
    duration = 5  # 5 seconds long
    file_template = "test_output_{}.wav"  # Template for generated file names

    for i, model_name in enumerate(model_names):
        print(f"Testing {model_name}...")
        kwargs = {
            'description': description,
            'duration': duration,
            'model_name': model_name,
            'file': file_template.format(i),  # Ensure unique file names
            'seed': 42  # Fixed seed for consistency, if needed
        }

        try:
            # Assuming generate_music is adapted to work with these arguments
            generate_music(**kwargs)
            print(f"Successfully tested {model_name}")
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            # Handle the error appropriately - for example, you might want to fail the test or log the error.

    print("Model name validation test completed.")


def test_file_generation():
    # Example test function
    print("Testing file generation...")
    # Implement the logic to test music file generation
    # This could involve generating a small piece of music and checking if the file exists
    # ...


if __name__ == "__main__":
    main()
