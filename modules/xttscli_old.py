import argparse
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import unittest
from unittest.mock import patch, MagicMock


def generate_xtts_for_text(text, file, lang, speaker):
    from TTS.api import TTS
    import torch
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker,
        language=lang,
        file_path=file,
    )


def generate_xtts(**kwargs):
    import torch
    import ast
    import uuid
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    torch.cuda.empty_cache()

    text = kwargs.get('text')
    folder = kwargs.get('folder', '.')
    folder = os.path.abspath(folder)
    lang = kwargs.get('lang', 'en')
    speaker = kwargs.get('speaker', '')

    if isinstance(text, str):
        if text != None and text.startswith("[") and text.endswith("]"):
            try:
                texts = ast.literal_eval(text)
            except SyntaxError:
                texts = [text]
    elif isinstance(text, list):
        texts = text

    o_files = []
    batch_size = 4
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for i, text in enumerate(texts):
            if len(text) > 1:
                ii = str(i).zfill(1 + len(str(1 + len(texts))))
                file_to_use = f"{folder}/tts_{ii}_{str(uuid.uuid4())}.wav"
                future = executor.submit(generate_xtts_for_text, text, file_to_use, lang, speaker)
                futures.append((future, file_to_use))

        for future, file_to_use in futures:
            future.result()
            o_files.append(file_to_use)

    torch.cuda.empty_cache()

    return (o_files[0], o_files,)


def cache_parameters(cache_file, params):
    """Cache the input parameters to a JSON file."""
    import json
    with open(cache_file, 'w') as f:
        json.dump(params, f)


def load_cached_parameters(cache_file):
    """Load the cached parameters from the JSON file."""
    import json
    import os
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None


def run_test():
    """Run unit tests for the module."""
    unittest.main(argv=[''], exit=False)


class TestGenerateXTTS(unittest.TestCase):
    """Unit tests for generate_xtts functions."""

    @patch('os.path.exists', return_value=True)
    @patch('TTS.api.TTS.tts_to_file')
    def test_generate_xtts_single_text(self, mock_tts_to_file, mock_exists):
        """Test generating xtts with a single text input."""
        mock_tts_to_file.return_value = None
        kwargs = {
            'text': 'Hello, this is a test.',
            'folder': './output',
            'lang': 'en',
            'speaker': '/path/to/speaker.wav'
        }

        result = generate_xtts(**kwargs)

        # Assertions
        self.assertEqual(len(result[1]), 1)
        mock_tts_to_file.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch('TTS.api.TTS.tts_to_file')
    def test_generate_xtts_multiple_texts(self, mock_tts_to_file, mock_exists):
        """Test generating xtts with multiple texts."""
        mock_tts_to_file.return_value = None
        kwargs = {
            'text': ['Hello, this is a test.', 'This is another test.'],
            'folder': './output',
            'lang': 'en',
            'speaker': '/path/to/speaker.wav'
        }

        result = generate_xtts(**kwargs)

        # Assertions
        self.assertEqual(len(result[1]), 2)
        self.assertEqual(mock_tts_to_file.call_count, 2)

    def test_cache_parameters(self):
        """Test caching parameters to a file."""
        import os
        import json
        test_cache_file = './test_cache.json'
        test_params = {'text': 'sample text', 'lang': 'en'}

        cache_parameters(test_cache_file, test_params)

        # Verify file content
        with open(test_cache_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(data, test_params)

        # Clean up
        os.remove(test_cache_file)

    def test_load_cached_parameters(self):
        """Test loading cached parameters from a file."""
        import os
        import json
        test_cache_file = './test_cache.json'
        test_params = {'text': 'sample text', 'lang': 'en'}

        # Write test data to the file
        with open(test_cache_file, 'w') as f:
            json.dump(test_params, f)

        # Load the parameters
        loaded_params = load_cached_parameters(test_cache_file)
        self.assertEqual(loaded_params, test_params)

        # Clean up
        os.remove(test_cache_file)


def main():
    import sys
    import os
    import argparse
    import logging
    import traceback

    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xttscli.log')
    logging.basicConfig(filename=log_file, level=logging.ERROR,
                        format='%(asctime)s %(levelname)s %(message)s')

    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params_cache.json')

    try:
        parser = argparse.ArgumentParser(description='Generate from given text.')
        parser.add_argument('--text', help='Text to generate for')
        parser.add_argument('--folder', type=str, default='.', help='Output folder')
        parser.add_argument('--lang', type=str, default='en')
        parser.add_argument('--speaker', type=str, default='', help='Speaker wav to use for generation')
        parser.add_argument('--file', type=str, default="", help='Newline separated file.')
        parser.add_argument('--rerun', action='store_true', help='Rerun with the last cached parameters.')
        parser.add_argument('--test', action='store_true', help='Run the test.')

        args = parser.parse_args()
        kwargs = vars(args)

        if args.test:
            run_test()
            sys.exit(0)

        if args.rerun:
            cached_params = load_cached_parameters(cache_file)
            if cached_params:
                print(f"Rerunning with cached parameters: {cached_params}")
                kwargs.update(cached_params)
            else:
                print("No cached parameters found. Exiting.")
                sys.exit(1)
        else:
            cache_parameters(cache_file, kwargs)

        if kwargs['file']:
            with open(kwargs['file']) as f:
                kwargs['text'] = f.read().split('\n')

        if kwargs['speaker']:
            if not os.path.exists(kwargs['speaker']):
                raise ValueError(f"Speaker file {kwargs['speaker']} does not exist.")

        generated_files = generate_xtts(**kwargs)
        print(f"Generated files: {generated_files}")

    except Exception as e:
        logging.error("An error occurred: %s", traceback.format_exc())
        print(f"An error occurred. Check the log file at {log_file} for details.")


if __name__ == '__main__':
    main()
