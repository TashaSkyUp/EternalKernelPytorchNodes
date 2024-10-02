import argparse
import torch.multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, as_completed


def generate_xtts_for_text(text, file, lang, speaker):
    from TTS.api import TTS
    import torch
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker,
        language=lang,
        file_path=file,
        # emotion="angry", # doesnt do anything as far as i can tell with xtts2
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
    # make the folder absolute
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
            if len(text)>1:
                ii = str(i).zfill(1 + len(str(1 + len(texts))))
                file_to_use = f"{folder}/tts_{ii}_{str(uuid.uuid4())}.wav"
                future = executor.submit(generate_xtts_for_text, text, file_to_use, lang, speaker)
                futures.append((future, file_to_use))

        for future, file_to_use in futures:
            future.result()
            o_files.append(file_to_use)

    torch.cuda.empty_cache()

    return (o_files[0], o_files,)


# def generate_xtts_for_text_advanced(text, file, lang, speaker):
#     from TTS.api import TTS
#     import torch
#
#     # Create a SharedMemoryManager
#     manager = mp.Manager()
#     # Create a shared list to store the result
#     result = manager.list()
#     model = manager.model()
#
#
#     # also we want to SHARE the TTS model
#     something = manager.


import os
import argparse
import logging
import traceback

def main():
    # Set up logging to a file
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xttscli.log')
    logging.basicConfig(filename=log_file, level=logging.ERROR,
                        format='%(asctime)s %(levelname)s %(message)s')

    try:
        parser = argparse.ArgumentParser(description='Generate from given text.')
        parser.add_argument('--text', help='Text to generate for')
        parser.add_argument('--folder', type=str, default='.', help='Output folder')
        parser.add_argument('--lang', type=str, default='en')
        parser.add_argument('--speaker', type=str, default='', help='Speaker wav to use for generation')
        parser.add_argument('--file', type=str, default="", help='Newline separated file.')

        args = parser.parse_args()
        kwargs = vars(args)

        # If file is set, open it and read the content
        if kwargs['file']:
            with open(kwargs['file']) as f:
                kwargs['text'] = f.read().split('\n')

        # Ensure the speaker file exists
        if kwargs['speaker']:
            if not os.path.exists(kwargs['speaker']):
                raise ValueError(f"Speaker file {kwargs['speaker']} does not exist.")

        # Call your generate function
        generated_files = generate_xtts(**kwargs)
        print(f"Generated files: {generated_files}")

    except Exception as e:
        # Log the full exception details including the stack trace
        logging.error("An error occurred: %s", traceback.format_exc())
        print(f"An error occurred. Check the log file at {log_file} for details.")

if __name__ == '__main__':
    main()



if __name__ == "__main__":
    main()
