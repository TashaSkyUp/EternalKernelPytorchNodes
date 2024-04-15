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
    from concurrent.futures import ProcessPoolExecutor, as_completed

    torch.cuda.empty_cache()

    text = kwargs.get('text')
    folder = kwargs.get('folder', '.')
    lang = kwargs.get('lang', 'en')
    speaker = kwargs.get('speaker', '')

    try:
        texts = ast.literal_eval(text)
    except SyntaxError:
        texts = [text]

    o_files = []
    batch_size = 3
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for i, text in enumerate(texts):
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








def main():
    parser = argparse.ArgumentParser(description='Generate  from given text.')
    parser.add_argument('--text', help='Text to generate  for')
    parser.add_argument('--folder', type=str, default='.', help='output folder')
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--speaker', type=str, default='', help='Speaker wav use for generation')

    args = parser.parse_args()
    kwargs = vars(args)

    print(f"Generating with the following arguments: {kwargs}")
    generated_files = generate_xtts(**kwargs)
    print(f"Generated files: {generated_files}")

if __name__ == "__main__":
    main()