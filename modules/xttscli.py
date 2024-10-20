import sys
import os
import argparse
import logging
import traceback
import torch
import torchaudio
import tempfile
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

XTTS_MODEL = None


def prepare_speaker_audio(speaker):
    """
    Prepares the speaker audio file and returns the speaker file path.
    For .wav and .mp3 files, it directly returns the file path for model processing.
    """
    if speaker.endswith(".wav") or speaker.endswith(".mp3"):
        print(f"Using speaker audio file: {speaker}")
        return speaker, None  # No precomputed speaker embedding
    elif speaker.endswith(".pth"):
        print(f"Loading preprocessed speaker embedding from .pth file: {speaker}")
        speaker_embedding = torch.load(speaker)
        return None, speaker_embedding  # Return precomputed speaker embedding
    else:
        raise ValueError("Unsupported speaker file format. Please provide a .wav, .mp3, or .pth file.")


def run_tts(lang, tts_text, speaker_audio_file, speaker_embedding, device):
    """
    Generates TTS using the XTTS model, either from speaker audio or precomputed speaker embedding.
    """
    if XTTS_MODEL is None:
        raise ValueError("You need to load the model.")

    # Generate conditioning latents from speaker audio if provided
    if speaker_audio_file is not None:
        print(f"Using speaker audio file: {speaker_audio_file}")
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs
        )
    elif speaker_embedding is not None:
        raise ValueError("Precomputed speaker embeddings are not directly usable without gpt_cond_latent.")

    # Perform inference
    out = XTTS_MODEL.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=XTTS_MODEL.config.temperature,
        length_penalty=XTTS_MODEL.config.length_penalty,
        repetition_penalty=XTTS_MODEL.config.repetition_penalty,
        top_k=XTTS_MODEL.config.top_k,
        top_p=XTTS_MODEL.config.top_p,
        enable_text_splitting=True
    )

    # Move the output tensor to CPU before saving
    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0).to("cpu")  # Ensure tensor is on CPU

    # Save the generated speech to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)  # Save the tensor to disk

    return out_path


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(xtts_checkpoint, xtts_config, xtts_vocab, device):
    """
    Loads the XTTS model from checkpoint, config, and vocab files.
    """
    global XTTS_MODEL
    clear_gpu_cache()

    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        raise ValueError("You need to set the XTTS checkpoint path, XTTS config path, and XTTS vocab path!")

    config = XttsConfig()
    config.load_json(xtts_config)

    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model!")

    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)

    if "cuda" in device and torch.cuda.is_available():
        XTTS_MODEL.cuda(device)
        print(f"Model moved to GPU: {device}")
    else:
        print(f"Using device: {device}")


def main():
    # Set up logging
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xtts_cli.log')
    logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')

    # Default paths for the config, checkpoint, and vocab
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, 'config.json')
    default_checkpoint_path = os.path.join(script_dir, 'model.pth')
    default_vocab_path = os.path.join(script_dir, 'vocab.json')
    default_speaker_path = os.path.join(script_dir, 'speaker.mp3')  # Default speaker audio file
    default_test_file_path = os.path.join(script_dir, 'test.speech')  # Default text file for --test

    parser = argparse.ArgumentParser(description="CLI for XTTS model loading and TTS generation.")
    parser.add_argument('--checkpoint', default=default_checkpoint_path,
                        help=f"Path to XTTS checkpoint file. Default: {default_checkpoint_path}")
    parser.add_argument('--config', default=default_config_path,
                        help=f"Path to XTTS config file. Default: {default_config_path}")
    parser.add_argument('--vocab', default=default_vocab_path,
                        help=f"Path to XTTS vocab file. Default: {default_vocab_path}")
    parser.add_argument('--speaker', default=default_speaker_path,
                        help=f"Path to speaker audio file (raw .wav, .mp3, or preprocessed .pth). Default: {default_speaker_path}")
    parser.add_argument('--text', help="Text to synthesize.")
    parser.add_argument('--file', help="File containing text to synthesize (one text per line).")
    parser.add_argument('--lang', default='en', help="Language for the TTS.")
    parser.add_argument('--output', help="Path to save the output TTS file.")
    parser.add_argument('--device', default='cuda:0', help="Device to run the model on. Default: cuda:0")
    parser.add_argument('--test', action='store_true', help="Run a test TTS using the test.speech file.")

    args = parser.parse_args()

    try:
        # Preprocess the speaker file if it's a .wav or .mp3 file
        args.speaker, speaker_embedding = prepare_speaker_audio(args.speaker)

        # Handle test case
        if args.test:
            print("Running test with default parameters...")
            if not os.path.exists(default_test_file_path):
                raise FileNotFoundError(f"The test file '{default_test_file_path}' does not exist. Please create it.")

            args.file = default_test_file_path
            args.output = os.path.join(script_dir, 'test_output.wav')
            print(f"Test file: {args.file}")
            print(f"Output will be saved to: {args.output}")

        # Handle the --file argument
        if args.file:
            print(f"Reading text from file: {args.file}")
            with open(args.file, 'r') as f:
                args.text = f.read().splitlines()

        elif not args.text:
            raise ValueError("Text or a file with text must be provided unless running in test mode.")

        # Load the XTTS model
        load_model(args.checkpoint, args.config, args.vocab, args.device)

        # Process multiple lines of text (if from file)
        if isinstance(args.text, list):
            for i, tts_text in enumerate(args.text):
                output_path = run_tts(
                    lang=args.lang,
                    tts_text=tts_text,
                    speaker_audio_file=args.speaker,
                    speaker_embedding=speaker_embedding,
                    device=args.device
                )

                output_file = f"{args.output}_{i}.wav" if args.output else f'./output_{i}.wav'
                try:
                    os.rename(output_path, output_file)
                except Exception as e:
                    import shutil
                    # try copy then delete
                    shutil.copy(output_path, output_file)
                    os.remove(output_path)

                print(f"TTS generation complete for line {i + 1}. Output saved to {output_file}")
        else:
            # Generate TTS for a single text input
            output_path = run_tts(
                lang=args.lang,
                tts_text=args.text,
                speaker_audio_file=args.speaker,
                speaker_embedding=speaker_embedding,
                device=args.device
            )

            output_file = args.output if args.output else './output.wav'
            try:
                os.rename(output_path, output_file)
            except Exception as e:
                import shutil
                # try copy then delete
                shutil.copy(output_path, output_file)
                os.remove(output_path)

            print(f"TTS generation complete. Output saved to {output_file}")

    except Exception as e:
        logging.error("An error occurred: %s", traceback.format_exc())
        print(f"An error occurred. Check the log file at {log_file} for details.")


if __name__ == '__main__':
    # For testing
    sys.argv = [sys.argv[0], '--test']
    main()
