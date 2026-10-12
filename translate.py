import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

load_dotenv()
client = OpenAI()


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
)
def noop():
    pass


def translate_text_file(
    input_folder, output_folder, filepath, options
):
    input_filepath = os.path.join(input_folder, filepath)
    output_filepath = os.path.join(output_folder, filepath)

    if Path(output_filepath).exists() and options.skip_existing:
        print(f"File {output_filepath} already exists, skipping.")
        return

    # Build containing folders to the output file location.
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    translate_instructions = f"""
    You are a technical translator, with a good knowledge of Python programming.
    You translate provided Asciidoctor documents from {options.from_language} to {options.to_language} while
    preserving meaning, tone and formatting. Write in idomatic, modern {options.to_language}. Be mindful of technical terms and jargon which are not commonly
    translated. Do not translate ANY code blocks, leave them as-is. Just return the translated text, no discussion of it.
    """

    with open(input_filepath, "r", encoding="utf-8") as f:
        text_to_translate = f.read()

    try:
        response = client.responses.create(
            model=options.model,
            instructions=translate_instructions,
            input=text_to_translate,
            temperature=0.3,
        )
    except BadRequestError as e:
        print(f"ERROR: {input_filepath}", e)
        # Ignore files on error, probably empty.
        translated_text = text_to_translate
    else:
        translated_text = response.output_text

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(translated_text)


def translate(input_folder, output_folder, options):
    futures = []
    with ThreadPoolExecutor(
        max_workers=options.num_threads
    ) as executor:
        # Find all asciidoc files in input_folder, and submit to worked to execute.
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if not file.endswith(".adoc"):
                    continue

                filepath = os.path.join(root, file)
                filepath = os.path.relpath(
                    filepath, start=input_folder
                )

                future = executor.submit(
                    translate_text_file,
                    input_folder,
                    output_folder,
                    filepath,
                    options,
                )
                futures.append(future)

        # Iterate over the futures as they complete.
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Translating files",
            unit="files",
        ):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred during translation: {e}")
                raise e

    print("All files translated.")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()

    arguments = [
        (
            "input_folder",
            {
                "type": str,
                "help": "input folder to translate",
            },
        ),
        (
            "output_folder",
            {
                "type": str,
                "help": "output folder for translation",
            },
        ),
        (
            "to_language",
            {
                "type": str,
                "help": "target language to translate to",
            },
        ),
        (
            "--model",
            {
                "type": str,
                "default": "gpt-5.1",
                "help": "Model to use for translation, e.g., 'gpt-3.5-turbo' or 'gpt-4'",
            },
        ),
        (
            "--num_threads",
            {
                "type": int,
                "default": 10,
                "help": "number of threads to use for translation",
            },
        ),
        (
            "--from_language",
            {
                "type": str,
                "default": "English",
                "help": "source language to translate from",
            },
        ),
        (
            "--skip_existing",
            {
                "default": False,
                "help": "skip existing files",
                "action": "store_true",
            },
        ),
    ]

    for argument, kwargs in arguments:
        parser.add_argument(argument, **kwargs)

    options = parser.parse_args()
    return options


def main():
    """Main function"""
    options = parse_arguments()
    input_folder = options.input_folder
    output_folder = options.output_folder
    if not Path(input_folder).is_dir():
        raise Exception("Input path must be a folder.")

    translate(input_folder, output_folder, options)


if __name__ == "__main__":
    main()
