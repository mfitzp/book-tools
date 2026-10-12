import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
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


def proof_text_file(input_folder, filepath, options):
    input_filepath = os.path.join(input_folder, filepath)

    strict = "Ignore optional tweaks, focus on things that are wrong or confusing."
    if options.strict:
        strict = ""

    proof_instructions = f"""
    You are a technical proofreader, with a good knowledge of Python programming.
    You proofread provided Asciidoctor documents to look for technical and language errors, paying
    attention to meaning, tone and formatting. {strict} Be mindful of the correct use of technical terms and jargon,
    idioms and other aspects. Do not analyse ANY code blocks, leave them as-is. Return your suggestions for
    improvements. Give a short description of the change, following by the unchanged, and then the changed text,
    prefixed with the line numbers of the provided input as NN:
    """

    with open(input_filepath, "r", encoding="utf-8") as f:
        text_to_proof = f.read()

    try:
        response = client.responses.create(
            model=options.model,
            instructions=proof_instructions,
            input=text_to_proof,
            temperature=0.2,
        )
    except Exception as e:
        print(f"Error proofing {input_filepath}: {e}")
        return

    return (filepath, response.output_text)


def proof(input_folder, options):
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
                    proof_text_file,
                    input_folder,
                    filepath,
                    options,
                )
                futures.append(future)

                if (
                    options.max_files
                    and len(futures) > options.max_files
                ):
                    break

        # Iterate over the futures as they complete.
        output = {}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Proofing files",
            unit="files",
        ):
            try:
                result = future.result()
                if result:
                    filepath, notes = result
                    output[filepath] = notes
            except Exception as e:
                print(f"Error occurred during proofing: {e}")

                raise e

    print("All files proofed. Generating report.")
    with open(os.path.join(input_folder, "proof.md"), "w") as f:
        f.write(f"Proof report for {input_folder}:\n")
        for file, notes in output.items():
            f.write(f"{file}\n")
            f.write(f"{notes}\n")
            f.write("\n\n")

    print("Report generated.")


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
            "--strict",
            {
                "default": False,
                "help": "use strict proofing",
                "action": "store_true",
            },
        ),
        (
            "--max_files",
            {
                "type": int,
                "default": 0,
                "help": "maximum files to process",
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
    if not Path(input_folder).is_dir():
        raise Exception("Input path must be a folder.")

    proof(input_folder, options)


if __name__ == "__main__":
    main()
