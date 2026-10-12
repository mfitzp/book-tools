import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from random import shuffle

from dotenv import load_dotenv
from git import Repo
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

load_dotenv()
client = OpenAI()
current_timestamp = int(time.time())
base_folder = os.path.dirname(os.path.abspath(__file__))


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
)
def noop():
    pass


def proof_text_file(input_folder, filepath, options):
    input_filepath = os.path.join(input_folder, filepath)

    strict = "Ignore optional tweaks, focus on this that are wrong or confusing."
    if options.strict:
        strict = ""

    proof_instructions = f"""
    You are a technical proofreader, with a good knowledge of Python programming.
    You proofread provided Asciidoctor documents to look for technical and language errors, paying
    attention to meaning, tone and formatting. {strict}
    * Be mindful of the correct use of technical terms and jargon, idioms and other aspects.
    * Do not analyse ANY code blocks, leave them as-is.
    * Do not modify Asciidoctor includes or cross-links.
    * Don't fix which -> that.
    * Don't modify pagination markup (such as <<<).
    * Don't remove blank lines.
    * Don't modify Asciidoctor escapes such as `\_.
    * Don't remove index references such as ((bitwise)).
    In the output return the corrected input without any preamble or description of what you have changed.
    If there are no changes, return "NO CHANGES".
    """

    with open(input_filepath, "r", encoding="utf-8") as f:
        text_to_proof = f.read()

    if not text_to_proof:
        return

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

    output_text = response.output_text

    if (
        output_text == "NO CHANGES"
        or text_to_proof.strip() == output_text.strip()
    ):
        print("No changes in {input_filepath}, skipping write.")
        return

    return (input_filepath, response.output_text)


def proof(input_folder, options):
    repo = Repo(base_folder)
    print("Working in repository: ", repo)
    output_branch = f"autoproof_{input_folder}_{current_timestamp}"
    print(f"Switching to branch {output_branch}")
    if not options.dry_run:
        repo.git.checkout("-b", output_branch)

    files_to_process = []
    with ThreadPoolExecutor(
        max_workers=options.num_threads
    ) as executor:
        # Find all asciidoc files in input_folder, and submit to worked to execute.
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if not file.endswith(".adoc"):
                    continue  # not asciidoctor
                if file.startswith("Book-"):
                    continue  # not content

                filepath = os.path.join(root, file)
                if options.ignore_recently_modified:
                    mtime = os.path.getmtime(filepath)
                    if mtime > current_timestamp - 2419200:
                        continue  # skip file, if recently modified.

                filepath = os.path.relpath(
                    filepath, start=input_folder
                )

                files_to_process.append(filepath)

        # Shuffle.
        shuffle(files_to_process)

        if options.max_files:
            files_to_process = files_to_process[: options.max_files]

        print(f"Preparing to proof {len(files_to_process)} file(s).")

        if options.dry_run:
            return

        # Start runners.
        futures = []
        for file_to_process in files_to_process:
            future = executor.submit(
                proof_text_file,
                input_folder,
                file_to_process,
                options,
            )
            futures.append(future)

        results = {}
        # Iterate over the futures as they complete.
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Proofing files",
            unit="files",
        ):
            try:
                result = future.result()
                if result:
                    filepath, content = result
                    results[filepath] = content
            except Exception as e:
                print(f"Error occurred during proofing: {e}")

                raise e

        # Write each file out, with a commit.
        for filepath, content in results.items():
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            repo.index.add([filepath])
            repo.index.commit(f"autoproof edits {filepath}")
            print(f"Commit: autoproof edits {filepath}")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()

    arguments = [
        (
            "input_folder",
            {
                "type": str,
                "help": "input folder to proof",
            },
        ),
        (
            "--model",
            {
                "type": str,
                "default": "gpt-5.1",
                "help": "Model to use for proofing, e.g., 'gpt-3.5-turbo' or 'gpt-4'",
            },
        ),
        (
            "--num_threads",
            {
                "type": int,
                "default": 10,
                "help": "number of threads to use for proofing",
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
        (
            "--ignore_recently_modified",
            {
                "default": False,
                "help": "skip files that have been modified in the past month",
                "action": "store_true",
            },
        ),
        (
            "--dry_run",
            {
                "default": False,
                "help": "don't actually proof",
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
    if not Path(input_folder).is_dir():
        raise Exception("Input path must be a folder.")

    proof(input_folder, options)


if __name__ == "__main__":
    main()
