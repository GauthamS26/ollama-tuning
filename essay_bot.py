#!/usr/bin/env python3
"""Simple local essay bot powered by Ollama."""

import argparse
import shutil
import subprocess
import sys

DEFAULT_MODEL = "llama2"


def generate_essay(topic: str, model: str = DEFAULT_MODEL) -> str:
    prompt = (
        "You are an essay writing assistant. "
        "Write a clear, well-structured essay of about 500-700 words on the given topic. "
        "Use a title, introduction, 3-4 body paragraphs, and a conclusion. "
        "Keep the tone informative and readable.\n\n"
        f"Topic: {topic}\n"
    )

    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"Failed to generate essay with model '{model}'. "
            f"Ollama error: {stderr or 'unknown error'}"
        )

    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate essays locally with Ollama")
    parser.add_argument("--topic", help="Generate one essay for this topic and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    args = parser.parse_args()

    if shutil.which("ollama") is None:
        print("Error: 'ollama' is not found in PATH.")
        print("Open a new terminal, then run again.")
        return 1

    if args.topic:
        try:
            essay = generate_essay(args.topic, model=args.model)
            print(essay)
            return 0
        except Exception as exc:
            print(f"Error: {exc}")
            return 1

    print("Local Essay Bot (Ollama)")
    print("Type a topic, or type 'exit' to quit.\n")

    while True:
        topic = input("Enter essay topic: ").strip()

        if not topic:
            print("Please enter a non-empty topic.\n")
            continue

        if topic.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return 0

        try:
            print("\nGenerating essay...\n")
            essay = generate_essay(topic, model=args.model)
            print(essay)
            print("\n" + "-" * 80 + "\n")
        except KeyboardInterrupt:
            print("\nStopped by user.")
            return 1
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
