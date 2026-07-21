import asyncio
import os
from typing import Optional
from dotenv import load_dotenv
import click
from groq import AsyncGroq
import time

try:
    from ner.interfaces.llm_client import LLMClient
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ner.interfaces.llm_client import LLMClient


class GroqMountainExtractor(LLMClient):
    def __init__(
            self,
            api_key: str,
            model_name: str = "qwen/qwen3.6-27b",
            queries_per_minute: Optional[int] = None,
    ) -> None:
        """Create a Groq-backed extractor with the selected model and rate limit."""
        system_prompt = (
            "You are a strict entity extraction assistant. Your task is to identify and extract "
            "the name of a mountain from the provided text. "
            "If a mountain is mentioned, return ONLY its name, without any extra words, punctuation, or explanation. "
            "If no mountain is mentioned in the text, return exactly the word 'None'."
        )
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            queries_per_minute=queries_per_minute,
            system_prompt=system_prompt,
        )

        self.client = AsyncGroq(api_key=self.api_key)

    async def process_text(self, text: str) -> str:
        """Extract a mountain name from one text and return an empty result when absent."""
        try:
            chat_completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                max_tokens=100,
                reasoning_effort='none',
            )
            result = chat_completion.choices[0].message.content.strip()
            if result.lower() == "none":
                return ""
            return result
        except Exception as e:
            print(f"Request failed: {e}")
            return "0 Error"


@click.command()
@click.option('-t', '--text', multiple=True,
              help="Text to process. This flag can be used multiple times for a batch.")
@click.option('-l', '--limit', default=0, show_default=True, type=int,
              help="Rate limit in queries per minute. Use 0 for unrestricted concurrency.")
@click.option('-w', '--workers', default=2, show_default=True, type=int,
              help="Number of concurrent API requests (semaphores).")
@click.option('--model-name', default="qwen/qwen3.6-27b", show_default=True,
              help="Groq model ID. It must be enabled for the current Groq project.")
def cli(text: tuple, limit: int, workers: int, model_name: str):
    """Extract mountain names from Click-provided texts using the Groq API."""
    # Grab the API key directly from the environment
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY') or os.getenv('GROQ_API')
    assert api_key, "GROQ_API_KEY environment variable is not set. Please export it first."

    # Assert if absolutely no text is provided
    assert text, "Input text is required. Use -t to pass text."

    # Convert the tuple of inputs to a list
    texts_to_process = list(text)

    # Validate each input before initializing the client
    for t in texts_to_process:
        assert t and t.strip(), "One of the provided texts is empty."
        assert len(t) <= 100, f"Text exceeds 100 characters limit: '{t}'"

    # Initialize the extractor
    extractor = GroqMountainExtractor(
        api_key=api_key,
        model_name=model_name,
        queries_per_minute=limit,
    )

    print(f"\nProcessing batch ({len(texts_to_process)} sentences) with {model_name} via Groq SDK...\n")

    # Start timer
    start_time = time.time()

    timed_results = asyncio.run(extractor.process_batch(texts_to_process, max_workers=workers))

    # Stop timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print out the results
    for sentence, (result, request_time) in zip(texts_to_process, timed_results):
        mountain = result if result else "[No mountain found]"
        print(
            f"Text: '{sentence}'\n"
            f"Mountain: {mountain}\n"
            f"Request time: {request_time:.2f} seconds\n"
            f"{'-' * 40}"
        )
    print(f"Total processing time: {elapsed_time:.2f} seconds\n")

if __name__ == "__main__":
    cli()
