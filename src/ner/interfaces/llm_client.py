import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional, List


class LLMClient(ABC):
    def __init__(
            self,
            model_name: str,
            api_key: str,
            queries_per_minute: Optional[int] = None,
            system_prompt: str = None,
    ) -> None:
        """Initialize shared client settings for rate-limited LLM requests."""
        self.model_name = model_name
        self.api_key = api_key
        self.queries_per_minute = queries_per_minute
        self.system_prompt = system_prompt

        if queries_per_minute:
            self.rate_limit_delay = 60.0 / queries_per_minute
        else:
            self.rate_limit_delay = 0.0

        self.last_request_time = 0.0
        self._rate_limit_lock = asyncio.Lock()

    async def _wait_for_rate_limit(self):
        """Reserve the next request slot without blocking other tasks on sleep."""
        async with self._rate_limit_lock:
            now = time.monotonic()
            allowed_at = max(now, self.last_request_time + self.rate_limit_delay)
            self.last_request_time = allowed_at

        wait_seconds = allowed_at - time.monotonic()
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

    @abstractmethod
    async def process_text(self, text: str) -> str:
        """Process a single text with the LLM. To be implemented by subclasses."""
        pass

    async def _process_single_text(
        self,
        text: str,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, float]:
        """Process one text under the shared concurrency limit."""
        await self._wait_for_rate_limit()
        async with semaphore:
            started_at = time.perf_counter()
            try:
                result = await self.process_text(text)
            except Exception as e:
                print(f"Error processing text: {e}")
                result = "0 Error"
            return result, time.perf_counter() - started_at

    async def process_batch(
        self,
        texts: List[str],
        max_workers: int = 10,
    ) -> List[tuple[str, float]]:
        """Process multiple texts concurrently and keep per-request timings."""
        semaphore = asyncio.Semaphore(max_workers)
        tasks = [self._process_single_text(text, semaphore) for text in texts]
        return await asyncio.gather(*tasks)
