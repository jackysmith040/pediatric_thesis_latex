import asyncio
import logging
import httpx
from datetime import datetime
from src.engine.config import settings
from src.engine.counter import Counter

logger = logging.getLogger(__name__)

class TelemetryDispatcher:
    def __init__(self, counter: Counter):
        self.counter = counter
        self.is_running = False
        self.client = httpx.AsyncClient()

    async def start(self):
        """Starts the background telemetry loop."""
        self.is_running = True
        logger.info(f"Starting telemetry dispatcher targeting {settings.TELEMETRY_ENDPOINT}")
        
        while self.is_running:
            payload = {
                "camera_id": settings.CAMERA_ID,
                "current_adults": self.counter.current_adults,
                "current_children": self.counter.current_children,
                "total_daily_adults": self.counter.total_daily_adults,
                "total_daily_children": self.counter.total_daily_children,
                "overcrowding_alert": self.counter.is_overcrowded()
            }
            
            logger.info(
                f"[TELEMETRY] Adults: {payload['current_adults']} | "
                f"Children: {payload['current_children']} | "
                f"Alert: {'YES' if payload['overcrowding_alert'] else 'no'}"
            )

            try:
                # Post to Laravel Restify
                response = await self.client.post(settings.TELEMETRY_ENDPOINT, json=payload)
                response.raise_for_status()
                logger.debug(f"Telemetry sent successfully: {response.status_code}")
            except httpx.RequestError as exc:
                logger.error(f"Failed to send telemetry. Is the dashboard offline? Error: {exc}")
            except httpx.HTTPStatusError as exc:
                logger.error(f"Dashboard returned an error: {exc.response.status_code} - {exc.response.text}")
            
            await asyncio.sleep(settings.TELEMETRY_INTERVAL_SECONDS)

    def stop(self):
        """Stops the telemetry loop."""
        self.is_running = False
        asyncio.create_task(self.client.aclose())
        logger.info("Stopping telemetry dispatcher.")
