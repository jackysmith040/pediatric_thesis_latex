import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from src.engine.config import settings
from src.engine.tracker_manager import TrackerManager
from src.engine.counter import Counter
from src.engine.detector import Detector
from src.engine.streamer import mjpeg_generator
from src.engine.telemetry import TelemetryDispatcher

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core Application State
tracker_manager = TrackerManager(expiry_seconds=settings.ID_EXPIRY_SECONDS)
counter = Counter(tracker_manager=tracker_manager)
detector = Detector(counter=counter)
telemetry = TelemetryDispatcher(counter=counter)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting CV Engine...")
    asyncio.create_task(telemetry.start())
    yield
    # Shutdown
    logger.info("Shutting down CV Engine...")
    telemetry.stop()
    detector.release()

app = FastAPI(title="Pediatric Patient CV Engine", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "CV Engine is running. Access /video_feed for MJPEG or /ws/video_feed for WebSocket."}

@app.get("/video_feed")
def video_feed():
    """
    MJPEG streaming endpoint for the live video feed (Legacy Fallback).
    """
    return StreamingResponse(
        mjpeg_generator(detector),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws/video_feed")
async def websocket_video_feed(websocket: WebSocket):
    """
    WebSocket endpoint for the live video feed.
    Sends raw JPEG bytes directly to the browser for ultra-low latency.
    """
    await websocket.accept()
    generator = detector.get_frame_generator()
    try:
        while True:
            # generator is blocking (uses time.sleep and cv2.imencode)
            # Run the generator step in a thread pool to avoid blocking the async event loop
            frame_bytes = await asyncio.to_thread(next, generator)
            await websocket.send_bytes(frame_bytes)
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket video feed")
    except Exception as e:
        logger.error(f"WebSocket video feed error: {e}")
