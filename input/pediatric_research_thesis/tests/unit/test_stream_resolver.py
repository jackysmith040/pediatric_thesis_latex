import pytest
from unittest.mock import MagicMock, patch
from src.engine.stream_resolver import StreamResolver, PRESET_TEST_STREAMS

def test_is_youtube_url_identifies_youtube_domains():
    assert StreamResolver.is_youtube_url("https://www.youtube.com/watch?v=LXb3EKWsInQ") is True
    assert StreamResolver.is_youtube_url("https://youtu.be/LXb3EKWsInQ") is True
    assert StreamResolver.is_youtube_url("http://youtube.com/embed/123") is True
    assert StreamResolver.is_youtube_url("https://example.com/video.mp4") is False
    assert StreamResolver.is_youtube_url("0") is False

def test_resolve_numeric_camera_source(monkeypatch):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    monkeypatch.setattr("cv2.VideoCapture", lambda idx: mock_cap)

    cap, label, err = StreamResolver.resolve_stream_source("0")
    assert cap is not None
    assert "Webcam 0" in label
    assert err is None

def test_resolve_youtube_url_success(monkeypatch):
    fake_url = "https://www.youtube.com/watch?v=LXb3EKWsInQ"
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    monkeypatch.setattr("cap_from_youtube.cap_from_youtube", lambda url, resolution='360p': mock_cap)
    
    cap, label, err = StreamResolver.resolve_stream_source(fake_url)
    assert cap is not None
    assert "Pediatric Clinic" in label
    assert err is None

def test_resolve_youtube_url_handles_extraction_failure(monkeypatch):
    fake_url = "https://www.youtube.com/watch?v=invalid_id"
    
    def raise_err(url, resolution='360p'):
        raise Exception("Video unavailable")

    monkeypatch.setattr("cap_from_youtube.cap_from_youtube", raise_err)
    
    cap, label, err = StreamResolver.resolve_stream_source(fake_url)
    assert cap is None
    assert "YouTube (Error)" in label
    assert "Video unavailable" in err

def test_presets_exist():
    assert len(PRESET_TEST_STREAMS) >= 2
    assert any(p["type"] == "webcam" for p in PRESET_TEST_STREAMS)
    assert any(p["type"] == "youtube" for p in PRESET_TEST_STREAMS)
