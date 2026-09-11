import cv2
import os
import logging
import numpy as np
import torch
import threading
import time
import math
import ultralytics.nn.tasks
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from src.engine.config import settings, PRESET_MODELS

from src.engine.counter import Counter
from typing import Generator

# Security override for PyTorch 2.6+ loading YOLO weights
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

logger = logging.getLogger(__name__)


import supervision as sv
from src.engine.stream_resolver import StreamResolver
from src.engine.tracker_engine import MultiTrackerEngine


class Detector:
    @staticmethod
    def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) in the LAB color space.
        Enhances contrast in low-light clinical triage environments without shifting color balance.
        """
        if frame is None or frame.size == 0:
            return frame
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a_channel, b_channel))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.error(f"Error applying CLAHE preprocessing: {e}")
            return frame
    def __init__(self, counter: Counter):
        self.counter = counter
        self.model_lock = threading.Lock()
        
        # Initialize YOLO Model dynamically with fallback cascade
        self._init_model(settings.MODEL_PATH)

        # Initialize Roboflow Supervision Multi-Tracker Engine & Annotators
        self.tracker_engine = MultiTrackerEngine(initial_mode=settings.DEFAULT_TRACKER_MODE)
        self.current_tracker_status_label = "ByteTrack [Auto]"
        
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace_annotator = sv.TraceAnnotator(trace_length=30, thickness=2)




        # Dynamic Video Source & Lock

        self.source_lock = threading.Lock()
        self.current_source_input = settings.VIDEO_SOURCE
        self.current_source_label = "Webcam 0"
        self.current_source_type = "webcam"
        self.last_error = None
        self.is_webcam = True
        self.cap = None

        # Initialize video source
        self._init_capture(settings.VIDEO_SOURCE)

        # Threading for Decoupled Architecture
        self.latest_frame = None
        self.latest_boxes = []
        self.latest_jpeg_bytes = None
        self.frame_count = 0
        self.box_lock = threading.Lock()
        self.running = True
        self._synthetic_id_counter = 10000

        
        # Thread 1: Camera Capture
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # Thread 2: YOLO Inference
        self.yolo_thread = threading.Thread(target=self._yolo_loop, daemon=True)
        self.yolo_thread.start()

        # Store model class names
        self.names = self.model.names
        self.uses_coco_person = len(self.names) == 1 and 'person' in str(self.names.get(0, '')).lower()
        if self.uses_coco_person:
            logger.info("Generic COCO person model detected. Height heuristics active for pediatric detection.")

        # Tracking History UI
        self.track_history = defaultdict(lambda: [])
        self.rect_width = 1
        self.font = 0.4
        self.text_width = 1
        self.padding = 6
        self.margin = 6
        self.circle_thickness = 3
        self.polyline_thickness = 1

    def _resolve_track_id(self, box, track_id: int, class_id: int, claimed_ids: set) -> int:
        """
        Resolves missing/unassigned track IDs (-1) by matching against existing active centroids
        (excluding IDs already claimed in the current frame) or assigning a stable synthetic ID.
        """
        if track_id != -1 and track_id not in claimed_ids:
            return track_id

        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0

        best_match_id = None
        min_dist = settings.UNTRACKED_SPATIAL_MATCH_RADIUS

        for active_id, (acx, acy, aclass_id) in self.counter.active_centroids.items():
            if aclass_id == class_id and active_id not in claimed_ids:
                dist = math.hypot(cx - acx, cy - acy)
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = active_id

        if best_match_id is not None:
            return best_match_id

        self._synthetic_id_counter += 1
        return self._synthetic_id_counter

    def _init_capture(self, source_input: str) -> tuple[bool, str]:
        """Resolves and opens a video capture source thread-safely."""
        cap, label, err = StreamResolver.resolve_stream_source(source_input)
        if err or cap is None:
            err_msg = err or f"Failed to open video source: {label}"
            self.last_error = err_msg
            logger.error(f"Stream resolution error for '{source_input}': {err_msg}")
            return False, err_msg

        with self.source_lock:
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()
            
            self.cap = cap
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            source_str = str(source_input).strip()
            self.is_webcam = source_str == "" or source_str.isdigit()
            
            if self.is_webcam:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self.current_source_input = source_input
            self.current_source_label = label
            self.last_error = None
            logger.info(f"Successfully opened stream source: {label}")
            return True, label

    def change_source(self, source_input: str) -> tuple[bool, str]:
        """Dynamic runtime method to switch video input sources."""
        return self._init_capture(source_input)

    def _init_model(self, model_path: str) -> tuple[bool, str]:
        """Loads a YOLO model (PyTorch or ONNX) with dynamic fallback cascade and class mapping resolution."""
        target_path = str(model_path).strip()
        selected_path = None
        
        if os.path.exists(target_path):
            selected_path = target_path
        else:
            # Fallback cascade: prioritize ONNX high-speed models, then PyTorch, then base
            fallbacks = [
                settings.MODEL_PATH,
                "models/onnx_versions_fine_tuned/yolo26s_distilled.onnx",
                "models/fine_tuned/yolo26s_distilled.pt",
                "models/onnx_versions_fine_tuned/pediatric-model.onnx",
                "models/onnx_versions_fine_tuned/pediatric-smaller-dataset-trained.onnx",
                "models/onnx_versions_fine_tuned/pediatric-kids-only.onnx",
                "models/fine_tuned/pediatric-model.pt",
                "models/fine_tuned/pediatric-smaller-dataset-trained.pt",
                "models/fine_tuned/pediatric-kids-only.pt",
                "models/base_model/yolo26s.pt",
                "models/yolov8n.pt",
                "yolov8n.pt"
            ]
            for fb in fallbacks:
                if os.path.exists(fb):
                    selected_path = fb
                    break
        
        if not selected_path:
            selected_path = "yolov8n.pt"

        try:
            is_onnx = selected_path.lower().endswith(".onnx")
            if is_onnx:
                logger.info(f"Loading ONNX model weights from {selected_path} for ONNX Runtime inference")
                model_instance = YOLO(selected_path, task='detect')
            else:
                logger.info(f"Loading PyTorch YOLO model weights from {selected_path}")
                model_instance = YOLO(selected_path)
            
            with getattr(self, 'model_lock', threading.Lock()):
                self.model = model_instance
                self.current_model_path = selected_path
                self.is_onnx = is_onnx

                names = getattr(self.model, 'names', None)
                if (not names or len(names) == 0) and is_onnx:
                    try:
                        import onnx
                        import ast
                        loaded_onnx = onnx.load(selected_path)
                        props = {p.key: p.value for p in loaded_onnx.metadata_props}
                        if 'names' in props:
                            names = ast.literal_eval(props['names'])
                    except Exception as meta_err:
                        logger.warning(f"Could not extract metadata class names from ONNX file: {meta_err}")

                if not names:
                    names = {0: 'person'}

                self.names = names
                
                # Dynamic class resolution for fine-tuned and base models
                found_child = None
                found_adult = None
                
                for idx, name in self.names.items():
                    n = str(name).lower()
                    if 'child' in n or 'kid' in n or 'pediatric' in n:
                        found_child = idx
                    elif 'adult' in n:
                        found_adult = idx
                
                if found_child is not None and found_adult is not None:
                    self.child_class_id = found_child
                    self.adult_class_id = found_adult
                elif found_child is not None:
                    self.child_class_id = found_child
                    self.adult_class_id = -1  # Dedicated Kids-Only Model
                elif found_adult is not None:
                    self.adult_class_id = found_adult
                    self.child_class_id = -1  # Dedicated Adult-Only Model
                else:
                    self.adult_class_id = settings.ADULT_CLASS_ID
                    self.child_class_id = settings.CHILD_CLASS_ID
                
                names_str = ' '.join(str(v).lower() for v in self.names.values())
                self.uses_coco_person = 'person' in str(self.names.get(0, '')).lower() and 'child' not in names_str and 'adult' not in names_str

                if hasattr(self, 'counter') and self.counter is not None:
                    self.counter.child_class_id = self.child_class_id
                    self.counter.adult_class_id = self.adult_class_id

                filename = os.path.basename(selected_path)
                self.current_model_label = filename
                
            runtime_tag = "ONNX Runtime" if is_onnx else "PyTorch"
            logger.info(f"Successfully initialized {runtime_tag} model '{filename}' (Child Class ID: {self.child_class_id}, Adult Class ID: {self.adult_class_id})")
            return True, f"Loaded model: {filename} ({runtime_tag})"

        except Exception as e:
            err_msg = f"Failed to load model from {target_path}: {e}"
            logger.error(err_msg)
            return False, err_msg

    def change_model(self, model_path: str) -> tuple[bool, str]:
        """Dynamic runtime method to switch YOLO model weights."""
        return self._init_model(model_path)

    def change_tracker_mode(self, mode: str) -> tuple[bool, str]:
        """Dynamic runtime method to switch tracking algorithms or auto mode."""
        success, msg = self.tracker_engine.set_mode(mode)
        if success:
            self.current_tracker_status_label = self.tracker_engine.get_info()["status_label"]
        return success, msg



    def _capture_loop(self):
        """Continuously drain buffer and stream frames. Auto-loops finite video files/streams."""

        while self.running:
            t_start = time.time()
            with self.source_lock:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.05)
                    continue

                ret, frame = self.cap.read()

            if ret:
                # Downscale high-resolution video frames (e.g. 1080p/4K) to MAX_FRAME_WIDTH for smooth 30 FPS
                h, w = frame.shape[:2]
                if w > settings.MAX_FRAME_WIDTH:
                    new_h = int(h * (settings.MAX_FRAME_WIDTH / float(w)))
                    frame = cv2.resize(frame, (settings.MAX_FRAME_WIDTH, new_h), interpolation=cv2.INTER_AREA)

                # Mirror flip for live webcam, original orientation for streams & video files
                self.latest_frame = cv2.flip(frame, 1) if self.is_webcam else frame
                
                # Pre-encode annotated JPEG frame immediately for zero-latency video streaming
                try:
                    annotated = self.latest_frame.copy()
                    with self.box_lock:
                        current_boxes = list(self.latest_boxes)
                        
                    sorted_boxes = sorted(current_boxes, key=lambda b: float(b[0][0]))
                    class_counters = defaultdict(int)
                    box_indices = {}
                    for box, track_id, class_id in sorted_boxes:
                        class_counters[class_id] += 1
                        box_indices[id(box)] = class_counters[class_id]

                    for box, track_id, class_id in current_boxes:
                        idx = box_indices.get(id(box))
                        self.draw_bbox(annotated, box, track_id, class_id, class_index=idx)
                        if track_id != -1:
                            self.draw_trail(annotated, box, track_id, class_id)

                    ret_jpg, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ret_jpg:
                        self.latest_jpeg_bytes = buffer.tobytes()
                        self.frame_count += 1
                except Exception as e:
                    logger.error(f"Error rendering frame in capture loop: {e}")

                if not self.is_webcam:
                    file_fps = float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 30.0
                    target_delay = 1.0 / max(10.0, file_fps) if file_fps > 0 else 0.033
                    t_elapsed = time.time() - t_start
                    remaining_delay = max(0.001, target_delay - t_elapsed)
                    time.sleep(remaining_delay)
            else:
                # EOF reached on video stream/file -> rewind to frame 0 for continuous looping test
                with self.source_lock:
                    if self.cap is not None and not self.is_webcam:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)


    def _yolo_loop(self):
        """Continuously run CLAHE preprocessing, YOLO inference, and Supervision ByteTrack multi-object tracking."""
        while self.running:
            if self.latest_frame is None:
                time.sleep(0.01)
                continue
                
            frame = self.latest_frame.copy()
            frame_h = float(frame.shape[0])
            
            # Apply CLAHE Preprocessing in LAB color space if enabled
            proc_frame = self.apply_clahe(frame, settings.CLAHE_CLIP_LIMIT, settings.CLAHE_TILE_GRID_SIZE) if settings.ENABLE_CLAHE else frame
            
            # Clean expired IDs from memory
            self.counter.tracker_manager.clean_expired_ids()
            self.counter.cleanup_lost_tracks()

            # Run YOLO prediction (optimal conf threshold per model type)
            conf_thresh = settings.PEDIATRIC_CONF_THRESHOLD if self.uses_coco_person else min(0.25, settings.PEDIATRIC_CONF_THRESHOLD)
            predict_kwargs = {
                "conf": conf_thresh,
                "iou": settings.IOU_THRESHOLD,
                "imgsz": 480,
                "verbose": False
            }
            if getattr(self, 'uses_coco_person', False):
                predict_kwargs["classes"] = [0]  # Restrict COCO predictions strictly to person class 0

            results = self.model.predict(proc_frame, **predict_kwargs)

            new_boxes = []
            if results and len(results) > 0 and len(results[0].boxes) > 0:
                # Convert Ultralytics results to Supervision Detections
                detections = sv.Detections.from_ultralytics(results[0])
                
                # Update Multi-Tracker Engine with detections and frame for scene analysis
                detections, status_label = self.tracker_engine.update_with_detections(detections, frame)
                self.current_tracker_status_label = status_label

                boxes = detections.xyxy

                raw_track_ids = detections.tracker_id
                if raw_track_ids is None:
                    track_ids = [-1] * len(boxes)
                else:
                    track_ids = [int(tid) if tid is not None else -1 for tid in raw_track_ids]
                
                class_ids = detections.class_id.astype(int).tolist() if detections.class_id is not None else [0] * len(boxes)

                claimed_ids = set()
                for raw_id in track_ids:
                    if raw_id != -1:
                        claimed_ids.add(raw_id)
                
                for box, raw_track_id, raw_class_id in zip(boxes, track_ids, class_ids):
                    target_class_id = raw_class_id

                    child_cls = getattr(self, 'child_class_id', settings.CHILD_CLASS_ID)
                    adult_cls = getattr(self, 'adult_class_id', settings.ADULT_CLASS_ID)

                    # Apply Ground-Plane Perspective Normalization ONLY for generic COCO person models
                    if self.uses_coco_person:
                        target_class_id = self.calculate_perspective_class(box, frame_h, raw_class_id, child_cls, adult_cls)

                    valid_classes = [c for c in [child_cls, adult_cls] if c != -1]
                    if target_class_id in valid_classes or (target_class_id == 0 and len(valid_classes) == 0):
                        resolved_id = self._resolve_track_id(box, raw_track_id, target_class_id, claimed_ids)
                        claimed_ids.add(resolved_id)
                        self.counter.process_detection(resolved_id, target_class_id, box)
                        new_boxes.append((box, resolved_id, target_class_id))
            
            with self.box_lock:
                self.latest_boxes = new_boxes



    @staticmethod
    def calculate_perspective_class(box: np.ndarray, frame_h: float, raw_class_id: int, child_cls: int, adult_cls: int) -> int:
        """
        Calculates ground-plane perspective normalized class.
        In 2D wall/ceiling camera feeds, vertical position y_bottom correlates with ground distance Z.
        Incorporate sitting-pose aspect ratio compensation (w/h > 0.55) and non-human geometry filtering.
        """
        if not getattr(settings, 'ENABLE_PERSPECTIVE_CORRECTION', True):
            return raw_class_id

        # Ignore non-target raw classes
        if raw_class_id not in [child_cls, adult_cls, 0]:
            return raw_class_id

        y_bottom = float(box[3]) / max(1.0, frame_h)
        box_h = float(box[3] - box[1]) / max(1.0, frame_h)
        box_w = float(box[2] - box[0]) / max(1.0, frame_h)

        if box_h <= 0.01 or box_w <= 0.01:
            return raw_class_id

        aspect_ratio = box_w / max(0.001, box_h)

        # Reject non-person geometry noise (e.g. extremely wide bench or narrow furniture edge)
        if aspect_ratio > 2.5 or aspect_ratio < 0.15:
            return -1

        horizon_y = getattr(settings, 'PERSPECTIVE_HORIZON_Y', 0.20)
        far_scale = getattr(settings, 'PERSPECTIVE_FAR_HEIGHT_RATIO', 0.22)
        near_scale = getattr(settings, 'PERSPECTIVE_NEAR_HEIGHT_RATIO', 0.55)

        # Pose-aware height compensation: Sitting/bending adults shorten vertical height by ~35-50%
        effective_h = box_h
        if aspect_ratio > 0.55:
            effective_h = box_h * 1.50

        norm_y = max(0.0, min(1.0, (y_bottom - horizon_y) / max(0.01, 1.0 - horizon_y)))
        expected_adult_h = far_scale + norm_y * (near_scale - far_scale)
        norm_ratio = effective_h / max(0.01, expected_adult_h)

        if norm_ratio < 0.70:
            return child_cls
        elif norm_ratio >= 0.85:
            return adult_cls
        else:
            return child_cls if aspect_ratio > 0.60 else adult_cls


    def draw_bbox(self, frame, box, track_id, class_id, class_index=None):
        """Draw modern sci-fi bounding box with corner brackets, translucent fill, and sequential count label."""
        x1, y1, x2, y2 = map(int, box)
        color = colors(int(class_id), True)

        # 1. Semi-transparent fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # 2. Corner brackets
        length = 15
        thickness = 2
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

        # 3. Label with modern translucent background
        child_cls = getattr(self, 'child_class_id', settings.CHILD_CLASS_ID)
        adult_cls = getattr(self, 'adult_class_id', settings.ADULT_CLASS_ID)
        if class_id == child_cls and child_cls != -1:
            class_name = "Child"
        elif class_id == adult_cls and adult_cls != -1:
            class_name = "Adult"
        else:
            class_name = "Patient"

        if class_index is not None:
            label = f"{class_name} #{class_index} [{track_id}]"
        else:
            label = f"{class_name} [{track_id}]"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Text background overlay
        text_overlay = frame.copy()
        cv2.rectangle(text_overlay, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.addWeighted(text_overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_trail(self, frame, box, track_id, class_id):
        """Draw fading tracking trail with a sci-fi centroid dot."""
        x1, y1, x2, y2 = map(int, box)
        track = self.track_history[track_id]
        
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        track.append((cx, cy))
        
        if len(track) > 30:  # Shorter, cleaner trail
            track.pop(0)

        color = colors(int(class_id), True)
        
        # Draw fading trail
        if len(track) > 1:
            for i in range(1, len(track)):
                thickness = int(np.sqrt(float(i)) * 0.8) + 1
                cv2.line(frame, track[i - 1], track[i], color, thickness)
                
        # Draw current centroid (double ring effect)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 6, color, 1)

    def get_latest_jpeg_bytes(self) -> bytes:
        """
        Returns pre-encoded MJPEG frame buffer instantly for zero-latency 30 FPS playback.
        """
        if getattr(self, 'latest_jpeg_bytes', None) is not None:
            return self.latest_jpeg_bytes

        if getattr(self, 'latest_frame', None) is not None:
            frame = self.latest_frame.copy()
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret:
                return buffer.tobytes()

        # Fallback synthetic dark placeholder JPEG during initial camera warm-up
        placeholder = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(placeholder, "INITIALIZING CLINICAL MONITOR...", (180, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 180, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buffer.tobytes() if ret else b""



    def release(self):
        self.running = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if hasattr(self, 'yolo_thread') and self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
