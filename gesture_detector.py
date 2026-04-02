"""
Recognised gestures
───────────────────
  FIST        all fingers curled          → "charge chakra"
  OPEN_PALM  all fingers extended        → "rasengan"
  POINTING   only index finger extended  → "shoot"
  UNKNOWN    anything else
"""

import os
import sys
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
)

#Model path 

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "hand_landmarker.task")


# Gesture enum

class Gesture(Enum):
    UNKNOWN = auto()
    FIST = auto()
    OPEN_PALM = auto()
    POINTING = auto()


# Landmark indices (same as the legacy mp.solutions.hands)

class HandLandmark:
    """Landmark indices for the 21-point hand model."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


#  Finger state helpers 
_FINGER_TIPS = [
    HandLandmark.INDEX_FINGER_TIP,
    HandLandmark.MIDDLE_FINGER_TIP,
    HandLandmark.RING_FINGER_TIP,
    HandLandmark.PINKY_TIP,
]

_FINGER_PIPS = [
    HandLandmark.INDEX_FINGER_PIP,
    HandLandmark.MIDDLE_FINGER_PIP,
    HandLandmark.RING_FINGER_PIP,
    HandLandmark.PINKY_PIP,
]


def _is_finger_extended(landmarks, tip_id: int, pip_id: int) -> bool:
    """A finger is extended when its tip is above (lower y) its PIP joint."""
    return landmarks[tip_id].y < landmarks[pip_id].y


def _is_thumb_extended(landmarks) -> bool:
    """Thumb extended check via distance ratio."""
    tip = np.array([landmarks[HandLandmark.THUMB_TIP].x,
                    landmarks[HandLandmark.THUMB_TIP].y])
    ip = np.array([landmarks[HandLandmark.THUMB_IP].x,
                   landmarks[HandLandmark.THUMB_IP].y])
    mcp = np.array([landmarks[HandLandmark.THUMB_MCP].x,
                    landmarks[HandLandmark.THUMB_MCP].y])
    return float(np.linalg.norm(tip - mcp)) > float(np.linalg.norm(ip - mcp)) * 1.1


def _get_finger_states(landmarks) -> List[bool]:
    """Return [thumb, index, middle, ring, pinky] — True = extended."""
    states = [_is_thumb_extended(landmarks)]
    for tip_id, pip_id in zip(_FINGER_TIPS, _FINGER_PIPS):
        states.append(_is_finger_extended(landmarks, tip_id, pip_id))
    return states


#  Gesture classifier 

def classify_gesture(landmarks) -> Gesture:
    """Classify a gesture from hand landmark list."""
    states = _get_finger_states(landmarks)
    thumb, index, middle, ring, pinky = states
    fingers_up = sum(states[1:])  # ignore thumb for most gestures

    if fingers_up == 0:
        return Gesture.FIST

    if fingers_up >= 4 and thumb:
        return Gesture.OPEN_PALM

    if index and fingers_up == 1:
        return Gesture.POINTING

    return Gesture.UNKNOWN


#  Palm centre helper 

def get_palm_center(landmarks, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """Return pixel coordinates of the palm centre."""
    ids = [
        HandLandmark.WRIST,
        HandLandmark.INDEX_FINGER_MCP,
        HandLandmark.MIDDLE_FINGER_MCP,
        HandLandmark.RING_FINGER_MCP,
        HandLandmark.PINKY_MCP,
    ]
    xs = [landmarks[i].x for i in ids]
    ys = [landmarks[i].y for i in ids]
    cx = int(np.mean(xs) * frame_w)
    cy = int(np.mean(ys) * frame_h)
    return cx, cy


def get_index_tip(landmarks, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """Return pixel coordinates of the index fingertip."""
    tip = landmarks[HandLandmark.INDEX_FINGER_TIP]
    return int(tip.x * frame_w), int(tip.y * frame_h)


#  Detector class (MediaPipe Tasks API) 

class HandGestureDetector:
    """
    High-level wrapper around the MediaPipe Tasks HandLandmarker
    for per-frame gesture detection in VIDEO mode.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        model_path: str = MODEL_PATH,
    ):
        if not os.path.isfile(model_path):
            print(f"[ERROR] Hand landmarker model not found at: {model_path}")
            print("        Download it with:")
            print("        curl -L -o hand_landmarker.task "
                  "https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            sys.exit(1)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts = 0  # monotonically increasing timestamp (ms)

    def process(self, frame_rgb: np.ndarray):
        """
        Process an RGB frame 
            {
                "gesture": Gesture,
                "palm_center": (cx, cy),
                "index_tip": (ix, iy),
                "landmarks": list of NormalizedLandmark,
            }
        """
        h, w, _ = frame_rgb.shape

        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Advance timestamp
        self._frame_ts += 33  # ~30 fps
        result: HandLandmarkerResult = self.landmarker.detect_for_video(
            mp_image, self._frame_ts
        )

        detections = []
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                gesture = classify_gesture(hand_lms)
                palm_center = get_palm_center(hand_lms, w, h)
                index_tip = get_index_tip(hand_lms, w, h)
                detections.append({
                    "gesture": gesture,
                    "palm_center": palm_center,
                    "index_tip": index_tip,
                    "landmarks": hand_lms,
                })

        return detections

    def release(self):
        self.landmarker.close()
