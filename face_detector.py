
import cv2
import numpy as np
from typing import List, Tuple


class FaceDetector:


    def __init__(self, detect_interval: int = 3):

        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.detect_interval = detect_interval
        self._frame_count = 0
        self._last_faces: List[Tuple[int, int, int, int]] = []

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Returns a list of (x, y, w, h) bounding boxes.
        Uses cached results between detection intervals.
        """
        self._frame_count += 1

        if self._frame_count % self.detect_interval == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces) > 0:
                self._last_faces = [
                    (int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces
                ]
            else:
                self._last_faces = []

        return self._last_faces

    def get_face_center(
        self, face: Tuple[int, int, int, int]
    ) -> Tuple[int, int]:
        x, y, w, h = face
        return x + w // 2, y + h // 2
