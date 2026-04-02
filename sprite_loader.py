
import os
import cv2
import numpy as np
from typing import List

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITE_PATH = os.path.join(_SCRIPT_DIR, "rasengan2.png")

# Number of rotation frames to generate for smooth spinning
NUM_ROTATION_FRAMES = 36  # every 10 degrees


def load_sprite_frames(
    path: str = SPRITE_PATH,
    num_frames: int = NUM_ROTATION_FRAMES,
) -> List[np.ndarray]:
    """
    Load a single Rasengan image and create rotation frames for animation.

    The image is rotated in equal steps to create a spinning animation.

    Returns
    -------
    List[np.ndarray]
        BGRA frames, each rotated by a different angle.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load Rasengan image: {path}")

    # Ensure BGRA
    if img.shape[2] == 3:
        img = _white_to_alpha(img)

    # Centre-crop to square if not already
    h, w = img.shape[:2]
    if h != w:
        sq = min(h, w)
        cy, cx = h // 2, w // 2
        half = sq // 2
        img = img[cy - half:cy + half, cx - half:cx + half]

    frames = []
    for i in range(num_frames):
        angle = (360.0 / num_frames) * i
        rotated = _rotate_bgra(img, angle)
        frames.append(rotated)

    return frames


def _rotate_bgra(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a BGRA image by the given angle (degrees), keeping alpha."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Use border value of (0,0,0,0) for transparent background
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


def _white_to_alpha(img_bgr: np.ndarray) -> np.ndarray:
    """Convert white pixels in a BGR image to transparent alpha in BGRA."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    sat_mask = s.astype(np.float32) / 255.0
    val_mask = 1.0 - (v.astype(np.float32) / 255.0)
    combined = np.clip(sat_mask * 3.0 + val_mask * 1.5, 0.0, 1.0)
    alpha = (combined * 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, alpha])


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize a BGRA frame to a square of the given size."""
    if size < 1:
        size = 1
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def overlay_sprite(
    background: np.ndarray,
    sprite: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    glow: bool = True,
) -> None:
    """Overlay a BGRA sprite onto the background at (cx, cy) with alpha blending."""
    if size < 4:
        return

    bh, bw = background.shape[:2]

    # Resize sprite to desired size
    resized = resize_frame(sprite, size)
    sh, sw = resized.shape[:2]

    # Calculate placement (centred on cx, cy)
    x1 = cx - sw // 2
    y1 = cy - sh // 2
    x2 = x1 + sw
    y2 = y1 + sh

    # Clip to frame bounds
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = sw - max(0, x2 - bw)
    src_y2 = sh - max(0, y2 - bh)

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(bw, x2)
    dst_y2 = min(bh, y2)

    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return

    sprite_crop = resized[src_y1:src_y2, src_x1:src_x2]
    if sprite_crop.shape[0] == 0 or sprite_crop.shape[1] == 0:
        return

    # Draw glow behind the sprite
    if glow:
        glow_size = int(size * 0.8)
        overlay = background.copy()
        cv2.circle(overlay, (cx, cy), glow_size, (255, 180, 50), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.10, background, 0.90, 0, background)
        overlay2 = background.copy()
        cv2.circle(overlay2, (cx, cy), glow_size // 2, (255, 220, 120), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay2, 0.06, background, 0.94, 0, background)

    # Re-extract ROI after possible glow modification
    roi = background[dst_y1:dst_y2, dst_x1:dst_x2]

    # Alpha blending
    alpha = sprite_crop[:, :, 3].astype(np.float32) / 255.0
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)

    sprite_bgr = sprite_crop[:, :, :3].astype(np.float32)
    roi_float = roi.astype(np.float32)

    blended = sprite_bgr * alpha_3ch + roi_float * (1.0 - alpha_3ch)
    background[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)
