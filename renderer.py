
import math
import time
from typing import List, Tuple, Optional
import ramdom
import cv2
import numpy as np

from projectile_system import Projectile, Particle
from gesture_detector import HandLandmark
from sprite_loader import load_sprite_frames, overlay_sprite, resize_frame


#  Colour palette (BGR) 

RASENGAN_CORE = (255, 200, 80)
RASENGAN_GLOW = (255, 160, 40)
RASENGAN_WHITE = (255, 255, 240)
CHARGE_COLOR = (180, 100, 255)
HUD_BG = (30, 30, 30)
HUD_TEXT = (220, 220, 220)
SHOOT_COLOR = (80, 220, 255)


#  Load sprite frames at module level 

try:
    RASENGAN_FRAMES = load_sprite_frames()
    _NUM_FRAMES = len(RASENGAN_FRAMES)
    print(f"[INFO] Loaded {_NUM_FRAMES} Rasengan rotation frames.")
except FileNotFoundError as e:
    print(f"[WARN] {e} — falling back to drawn Rasengan.")
    RASENGAN_FRAMES = []
    _NUM_FRAMES = 0


#  Hand skeleton connections 

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


#  Drawing helpers 

def _overlay_circle(frame, center, radius, color, alpha=0.4):
    """Draw a semi-transparent filled circle."""
    overlay = frame.copy()
    cv2.circle(overlay, center, int(radius), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_glow(frame, cx, cy, radius, color, layers=5):
    """Draw concentric semi-transparent circles to simulate glow."""
    for i in range(layers, 0, -1):
        r = int(radius + i * radius * 0.35)
        alpha = 0.08 / i
        _overlay_circle(frame, (cx, cy), r, color, alpha)


#  Rasengan
def draw_rasengan(frame, cx: int, cy: int, charge_time: float, t: float):

    if _NUM_FRAMES > 0:
        frame_idx = int(t * 20) % _NUM_FRAMES  # ~20 fps animation
        sprite = RASENGAN_FRAMES[frame_idx]

        
        MIN_SIZE = 10
        MAX_SIZE = 300
        GROW_DURATION = 3.0  # seconds to reach full size

        progress = min(charge_time / GROW_DURATION, 1.0)
        # Ease-out cubic: 1 - (1 - t)^3
        eased = 1.0 - (1.0 - progress) ** 3
        base_size = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * eased

        # Subtle pulse when near full size
        pulse_strength = eased * 0.06
        pulse = 1.0 + pulse_strength * math.sin(t * 10)
        size = int(base_size * pulse)

        overlay_sprite(frame, sprite, cx, cy, size, glow=True)
    else:
        _draw_rasengan_fallback(frame, cx, cy, charge_time, t)


def _draw_rasengan_fallback(frame, cx, cy, charge_time, t):
    base_radius = min(18 + charge_time * 12, 42)
    pulse = 1.0 + 0.08 * math.sin(t * 12)
    radius = base_radius * pulse

    _draw_glow(frame, cx, cy, radius, RASENGAN_GLOW, layers=6)
    _overlay_circle(frame, (cx, cy), int(radius), RASENGAN_CORE, 0.65)

    for i in range(4):
        angle_start = int(math.degrees(t * 8 + i * 90))
        arc_radius = int(radius * 0.75)
        cv2.ellipse(frame, (cx, cy), (arc_radius, arc_radius), 0,
                     angle_start, angle_start + 60,
                     RASENGAN_WHITE, 2, cv2.LINE_AA)

    inner_r = max(3, int(radius * 0.3))
    _overlay_circle(frame, (cx, cy), inner_r, RASENGAN_WHITE, 0.8)

    for i in range(6):
        angle = t * 6 + i * (math.pi / 3)
        sr = radius * 0.55
        sx = int(cx + sr * math.cos(angle))
        sy = int(cy + sr * math.sin(angle))
        cv2.circle(frame, (sx, sy), 2, RASENGAN_WHITE, -1, cv2.LINE_AA)


#  Projectile renderer — SPRITE VERSION 

def draw_projectile(frame, proj: Projectile, t: float):
    """Draw a flying Rasengan projectile with sprite animation + particle trail."""
    cx, cy = int(proj.x), int(proj.y)

    # Draw particles first (behind the ball)
    for p in proj.particles:
        alpha = p.alpha * 0.5
        pr = max(1, int(p.radius))
        _overlay_circle(frame, (int(p.x), int(p.y)), pr, p.color, alpha)

    if _NUM_FRAMES > 0:
        # Sprite-based projectile (faster animation for flying effect)
        frame_idx = int(t * 30) % _NUM_FRAMES
        sprite = RASENGAN_FRAMES[frame_idx]
        size = int(proj.radius * 2.5)
        overlay_sprite(frame, sprite, cx, cy, size, glow=True)
    else:
        # Fallback
        radius = proj.radius
        _draw_glow(frame, cx, cy, radius, RASENGAN_GLOW, layers=4)
        _overlay_circle(frame, (cx, cy), int(radius), RASENGAN_CORE, 0.7)
        for i in range(3):
            angle_start = int(math.degrees(t * 14 + i * 120))
            arc_r = int(radius * 0.6)
            cv2.ellipse(frame, (cx, cy), (arc_r, arc_r), 0,
                         angle_start, angle_start + 70,
                         RASENGAN_WHITE, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(2, int(radius * 0.25)),
                   RASENGAN_WHITE, -1, cv2.LINE_AA)


#  Charge indicator 

def draw_charge_indicator(frame, cx: int, cy: int, charge_time: float, t: float):
    """Draw a pulsing purple aura while fist is charging chakra."""
    pulse = 1.0 + 0.15 * math.sin(t * 10)
    radius = int((20 + charge_time * 8) * pulse)
    _draw_glow(frame, cx, cy, radius, CHARGE_COLOR, layers=4)

    for i in range(3):
        angle = t * 5 + i * (2 * math.pi / 3)
        r = radius * 0.7
        sx = int(cx + r * math.cos(angle))
        sy = int(cy + r * math.sin(angle))
        cv2.line(frame, (cx, cy), (sx, sy), CHARGE_COLOR, 1, cv2.LINE_AA)


#  Hand skeleton 

def draw_hand_landmarks(frame, landmarks):
    """Draw hand skeleton on the frame from NormalizedLandmark list."""
    h, w = frame.shape[:2]

    points = []
    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    for start, end in _HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end],
                 (200, 200, 200), 1, cv2.LINE_AA)

    for pt in points:
        cv2.circle(frame, pt, 3, (50, 255, 50), -1, cv2.LINE_AA)


#  Naruto Mask Overlay 


def draw_face_aura(
    frame,
    face_bbox,
    action_name: str,
    charge_time: float,
    t: float,
):

    x, y, w, h = face_bbox
    cx = x + w // 2
    cy = y + h // 2

    _draw_whiskers(frame, x, y, w, h)
    _draw_headband(frame, x, y, w, h)


def _draw_whiskers(frame, x: int, y: int, w: int, h: int):

    whisker_color = (40, 40, 50)  # dark brownish
    thickness = 2

    cheek_y_center = y + int(h * 0.62)
    whisker_len = int(w * 0.22)
    whisker_spacing = int(h * 0.05)

    left_cx = x + int(w * 0.22)
    for i in range(-1, 2):
        wy = cheek_y_center + i * whisker_spacing
        x1 = left_cx - whisker_len // 2
        x2 = left_cx + whisker_len // 2
        # Slight angle: outer end slightly droops
        y1 = wy - int(whisker_spacing * 0.15)
        y2 = wy + int(whisker_spacing * 0.15)
        cv2.line(frame, (x1, y1), (x2, y2), whisker_color, thickness, cv2.LINE_AA)

    # Right cheek (3 whiskers)
    right_cx = x + int(w * 0.78)
    for i in range(-1, 2):
        wy = cheek_y_center + i * whisker_spacing
        x1 = right_cx - whisker_len // 2
        x2 = right_cx + whisker_len // 2
        y1 = wy + int(whisker_spacing * 0.15)
        y2 = wy - int(whisker_spacing * 0.15)
        cv2.line(frame, (x1, y1), (x2, y2), whisker_color, thickness, cv2.LINE_AA)


def _draw_headband(frame, x: int, y: int, w: int, h: int):
    """Draw a Konoha headband on the forehead."""
    cx = x + w // 2

    #  Headband band (blue cloth) 
    band_top = y + int(h * 0.02)
    band_bottom = y + int(h * 0.18)
    band_h = band_bottom - band_top

    # Blue cloth — extends wider than the face
    cloth_x1 = x - int(w * 0.12)
    cloth_x2 = x + w + int(w * 0.12)

    # Draw band with slight perspective (trapezoid-ish)
    band_pts = np.array([
        [cloth_x1 + 5, band_top + 3],
        [cloth_x2 - 5, band_top + 3],
        [cloth_x2, band_bottom],
        [cloth_x1, band_bottom],
    ], dtype=np.int32)

    # Blue band fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [band_pts], (180, 80, 20))  # dark blue
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Band border
    cv2.polylines(frame, [band_pts], True, (120, 50, 10), 2, cv2.LINE_AA)

    #  Metal plate (silver rectangle in centre) 
    plate_w = int(w * 0.38)
    plate_h = int(band_h * 0.82)
    plate_x1 = cx - plate_w // 2
    plate_y1 = band_top + (band_h - plate_h) // 2
    plate_x2 = plate_x1 + plate_w
    plate_y2 = plate_y1 + plate_h

    # Metal gradient effect
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (plate_x1, plate_y1), (plate_x2, plate_y2),
                  (200, 200, 210), -1)  # silver
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)

    # Plate border with rivets
    cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2),
                  (140, 140, 150), 2, cv2.LINE_AA)

    # Rivets (small circles on left and right of plate)
    rivet_r = max(2, int(plate_h * 0.12))
    rivet_y = plate_y1 + plate_h // 2
    cv2.circle(frame, (plate_x1 + rivet_r + 3, rivet_y), rivet_r,
               (160, 160, 170), -1, cv2.LINE_AA)
    cv2.circle(frame, (plate_x2 - rivet_r - 3, rivet_y), rivet_r,
               (160, 160, 170), -1, cv2.LINE_AA)

    # Konoha leaf symbol
    _draw_konoha_symbol(frame, cx, plate_y1 + plate_h // 2, plate_h)

    #  Tail ribbons (hanging down on both sides) 
    tail_len = int(h * 0.18)
    # Left tail
    pts_l = np.array([
        [cloth_x1, band_bottom],
        [cloth_x1 + 8, band_bottom],
        [cloth_x1 - 5, band_bottom + tail_len],
        [cloth_x1 - 12, band_bottom + tail_len - 3],
    ], dtype=np.int32)
    overlay3 = frame.copy()
    cv2.fillPoly(overlay3, [pts_l], (180, 80, 20))
    cv2.addWeighted(overlay3, 0.75, frame, 0.25, 0, frame)

    # Right tail
    pts_r = np.array([
        [cloth_x2 - 8, band_bottom],
        [cloth_x2, band_bottom],
        [cloth_x2 + 12, band_bottom + tail_len - 3],
        [cloth_x2 + 5, band_bottom + tail_len],
    ], dtype=np.int32)
    overlay4 = frame.copy()
    cv2.fillPoly(overlay4, [pts_r], (180, 80, 20))
    cv2.addWeighted(overlay4, 0.75, frame, 0.25, 0, frame)


def _draw_konoha_symbol(frame, cx: int, cy: int, size: int):

    s = max(size // 3, 8)  # scale factor

    # Leaf shape: a pointed ellipse
    leaf_color = (80, 60, 30)  # dark colour for engraving look

    # Main leaf body (upward-pointing ellipse)
    cv2.ellipse(frame, (cx, cy - s // 6), (s // 2, s // 1),
                0, 200, 340, leaf_color, 2, cv2.LINE_AA)

    # Spiral in the centre (the swirl of the leaf symbol)
    # Draw as a small spiral using arcs
    spiral_r = s // 3
    cv2.ellipse(frame, (cx, cy), (spiral_r, spiral_r),
                0, 0, 270, leaf_color, 2, cv2.LINE_AA)
    # Inner smaller arc
    cv2.ellipse(frame, (cx + 1, cy + 1), (spiral_r // 2, spiral_r // 2),
                0, 180, 450, leaf_color, 1, cv2.LINE_AA)

    # Stem line (vertical line going down from spiral)
    cv2.line(frame, (cx, cy + spiral_r - 1), (cx, cy + s // 2 + 2),
             leaf_color, 2, cv2.LINE_AA)

    # Triangle point at top
    tip_y = cy - s // 1 + 2
    cv2.line(frame, (cx, tip_y), (cx - 3, cy - s // 3),
             leaf_color, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, tip_y), (cx + 3, cy - s // 3),
             leaf_color, 2, cv2.LINE_AA)


#  HUD 

def draw_hud(
    frame,
    gesture_name: str,
    action_name: str,
    cooldown_remaining: float,
    projectile_count: int,
    charge_time: float,
    fps: float,
):
    """Draw a translucent heads-up display in the top-left corner."""
    h, w = frame.shape[:2]

    bar_h = 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, bar_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 22
    cv2.putText(frame, f"GESTURE: {gesture_name}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_TEXT, 1, cv2.LINE_AA)
    y += 24
    cv2.putText(frame, f"ACTION:  {action_name}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                SHOOT_COLOR if action_name == "SHOOT!" else HUD_TEXT,
                1, cv2.LINE_AA)
    y += 24
    cv2.putText(frame, f"CHARGE:  {charge_time:.1f}s", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CHARGE_COLOR, 1, cv2.LINE_AA)
    y += 24
    cd_text = (f"COOLDOWN: {cooldown_remaining:.1f}s"
               if cooldown_remaining > 0 else "COOLDOWN: READY")
    cd_color = (0, 100, 255) if cooldown_remaining > 0 else (0, 255, 100)
    cv2.putText(frame, cd_text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cd_color, 1, cv2.LINE_AA)
    y += 24
    cv2.putText(frame, f"PROJECTILES: {projectile_count}  |  FPS: {fps:.0f}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    instructions = "FIST=Charge | OPEN PALM=Rasengan | POINT=Shoot | Q=Quit"
    tw = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    ix = (w - tw) // 2
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (ix - 10, h - 30), (ix + tw + 10, h), HUD_BG, -1)
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, instructions, (ix, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
