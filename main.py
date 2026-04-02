
import time
import cv2
import numpy as np

from gesture_detector import HandGestureDetector, Gesture
from face_detector import FaceDetector
from projectile_system import ProjectileManager
from sound_manager import SoundManager
from renderer import (
    draw_rasengan,
    draw_projectile,
    draw_charge_indicator,
    draw_hand_landmarks,
    draw_face_aura,
    draw_hud,
)


#  Configuration 

CAMERA_INDEX = 0
WINDOW_NAME = "Rasengan — Hand Gesture System"
SHOOT_COOLDOWN = 0.8        # seconds between shots
CHARGE_TO_RASENGAN = 0.3    # seconds of fist before rasengan unlocks
MIN_RASENGAN_TIME = 0.5     # seconds palm must be open before shooting
PROJECTILE_SPEED = 900.0    # pixels per second

#  State tracking 

class GameState:
    def __init__(self):
        self.is_charging = False
        self.charge_start = 0.0
        self.rasengan_active = False
        self.rasengan_start = 0.0
        self.last_shoot_time = 0.0
        self.last_palm_center = (0, 0)
        self.last_index_tip = (0, 0)
        self.action_label = "IDLE"
        self.gesture_label = "NONE"

    @property
    def charge_time(self) -> float:
        if self.is_charging:
            return time.time() - self.charge_start
        if self.rasengan_active:
            return time.time() - self.rasengan_start
        return 0.0

    @property
    def shoot_cooldown_remaining(self) -> float:
        elapsed = time.time() - self.last_shoot_time
        return max(0.0, SHOOT_COOLDOWN - elapsed)

    @property
    def can_shoot(self) -> bool:
        return (
            self.rasengan_active
            and self.shoot_cooldown_remaining <= 0
            and (time.time() - self.rasengan_start) >= MIN_RASENGAN_TIME
        )


#  Main loop 

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check CAMERA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandGestureDetector(max_num_hands=1)
    face_det = FaceDetector(detect_interval=3)
    projectiles = ProjectileManager()
    sound = SoundManager()
    state = GameState()

    prev_time = time.time()
    fps = 30.0

    print("[INFO] Rasengan system started. Show your hand to the camera!")
    print("[INFO] Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror 
        h, w = frame.shape[:2]
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = fps * 0.9 + (1.0 / max(dt, 1e-6)) * 0.1  # smoothed FPS

        #  Hand detection 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.process(rgb)

        gesture = Gesture.UNKNOWN
        palm_center = None
        index_tip = None

        if detections:
            det = detections[0]
            gesture = det["gesture"]
            palm_center = det["palm_center"]
            index_tip = det["index_tip"]
            state.last_palm_center = palm_center
            state.last_index_tip = index_tip

            # Draw hand skeleton
            draw_hand_landmarks(frame, det["landmarks"])

        #  State machine 
        state.gesture_label = gesture.name

        if gesture == Gesture.FIST:
            state.action_label = "CHARGE CHAKRA"
            if not state.is_charging:
                state.is_charging = True
                state.charge_start = now
            if state.rasengan_active:
                sound.stop_rasengan()  # stop sound 
            state.rasengan_active = False

        elif gesture == Gesture.OPEN_PALM:
            state.action_label = "RASENGAN"
            state.is_charging = False
            if not state.rasengan_active:
                state.rasengan_active = True
                state.rasengan_start = now
                sound.play_rasengan()  #  start looping rasengan sound

        elif gesture == Gesture.POINTING:
            if state.can_shoot:
                state.action_label = "SHOOT!"
                sound.play_shoot()  #  shoot burst sound
                # Direction: from palm toward index tip
                if palm_center and index_tip:
                    dx = index_tip[0] - palm_center[0]
                    dy = index_tip[1] - palm_center[1]
                else:
                    dx, dy = 0, -1  # default upward

                projectiles.spawn(
                    x=state.last_palm_center[0],
                    y=state.last_palm_center[1],
                    direction=(dx, dy),
                    speed=PROJECTILE_SPEED,
                )
                state.last_shoot_time = now
                state.rasengan_active = False
                state.is_charging = False
            else:
                if state.rasengan_active:
                    state.action_label = "AIMING..."
                else:
                    state.action_label = "POINT (no rasengan)"
                # Keep rasengan active while pointing
        else:
            # UNKNOWN or no hand
            if not detections:
                state.action_label = "NO HAND"
                state.gesture_label = "NONE"
            else:
                state.action_label = "IDLE"
            if state.rasengan_active:
                sound.stop_rasengan()  
            state.is_charging = False
            state.rasengan_active = False

        #  Render effects 

        # Face detection & aura
        faces = face_det.detect(frame)
        for face_bbox in faces:
            draw_face_aura(frame, face_bbox, state.action_label,
                           state.charge_time, now)

        # Charge aura
        if state.is_charging and palm_center:
            draw_charge_indicator(frame, palm_center[0], palm_center[1],
                                  state.charge_time, now)

        # Rasengan in palm
        if state.rasengan_active and palm_center:
            draw_rasengan(frame, palm_center[0], palm_center[1],
                          state.charge_time, now)

        # Update & draw projectiles
        projectiles.update(dt, w, h)
        for proj in projectiles.projectiles:
            draw_projectile(frame, proj, now)

        # HUD
        draw_hud(
            frame,
            gesture_name=state.gesture_label,
            action_name=state.action_label,
            cooldown_remaining=state.shoot_cooldown_remaining,
            projectile_count=projectiles.count,
            charge_time=state.charge_time,
            fps=fps,
        )

        #  Display 
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    # Cleanup
    sound.cleanup()
    detector.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Rasengan system shut down.")


if __name__ == "__main__":
    main()
