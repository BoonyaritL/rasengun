# 🌀 Rasengan — Hand Gesture System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.33-orange)

---



| Gesture | Action | Effect |
|---------|--------|--------|
|  **กำหมัด (Fist)** | Charge Chakra | Purple pulsing aura รอบฝ่ามือ |
|  **เปิดฝ่ามือ (Open Palm)** | Rasengan | กระสุนวงจักรค่อยๆ ขยายขึ้นในฝ่ามือ |
|  **ชี้ (Point)** | Shoot | ยิงกระสุนวงจักรออกไปตามทิศนิ้วชี้ |

### 🔊 Sound Effects
- เสียง Rasengan **loop** 



## 🏗 Project Structure

```
Rasengun/
├── main.py                      # Game loop, state machine, webcam I/O
├── gesture_detector.py          # MediaPipe Tasks API — ตรวจจับมือ 21 จุด
├── face_detector.py             # OpenCV Haar Cascade — ตรวจจับใบหน้า
├── renderer.py                  # วาดเอฟเฟกต์ทั้งหมด (Rasengan, mask, HUD)
├── sprite_loader.py             # โหลดภาพ Rasengan + สร้าง rotation frames
├── projectile_system.py         # ระบบกระสุน + particle physics
├── sound_manager.py             # ระบบเสียงด้วย pygame.mixer
├── rasengan2.png                # ภาพกระสุนวงจักร (894×894 BGRA)
├── rasengan-sound-effect.mp3    # เสียง Rasengan
├── hand_landmarker.task         # MediaPipe hand detection model
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install pygame
```

### 2. Run

```bash
python main.py
```


กด **Q** เพื่อออก

---

## 🔧 How It Works

### Gesture Detection (MediaPipe Tasks API)
- ใช้ **HandLandmarker** ตรวจจับ **21 จุด** บนมือ
- ตรวจสอบสถานะนิ้ว (งอ/เหยียด) จาก tip vs PIP joint
- จำแนกเป็น **FIST**, **OPEN_PALM**, **POINTING**, หรือ **UNKNOWN**

### Rasengan Rendering (Sprite-based)
- โหลดภาพ `rasengan2.png` (894×894 พร้อม alpha)
- สร้าง **36 rotation frames** (หมุนทุก 10°) สำหรับ animation
- ขนาดค่อยๆ โต **10px → 300px** ใน 3 วินาที (ease-out cubic)
- Overlay ด้วย alpha blending + glow effect


### State Machine
```
IDLE → FIST (Charge) → OPEN_PALM (Rasengan + Sound) → POINTING (Shoot!) → IDLE
```

---

## ⚙️ Configuration

ปรับค่าได้ที่ `main.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam device index |
| `SHOOT_COOLDOWN` | `0.8` | วินาทีระหว่างยิงแต่ละครั้ง |
| `PROJECTILE_SPEED` | `900.0` | ความเร็วกระสุน (pixels/sec) |
| `MIN_RASENGAN_TIME` | `0.5` | เวลาขั้นต่ำก่อนยิงได้ |

ขนาด Rasengan แก้ที่ `renderer.py` บรรทัด ~97:
- `MIN_SIZE` = ขนาดเริ่มต้น (default: 10px)
- `MAX_SIZE` = ขนาดเต็ม (default: 300px)

---

## 📋 Requirements

- Python 3.8+
- Webcam


### Dependencies
- `opencv-python` >= 4.8
- `mediapipe` >= 0.10
- `numpy` >= 1.24
- `pygame` >= 2.6
