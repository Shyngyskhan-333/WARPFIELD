# WARPFIELD - one-day side-project

![Demo](WARPFIELDdemo.gif)


A real-time **AR hand-tracking playground** built with Python. Point your webcam at your hands and watch live image/video panels warp and stretch between your fingertips — powered by MediaPipe, OpenCV, and a dark PyQt6 UI inspired by Vercel and Higgsfield.

---

## What It Does

The app detects your hands via webcam and uses your **fingertip positions as anchor points** to define quadrilateral "planes" in space. Each plane gets texture-mapped in real time — you can load images, videos, or apply visual effects, and they'll stretch and follow your hand movements frame by frame.

- **1 hand** → one quad anchored to your four fingertips  
- **2 hands** → four quads, each stretched between matching finger pairs across both hands (index-index, middle-middle, etc.)

---

## Features

- Real-time hand landmark detection (MediaPipe, up to 2 hands)
- Perspective-warped overlay planes that follow your fingers
- EMA smoothing on anchor points to reduce jitter
- **Camera selector** in the top navbar — auto-detects available cameras, hot-swaps without restart
- 4 independently configurable planes, each with:
  - **Texture source**: Static image, grid pattern, custom image, custom video, or live camera feed
  - **Filter effects**: Edge Detect, Pixelated View, Distance Modulator, Red Silhouette, Dotted Red Silhouette, Paper Tear Silhouette
  - **Opacity control**
- Live FPS counter in the status bar
- Animated pulsing status dot with per-hand-count feedback
- Collapsible plane panels with numbered badges
- Custom `Outfit.ttf` font + Vercel/Higgsfield-inspired monochrome dark UI

---

## Requirements

- Python 3.12
- Webcam

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** If you're using a non-standard Python install, always `pip install` via the same interpreter you run the script with:
> ```bash
> C:\path\to\your\python.exe -m pip install -r requirements.txt
> ```

---

## Running

```bash
python hand_tracker_step1.py
```

---

## Project Structure

```
WARPFIELD/
├── hand_tracker_step1.py   # Entire application (single file)
├── requirements.txt        # Pinned dependencies
├── placeholder.jpg         # Auto-generated on first run if missing
├── Outfit.ttf              # Custom UI font
└── README.md
```

---

## How the AR Overlay Works

1. MediaPipe detects hand landmarks at 21 points per hand
2. Specific fingertip landmarks are selected as **quad corners**
3. Each frame, those 4 points define a destination quadrilateral
4. `cv2.findHomography` computes the perspective transform from the texture's rectangular source to that quad
5. `cv2.warpPerspective` warps the texture into the frame
6. Alpha blending composites it over the camera feed with the configured opacity

EMA (Exponential Moving Average) smoothing is applied per landmark per frame to reduce jitter without adding noticeable lag.

---

## Controls

| Control | What it does |
|---|---|
| **Camera selector** (navbar) | Switch between detected cameras; hot-swaps live |
| **Show Anchors** | Toggles the square markers at quad corners |
| **Texture Source** | Selects what fills the plane (image, video, camera, grid) |
| **Browse File** | Picks a custom image or video file (shown when relevant) |
| **Filter Effect** | Applies a post-process effect to the texture |
| **Opacity** | Blends the plane from fully transparent to fully opaque |

---

## Known Limitations

- Single Python file — no tests, no packaging, no config file
- Segmentation effects (silhouettes) run MediaPipe Selfie Segmentation, which adds CPU load
