# EgoKit-Quest Hand Overlay

Paints Meta Quest 3 OpenXR hand tracking onto the EgoKit ego video → `internal_overlay.mp4`.

## 1. Install app on Quest 3
Sideload **EgoKit-Quest** from the Downloads section:
https://www.chuange.org/papers/EgoKit.html#downloads

## 2. Record
- Launch the app. **Crashes ~19/20 launches — just relaunch until it starts; once running it's stable.**
- **Vol +** = start, **Vol −** = stop. Saved to `DCIM/Recordings/<timestamp>/` → `internal.mp4`, `poses.txt`, `log.txt`.

## 3. Transfer to Mac
USB → **[OpenMTP](https://openmtp.ganeshrvel.com/)** → copy the session folder next to `overlay_hands.py`.

## 4. Run
Needs `ffmpeg` (`brew install ffmpeg`) and [`uv`](https://docs.astral.sh/uv/).

```bash
export UV_PROJECT_ENVIRONMENT=egokit-env   # env name (not .venv)
uv sync                                    # recreate env from uv.lock
curl -sSL -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
uv run python overlay_hands.py             # → internal_overlay.mp4
```

## Notes
- Calibration (intrinsics, distortion, camera-from-head extrinsic) + video↔pose offset are baked in for the tuned headset — a different Quest/mode needs re-tuning the constants atop `overlay_hands.py`.
- A hand is drawn only where MediaPipe confirms a real hand → no phantom/out-of-view skeletons.
