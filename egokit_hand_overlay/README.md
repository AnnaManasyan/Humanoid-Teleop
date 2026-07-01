# EgoKit-Quest Hand Overlay

Paints the Meta Quest 3 OpenXR 26-joint hand tracking onto the EgoKit ego-view
video, producing `internal_overlay.mp4`.

## 1. Install the app on the Quest 3
Sideload **EgoKit-Quest** from the Downloads section of
https://www.chuange.org/papers/EgoKit.html#downloads

## 2. Record
- Launch the EgoKit app on the headset. **It crashes on roughly 19 of 20
  launches — just keep relaunching. Once it actually starts it runs stable and
  will not crash mid-recording.**
- **Volume +** starts recording, **Volume −** stops it. Each session is saved to
  `DCIM/Recordings/<timestamp>/` as `internal.mp4`, `poses.txt`, `log.txt`.

## 3. Transfer to the Mac
Connect the Quest to the MacBook via USB, open **[OpenMTP](https://openmtp.ganeshrvel.com/)**,
and copy the session folder (`internal.mp4`, `poses.txt`, `log.txt`) somewhere
local. Put `overlay_hands.py` in that same folder.

## 4. Run the overlay script
From the folder that contains `internal.mp4`, `poses.txt`, `log.txt`:

```bash
# prerequisites: ffmpeg (brew install ffmpeg)
python3 -m venv .venv && source .venv/bin/activate
pip install numpy opencv-python mediapipe
curl -sSL -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

python overlay_hands.py        # -> writes internal_overlay.mp4
```

## Notes
- The camera **calibration** (intrinsics, lens distortion, camera-from-head
  extrinsic) and the **video↔pose time offset** are baked into the script for the
  headset/recording it was tuned on. A different Quest unit or capture mode will
  need re-tuning of those constants at the top of `overlay_hands.py`.
- A hand is only drawn when MediaPipe confirms a hand is actually visible there,
  so tracked-but-out-of-view or hallucinated hands are not painted onto the video.
