# G1 Teleoperation Setup Guide

## 1. Robot Setup

> **Warning:** Position the robot away from tables and obstacles — it raises its arms straight up on startup.

On the physical controller, execute the following modes in order:

1. **Damping**
2. **Locked Standing**
3. **Running Mode**

Refer to the labels printed on the controller for the corresponding buttons.

## 2. Robot (SSH)

```bash
ssh unitree@192.168.1.36       # Password: 123  (IP may change — check the robot)
cd deployment
bash start_image_server.sh     # Enter sudo password: 123 (frees the RealSense)
```

The image server streams RGB/IR/depth on `192.168.123.164:5556`. It runs from a
uv virtualenv at `~/deployment/.venv`.

> **First-time setup only** (if `~/deployment` is missing — e.g. after a reflash):
> copy `teleop/image_server/realsense_server.py` and
> `teleop/image_server/start_image_server.sh` to `~/deployment/` on the robot, then:
> ```bash
> cd ~/deployment
> uv venv --system-site-packages --python "$(which python3)" .venv
> uv pip install --python .venv/bin/python pyzmq
> ```
> (System python already provides pyrealsense2/cv2/numpy; only pyzmq is added.)

## 3. Host Computer

```bash
cd /home/g1/Desktop/G1/Humanoid-Teleop/teleop
micromamba activate tv
python main.py --robot g1_inspire
```

## 4. Quest (VR Headset)

1. Pick up the right-hand controller.
2. In the Quest browser, navigate to:
   `192.168.1.148:8012/?ws=wss://192.168.1.148:8012`
   *(This is the teleop computer's IP — update if it changes.)*
3. Press **Enter VR**.
4. Put the controller down — teleoperate using your hands only.

## 5. Pedal Controls

| Pedal | Action |
|-------|--------|
| Left  | Start episode recording |
| Right | Stop episode recording |

Always alternate: left (start) → right (stop) → left (start) → …

## 6. After the First Episode

Once the first episode finishes, the robot has already raised and lowered its arms — it's now safe to move it to the table for the remaining episodes.

## 7. Shutting Down

> **Warning:** The robot will lower its arms when exiting — move it away from the table first.

In the host computer terminal, type `exit` and press Enter.

## 8. Replay an Episode

To replay a recorded episode's actions on the robot:

```bash
cd /home/g1/Desktop/G1/Humanoid-Teleop/teleop
python3 replay_episode.py data/example/manipulation/fold_scarf/episode_3/data.json
```

Press **Ctrl+C** to stop early — the arm weight ramps down safely.

## 9. Tips

- **Move only your hands** — head tracking is disabled; control is relative to hand position.
- After pressing the right pedal, the robot will stop moving until the next episode is started.