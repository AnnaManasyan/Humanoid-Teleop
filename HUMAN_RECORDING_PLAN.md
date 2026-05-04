# Human-Only Recording Mode — Design Document

## Goal

Add a recording mode that captures **a human performing tasks** using the same Quest 3 setup currently used for robot teleoperation, **without** running the robot. Output episodes that are structurally compatible with the existing teleop episodes so a single VLA model can train on:

- Teleoperated robot data (existing pipeline, `python main.py --robot g1_inspire`)
- Human-in-the-wild data (new pipeline, this document)

## Requirements

| # | Requirement |
|---|---|
| R1 | **Egocentric viewpoint only** — must capture from the human's head, not an external/static camera. |
| R2 | **No Manus gloves** — must rely on Quest hand tracking only. |
| R3 | **No robot in the loop** — no IK, no arm/hand controllers, no robot ZMQ connection, no `ik_data.jsonl`. |
| R4 | **Same per-frame schema as teleop** for the parts that overlap (head pose, wrist poses, retargeted Inspire hand qpos, raw hand landmarks) so downstream training treats human and robot data uniformly. |
| R5 | **Same recording UX** — pedal/keyboard start/stop, audio cues, episode directory layout, post-session video + zip. |
| R6 | **30 Hz** synced frames (image + JSON line per frame). |

## Constraints discovered during research

- **WebXR Raw Camera Access** is **not supported** in the Quest Browser as of May 2026. Meta announced "stay tuned for WebXR support launching with v77" in [April 2025](https://developers.meta.com/horizon/blog/new-era-mixed-reality-passthrough-camera-api-machine-learning-computer-vision/), but as of Feb 2026 a developer [confirmed on Meta's forums](https://communityforums.atmeta.com/discussions/Questions_Discussions/request-webxr-raw-camera-access-camera-access-feature-in-quest-browser/1367463) that the `camera-access` feature is still rejected on Horizon OS v85. **Treat WebXR camera access as unavailable** — there is no public ETA and we should not architect around it shipping.
- **`getUserMedia`** in Quest Browser does not enumerate the headset cameras.
- **WebXR Depth Sensing** works on Quest 3 but returns low-res stereo disparity — not an RGB video source.
- **Passthrough Camera API (Camera2)** is **shippable since Horizon OS v76** (Apr 2025). It is the **only** way to access Quest 3 egocentric RGB cameras as a third-party developer. Requires a sideloaded native Android app, not browser-based. ([Meta Passthrough Camera API](https://developers.meta.com/horizon/documentation/android-apps/passthrough-camera/))
- Quest **3 / 3S only**. Quest 2 / Pro do not expose this API.

## Recommended architecture — Option A: two apps on Quest

Run **two apps on the Quest simultaneously**, both talking to the same Python recording stack on the laptop:

```
┌──────────────────────── Quest 3 ─────────────────────────┐
│                                                          │
│  ┌─────────────────────┐    ┌────────────────────────┐   │
│  │  Quest Browser      │    │  Sideloaded APK        │   │
│  │  (existing Vuer     │    │  (new — based on Meta  │   │
│  │   WebXR session)    │    │   PassthroughCamera-   │   │
│  │                     │    │   Samples)             │   │
│  │  • head pose        │    │                        │   │
│  │  • wrist 4×4        │    │  • Camera2 access      │   │
│  │  • 25 hand landmarks│    │    to left passthrough │   │
│  │                     │    │    camera              │   │
│  │  via HAND_MOVE      │    │  • JPEG encode         │   │
│  └──────────┬──────────┘    │  • ZMQ PUSH @ 30 Hz    │   │
│             │               └───────────┬────────────┘   │
└─────────────┼───────────────────────────┼────────────────┘
              │ WebXR / Vuer              │ ZMQ over Wi-Fi
              ▼                           ▼
┌────────────────────────── Laptop ─────────────────────────┐
│                                                           │
│   TeleoperatorProcess          ImageReceiver              │
│   (unchanged Vuer/             (modified worker —         │
│    OpenTeleVision)              connects to Quest IP)     │
│            │                            │                 │
│            ▼                            ▼                 │
│       teleop_shm_array            sample-and-hold buffer  │
│            │                            │                 │
│            └──────────┬─────────────────┘                 │
│                       ▼                                   │
│                 30 Hz saver loop                          │
│                       │                                   │
│                       ▼                                   │
│   episode_xxx/                                            │
│     color/frame_NNNNNN.jpg                                │
│     human_data.jsonl                                      │
│     video.mp4   (post-session)                            │
│     color.zip   (post-session)                            │
└───────────────────────────────────────────────────────────┘
```

**This is the recommended path because** it reuses the existing Vuer/WebXR pipeline as-is and keeps the new code surface minimal. **However, it has one architectural risk that must be validated before any APK work begins** — see the Phase 0 gating step in [Implementation order](#implementation-order). If validation fails, fall back to [Option B](#alternative-architecture--option-b-single-native-app).

## Components

### 1. New Quest APK (the only piece of new non-Python code)

- **Base**: start from Meta's official [`PassthroughCameraSamples`](https://developers.meta.com/horizon/documentation/android-apps/passthrough-camera/) (Unity or Native/NDK flavors), or fork [`samuelm2/OpenQuestCapture`](https://github.com/samuelm2/OpenQuestCapture) which already implements Camera2 + session-based recording on Quest 3 with a SideQuest install path. **Avoid QuestCameraKit** as the base — it's a community barebones sample that's been superseded by Meta's own samples.
- **Strip / extend to**: open left passthrough camera → JPEG encode → ZMQ PUSH socket to laptop.
- **Resolution**: 1280×960 @ 30 fps, mono (left camera only). Stereo doubles bandwidth for marginal benefit.
- **Permission**: `horizonos.permission.HEADSET_CAMERA` in manifest; user approves once. Include re-prompt logic — permission can be silently revoked after OS updates.
- **Timestamping**: tag every frame with `SystemClock.elapsedRealtimeNanos`. NTP-style handshake at session start is **required, not optional** — see the time-sync gotcha below.
- **Estimated effort**: ~600–1000 lines of Kotlin/Java + 1–2 weeks for a reliable first build, including iteration on camera lifecycle, permission re-prompting, ZMQ + threading, and time sync. Camera2 is famously verbose. (An earlier draft of this doc said ~200 LOC; that was unrealistic.)

### 2. Modified Python pipeline

Add `--mode human` to [main.py](teleop/main.py) and thread it through. Changes:

| File | Change |
|---|---|
| [main.py](teleop/main.py) | New `--mode {robot,human}` flag. |
| [manager.py](teleop/manager.py) | When `mode=human`: skip `robot_data_shm` allocation, do not spawn `RobotTaskmaster`. Keep `teleop_shm` (55 floats) unchanged. |
| [master.py](teleop/master.py) | No changes for human mode (process simply not started). Optionally add a stub `HumanTaskmaster` if you want to keep lidar. |
| [worker.py](teleop/worker.py) | ZMQ socket connects to Quest APK IP instead of robot at `tcp://192.168.123.164:5556`. Receive single JPEG part instead of multipart RGB+IR+depth. Replace `get_robot_data` with `get_human_data` that pulls only the teleop_shm slice. Write `human_data.jsonl` instead of `robot_data.jsonl`. |
| [vr.py](teleop/vr.py) | Make the `+0.55` Z and `+0.05` X wrist offsets ([vr.py:379-382](teleop/vr.py#L379-L382)) configurable — disabled in human mode so wrist poses stay in raw head-relative frame. |
| `writers.py` | Add `HumanDataWriter` parallel to `IKDataWriter`. |

### 3. Per-frame JSON schema (`human_data.jsonl`)

```json
{
  "time": 1714824000.123,
  "frame_idx": 42,
  "image": "color/frame_000042.jpg",
  "head_rmat": [[...], [...], [...]],
  "left_pose":  [[...4×4...]],
  "right_pose": [[...4×4...]],
  "left_qpos":  [...7 Inspire registers...],
  "right_qpos": [...7 Inspire registers...],
  "left_landmarks":  [[x,y,z], ... ×25],
  "right_landmarks": [[x,y,z], ... ×25]
}
```

Including raw landmarks alongside the retargeted qpos keeps the door open for re-retargeting later (e.g. to a different hand).

### 4. Reused without changes

- `OpenTeleVision` / `VuerTeleop` / `VuerPreprocessor` (hand tracking + retargeting)
- `InspireOptRetargeting` (produces the 7-DoF register output)
- Pedal + keyboard control loop, audio cues
- `ProgressTracker` (episode directory naming)
- Post-session video creation, `color.zip`, `depth.zip`

## Implementation order

### Phase 0 — Validate the architecture before writing anything (1 day)

This phase is **gating**. Do not proceed to Phase 1 if it fails.

0a. Verify Quest 3 OS ≥ v76 and enable Developer Mode.
0b. Sideload Meta's `PassthroughCameraSamples` (prebuilt) or `OpenQuestCapture` APK without modification.
0c. **Critical check**: open the Vuer WebXR session in Quest Browser, then launch the camera APK on top of it. Confirm that:
   - Camera APK successfully grabs frames.
   - Vuer's `HAND_MOVE` events keep firing while the camera APK is in the foreground.
   - WebXR session does not pause or throttle.

If 0c fails: **stop. Do not build the custom APK.** Move to [Option B](#alternative-architecture--option-b-single-native-app).

### Phase 1 — Build the camera APK (only if Phase 0 passes)

1. Fork the chosen base (`PassthroughCameraSamples` or `OpenQuestCapture`) and strip to camera-only.
2. Add ZMQ PUSH of JPEG frames to the laptop, each tagged with `elapsedRealtimeNanos`.
3. Add NTP-style time-sync handshake on connection.
4. Add re-prompt logic for `HEADSET_CAMERA` permission.

### Phase 2 — Python integration

5. Add `--mode human` flag to `main.py` and thread it through.
6. Modify `worker.py` to receive single-JPEG frames from the Quest IP.
7. Skip `RobotTaskmaster` allocation in human mode; add `HumanDataWriter`.
8. Disable wrist offsets in human mode so poses stay in raw head-relative frame.

### Phase 3 — Record and validate end-to-end

9. Record a 30-second test episode.
10. Verify schema parity with teleop episodes; spot-check sync between image timestamps and `HAND_MOVE` timestamps.

## Alternative architecture — Option B: single native app

If Phase 0 validation fails (i.e. WebXR pauses or throttles when the camera APK is foregrounded), collapse the two apps into one. A single Unity or Spatial-SDK app would:

- Open the passthrough camera (Camera2 under the hood).
- Read 25-joint hand tracking via OpenXR / Meta XR Core SDK — the same skeletal data Vuer currently surfaces over WebXR.
- Read head pose via OpenXR.
- Push everything to the laptop over a single ZMQ socket with a single timestamp clock.

The retargeting (`InspireOptRetargeting`) and the recording stack on the laptop side are unchanged.

### Trade-offs

| | Option A (two apps) | Option B (one app) |
|---|---|---|
| Camera access | sideloaded APK | sideloaded APK |
| Hand-tracking source | WebXR `HAND_MOVE` (browser) | OpenXR / Meta XR Core SDK (native) |
| Coexistence risk | **high** — WebXR pauses on focus loss | **none** — single process |
| Time sync | two clocks (browser + APK) | one clock |
| Reuse of `OpenTeleVision` / `VuerTeleop` | full reuse on Quest side | replaced with native XR SDK calls |
| Reuse of `InspireOptRetargeting` | full reuse on laptop | full reuse on laptop |
| Browser dependency | yes (and `HAND_MOVE` requires browser focus) | none |
| Estimated effort | 1–2 weeks APK + 2–3 days Python | 3–5 weeks Unity/native + 2–3 days Python |

Option B is more code, but it eliminates the architectural risk that makes Option A a gamble, and it produces a cleaner system long-term. If the goal is to scale this to many recording sessions or distribute the setup to other people, Option B is probably the better long-term investment regardless of whether Phase 0 passes.

**Recommendation:** start Option A (cheaper if it works), but only after Phase 0 validation. Treat Option B as a real backup, not a hypothetical.

## Gotchas

- **App coexistence (Option A only)**: gated by Phase 0. Don't write APK code until validated.
- **Permission revocation**: `HEADSET_CAMERA` can be silently revoked after an OS update — must include re-prompt logic.
- **Bandwidth**: ~30–60 Mbps with mono JPEG over Wi-Fi 6. Avoid stereo and avoid raw frames.
- **30 fps cap**: do not promise 60 fps from the camera API.
- **Time sync (required, not optional)**: APK frame timestamps (`elapsedRealtimeNanos`) and laptop wall-clock will drift. At 30 Hz over a 10-minute episode, drift will be visible to the policy. A one-time NTP-style handshake at session start is the minimum; sample-and-hold at 30 Hz on the laptop side absorbs jitter but not drift.
- **Hand tracking pause**: Quest hand tracking only fires when controllers are put down — confirmed in existing project memory.

## Out of scope (rejected paths)

- External camera (head-mounted webcam, RealSense, ZED) — rejected by user; egocentric Quest viewpoint required.
- Manus gloves — not available.
- Browser-only solution — Meta has not shipped WebXR raw camera access and shows no signs of doing so.
- Quest depth sensing as a video source — too low-res, no RGB.
- Apple Vision Pro / Project Aria — different hardware; out of scope, but worth noting that the closest comparable datasets (EgoDex, EgoMimic) chose those platforms specifically because their egocentric data pipelines are better-supported.

## Open questions

1. ~~Does running a sideloaded APK alongside the browser session keep `HAND_MOVE` events flowing reliably?~~ **Gating** — answered by Phase 0.
2. Is one camera (left only) sufficient, or does the model benefit from stereo? Default to mono; revisit if needed.
3. Do we want to record raw audio from the headset microphone too? Current pipeline does not.
4. **New**: if we end up on Option B, do we also retire the WebXR/Vuer pipeline for *robot* teleoperation, unifying both modes under one native app? Out of scope for this doc but worth flagging — could simplify the codebase substantially.

## References

- [WebXR Raw Camera Access spec](https://immersive-web.github.io/raw-camera-access/) — not supported on Quest as of v85
- [Meta WebXR Mixed Reality docs](https://developers.meta.com/horizon/documentation/web/webxr-mixed-reality/)
- [Meta Passthrough Camera API](https://developers.meta.com/horizon/documentation/android-apps/passthrough-camera/) — official starting point, includes Unity and Native sample repos
- [`samuelm2/OpenQuestCapture`](https://github.com/samuelm2/OpenQuestCapture) — community sideloaded app already doing Camera2 capture on Quest 3; closer to our use case than QuestCameraKit
- ~~[QuestCameraKit](https://github.com/xrdevrob/QuestCameraKit)~~ — superseded by Meta's official samples; not recommended as a base
- [EgoDex paper](https://arxiv.org/pdf/2505.11709) — example of an egocentric manipulation dataset. **Collected on Apple Vision Pro with ARKit, not Quest 3** (an earlier draft of this doc misattributed it). Cited as evidence the *idea* works, not as a Quest-specific reference.
- [EgoMimic](https://egomimic.github.io/) — co-training human + robot data, collected with Project Aria glasses (also not Quest)

---

## Revision notes

Changes from the previous draft:

- **Constraints**: noted that WebXR camera access is still missing on Horizon OS v85 (Feb 2026), well past Meta's promised "with v77" launch. Treat as permanently unavailable.
- **Renamed** the proposed architecture **Option A** and added an explicit **Option B** (single native app) as a real fallback, with a side-by-side trade-off table.
- **Reordered implementation**: introduced a **Phase 0 validation step** that is gating before any APK code is written. The biggest risk in Option A — does WebXR keep firing `HAND_MOVE` while a foregrounded APK runs — must be tested with a prebuilt APK first.
- **Changed APK starting point** from QuestCameraKit to Meta's official `PassthroughCameraSamples` or `OpenQuestCapture`. QuestCameraKit is a community sample superseded by Meta's own.
- **LOC estimate** revised from ~200 to ~600–1000 for a reliable first build.
- **Time-sync handshake** reclassified from "optional" to **required**.
- **EgoDex citation corrected**: it's an Apple Vision Pro / ARKit dataset, not Quest 3. Removed as a Quest-specific precedent and noted explicitly.
- **Open questions**: marked the coexistence question as gating-and-answered-by-Phase-0; added a forward-looking question about unifying robot teleop under Option B if we go that route.
