# Behavior — Design

**Date:** 2026-09-26
**Scope:** Sub-project 4 (behavior) of the MoreNeck robot: who to look at, following, glances,
lost/idle handling, lean-in, personality presets. Plus the small neck/perception changes it needs.

## Goal and context

| # | Sub-project | Status |
|---|---|---|
| 1–2 | Neck (`neck/`) | Done, verified on hardware |
| 3 | Perception (`perception/`) | Done, verified live (≈ 19 fps, camera-limited) |
| **4** | **Behavior** | **This spec** |
| 5 | Eyes (ESP32 displays) | Later |

**Done means:** `python tools/behave.py --preset calm` (run from Terminal.app — the only place the
camera is accessible) makes the head find people, follow the most engaged one smoothly, glance at
newcomers, lean in when someone close faces it, and settle or scan when alone — within the neck's
limits, pausing (never overriding) on neck faults. `--dry-run` shows the decisions without moving.

**Character:** one engine, personality presets (`calm`, `curious`, `shy`) that only change numbers.

## Constraints from the existing layers

- Neck `look(turn, nod, lean)` in degrees from neutral, clamped to soft limits: turn ≈ ±60°;
  **nod and lean only positive** from neutral (up / forward), since neutral = rest pose on the stops.
  Max speeds: turn 120 °/s, nod 30 °/s, lean 20 °/s. Faults latch; `look()` then returns `False`.
- Perception `Snapshot` ≈ 19 fps: people with stable `id`, `head_deg` (camera yaw/pitch; +yaw left,
  +pitch up), `distance_m`, `facing_score`/`facing`, `motion`, `visible`, `first_seen`;
  `people_or_empty()` for staleness.
- Sign match: +turn = left = +yaw; +nod = up = +pitch.

## Changes to existing layers

1. **Neck — `NeckController.pose() -> (turn_deg, nod_deg, lean_deg) | None`** from the latest
   *measured* positions (`JointConfig.to_deg`), `None` before start or if any reading is missing.
   Thread-safe (under the controller lock).
2. **Perception — `Snapshot.neck_pose: tuple[float, float, float] | None`**: the pose sampled when
   the frame was read (the pipeline already samples the pose provider there; the provider now returns
   the full `(turn, nod, lean)`; `world_yaw_deg` uses element 0). Default `None`.
3. **Perception — tracker ego-motion:** `Tracker.update(detections, t, shift_px=(0.0, 0.0))` adds
   `shift_px` to every predicted box. The pipeline computes it from the pose change between
   consecutive frames: turning left by Δturn moves the scene right in the image, nodding up by Δnod
   moves it down: `shift = (f_px·tan(Δturn), f_px·tan(Δnod))` (sensor-width focal length, as in
   features). No pose → no shift.

## Architecture

```
behavior/
  presets.yaml   calm / curious / shy parameter sets
  config.py      Preset dataclass, load/validate
  attention.py   pure: score people, pick target with stickiness + switch hold, glance requests
  follow.py      pure: follow law, dead zone, lean policy
  idle.py        pure: lost-hold, return to neutral, scan pattern
  engine.py      pure state machine: tick(people, snapshot, neck_status, now) -> Command | None
  runner.py      wires Perception + NeckController + Engine; 20 Hz loop; keys; shutdown
tools/
  behave.py      CLI: --preset NAME [--view] [--dry-run]
tests/
  behavior_sim.py   simulated room + camera + neck for closed-loop tests
```

`engine.py` and its helpers do no I/O and use an injected clock, so everything is tested offline.

### Engine states

| State | Enter when | Does |
|---|---|---|
| `TRACKING` | a target exists | follow the target; lean policy; shy averting |
| `GLANCING` | newcomer/movement glance fires while `TRACKING` | follow the glance person for `glance_s`, then back |
| `AVERTING` | shy preset, target facing continuously for `avert_after_s` | turn `avert_deg` away for `avert_s`, then back |
| `LOST` | target disappears (no longer in people) | hold the last command for `lost_hold_s` |
| `IDLE` | after `LOST`, or nobody from the start | ease to neutral `(0, 0, 0)` and hold |
| `SCANNING` | `idle_scan` preset, `IDLE` for `scan_after_s` | sweep turn between ±`scan_amplitude_deg` at `scan_speed_dps`, nod `scan_nod_deg`, pause `scan_pause_s` at each end |
| `PAUSED` | neck not running or fault latched, or `look()` returned `False` | send nothing; report; resume when the neck runs unlatched again |

Any confirmed, eligible person in `LOST`/`IDLE`/`SCANNING` → `TRACKING` (highest score).

### Attention (`attention.py`)

- Eligible to be acquired or to challenge: `confidence ≥ 0.5`. The current target stays while it is
  present at all (a one-frame confidence dip must not skip the switch hold). Coasting (`visible=False`) people score × `coast_factor` (0.5).
- `closeness = clamp((near_zero_m − d) / (near_zero_m − near_full_m), 0, 1)` with
  `near_full_m = 1.0`, `near_zero_m = 4.0`; unknown distance → 0.
- `score = w_facing·facing_score + w_near·closeness + w_approach·[motion == "approaching"]`,
  plus `stickiness` for the current target.
- **Switch:** a challenger must out-score the current target (including its stickiness)
  continuously for `switch_hold_s`; the timer resets whenever it doesn't.
- **Glances:** a *newcomer* is a person whose `first_seen` is within `newcomer_window_s` (0.5 s) and
  who is not the target; `motion == "passing"` non-targets are *movement* glance candidates. At most
  one glance per `glance_cooldown_s` (3 s); only when enabled by the preset.

### Follow (`engine.py` `_follow` / `_pursue`) — revised after the first live run

*Why revised:* stepping `gain × offset` per frame made the neck accelerate, brake and **stop ~5×/s**
(measured on the robot; reproduced in simulation with the real `step_axis` profile), and the head
chased coasting predictions off the image to its ±60° limit.

- **Aim** = the person's room direction via the exact head kinematics (`behavior/kinematics.py`:
  lean and nod tilt the base, turn rotates on top — turned far, nodding mostly rolls the view, so
  adding angles made nod creep up to 43–72° on the robot). If the aim's turn is beyond the turn
  soft limits, turn goes to the limit and nod holds (no compensating tilt). Computed from the pose
  at capture + camera offset, updated **once per new
  snapshot**, only while the person is **visible** (coasting predictions are never chased).
  Only a head point from real head keypoints counts (`Person.head_from_keypoints`): on the robot a
  box-top fallback pinned at the image edge (face out of frame at 0.8 m) ran nod away to +82°.
  Smoothed across frames: `aim += follow_gain × (measured − aim)`, but only when the difference is
  ≥ `deadband_deg` (3°, ~1.5× the 1.9° measured head-angle noise).
- **Pursuit:** every 20 Hz tick, a critically damped spring (`pursuit_hz`, 1.0 Hz) moves the
  commanded position toward the aim, so the neck receives a gliding target and never stop-starts.
- **No crawling (hops):** below ~8 °/s the servos stick-slip (4–7 Hz shudder measured on the robot;
  64% of turn motion was spent there). A pursuit move runs at ≥ `min_move_speed_dps` (10) and lands
  exactly on the aim; at rest, a new move starts only when the aim is ≥ `hop_start_deg` (4°) away.
  Slow drifts and noise are followed in brisk hops (simulation: crawl 70% → 4%).
- **Lean:** `lean_deg` when the target is within `lean_near_m` (1.5 m) and facing; back to 0 when
  beyond `lean_release_m` (1.8 m) or not facing for `lean_release_s` (1.0 s).
- `neck.look()` clamps to soft limits (nod ≥ 0, lean ≥ 0). Turn acceleration lowered to 200 °/s².
- Presets: `follow_gain` calm/shy 0.25, curious 0.35; shared `deadband_deg` 3.0, `pursuit_hz` 1.0.
- Acceptance (simulation, with 1.9° noise and the real motion profile): < 2 stop-go cycles per second
  of turning; walker at 20 °/s followed with lag < 12° (was 8° — smoothness traded for lag).

### Idle (`idle.py`)

- `LOST`: repeat the last command for `lost_hold_s`.
- `IDLE`: command `(0, 0, 0)`; hold.
- `SCANNING`: the engine advances a scan angle by `scan_speed_dps · dt` each tick (the neck's own
  limits still apply), reversing at ±`scan_amplitude_deg` after `scan_pause_s`.

### Presets (`behavior/presets.yaml`)

| Field | calm | curious | shy |
|---|---|---|---|
| `w_facing` / `w_near` / `w_approach` | 1.0 / 0.6 / 0.5 | 1.0 / 0.6 / 0.8 | 1.0 / 0.6 / 0.3 |
| `stickiness` / `switch_hold_s` | 0.6 / 1.5 | 0.3 / 0.7 | 0.8 / 2.0 |
| `follow_gain` | 0.4 | 0.5 | 0.4 |
| `glance_newcomer` / `glance_movement` | true / false | true / true | false / false |
| `glance_s` / `glance_cooldown_s` | 1.0 / 3.0 | 1.0 / 3.0 | 1.0 / 3.0 |
| `lost_hold_s` | 2.0 | 3.0 | 1.0 |
| `idle_scan` / `scan_after_s` | false / 5 | true / 5 | false / 5 |
| `scan_amplitude_deg` / `scan_speed_dps` / `scan_nod_deg` / `scan_pause_s` | 40 / 15 / 10 / 1 | 40 / 15 / 10 / 1 | 40 / 15 / 10 / 1 |
| `lean_deg` | 5 | 10 | 0 |
| `avert_after_s` / `avert_deg` / `avert_s` | null / 20 / 2 | null / 20 / 2 | 4 / 20 / 2 |

Follow gain 0.4 (not 0.3) for calm/shy: a timing model of the loop (19 fps, 60 ms latency,
pose at capture) gave ~10° walking lag at 0.3 vs 7.5° at 0.4, with no overshoot on a still target.

Shared: `deadband_deg 1.5`, `lean_near_m 1.5`, `lean_release_m 1.8`, `lean_release_s 1.0`,
`coast_factor 0.5`, `near_full_m 1.0`, `near_zero_m 4.0`, `newcomer_window_s 0.5`, `loop_hz 50` (same rate as the neck loop — at 20 Hz the neck finished each
small move and stopped between behavior updates: 8.3 vs 1.0 stop-go/s in simulation).
`avert_after_s: null` disables averting.

### Runner and CLI

- `behave.py --preset calm [--view] [--dry-run]` runs from **Terminal.app** (camera permission).
- Starts `Perception` (with `pose_provider = neck.pose`) and `NeckController` (unless `--dry-run`),
  then a 50 Hz loop (matching the neck): `people_or_empty` → `engine.tick` → `neck.look(*cmd)` only when the command
  changes.
- `--dry-run`: no neck; pose provider returns `None`; the follow base is neutral `(0, 0, 0)` (the head
  really stays there — building on its own last command would wind up); commands are printed (≤ 2/s)
  instead of sent.
- `--view`: the perception debug window plus state, target ID, and command; keys `q` quit,
  `c` clear a neck fault (the only way faults are cleared — never automatically).
- Ctrl-C / `q`: stop perception, `neck.close()` (rests and holds, as configured).
- Camera failure/stale → nobody present (`LOST`→`IDLE`), not `PAUSED`.

## Testing

**Unit (offline):** attention scoring, stickiness and switch hold, glance eligibility/cooldown;
follow signs, dead zone, once-per-snapshot, base pose from the snapshot; lean enter/release
hysteresis; idle/lost/scan transitions and scan reversal; shy averting; preset validation;
`NeckController.pose()`; `Snapshot.neck_pose`; tracker `shift_px`.

**Closed-loop simulation (`tests/behavior_sim.py`):** people at room angles; a simulated camera
producing snapshots at 19 fps with 60 ms latency from the simulated neck pose (FOV 60°); a
simulated neck following look targets at the real speed limits. Assertions:
- a still person 25° left is centred (|yaw| < 2°) within 2 s, overshoot < 2°;
- a person walking at 20 °/s is followed with lag < 8° after 1 s;
- two people: the target switches only after `switch_hold_s` of being out-scored;
- a newcomer glance lasts ≈ `glance_s` and returns to the original target;
- target leaves → `LOST` → neutral after `lost_hold_s`;
- `curious` scans and acquires a person at 30° it could not initially see;
- a neck fault → `PAUSED`, no commands; resumes after clear.

**Hardware (with the user):** `--dry-run` first (no motion), then `calm` live, then the other presets.

## Out of scope

Eyes/ESP32, re-identification of returning people, speech/microphone cues, multi-camera, predictive
(Kalman) following — only if closed-loop lag proves noticeable.
