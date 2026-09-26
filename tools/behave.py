"""Run the head: perception -> behavior -> neck.

Usage (from macOS Terminal.app - the only place the camera is accessible):
  .venv/bin/python tools/behave.py --preset calm --dry-run    # decide, print, don't move
  .venv/bin/python tools/behave.py --preset calm --view       # move, with a debug window
Keys in --view: q quit, c clear a neck fault. Ctrl-C quits otherwise; the neck rests and holds.
"""
import argparse
import logging
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # repo root, for package imports when run as a script


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="calm", help="calm, curious or shy")
    parser.add_argument("--view", action="store_true", help="show the camera with overlays")
    parser.add_argument("--dry-run", action="store_true", help="print decisions, don't move the neck")
    parser.add_argument("--log", action="store_true",
                        help="record every behavior and neck tick to logs/ (no images)")
    args = parser.parse_args()

    import cv2

    from behavior.config import load_preset
    from behavior.engine import Engine
    from behavior.runner import Runner
    from perception.config import load_config as load_perception_config
    from perception.detector import YoloPoseDetector
    from perception.pipeline import Perception
    from perception.sources import CameraSource
    from tools.perception_view import draw_overlay

    import json
    import time

    from tools.ticklog import TickLog

    logging.basicConfig(level=logging.INFO)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (REPO / "logs").mkdir(exist_ok=True)
    behavior_log = open(REPO / "logs" / f"behave-{stamp}.jsonl", "w") if args.log else None
    neck_log = TickLog() if args.log else None
    preset = load_preset(args.preset)
    pcfg = load_perception_config()
    neck = None
    if not args.dry_run:
        if input("This MOVES the neck. Hand on servo power? [y/N] ").strip().lower() != "y":
            return
        from neck.bus import ServoBus
        from neck.config import load_config as load_neck_config
        from neck.controller import NeckController

        ncfg = load_neck_config()
        neck = NeckController(ncfg, ServoBus.from_config(ncfg), on_tick=neck_log)
        neck.start()

    latest = {"snap": None, "img": None}
    perception = Perception(CameraSource.from_config(pcfg), YoloPoseDetector.from_config(pcfg, REPO),
                            pcfg, pose_provider=neck.pose if neck else None,
                            on_snapshot=lambda s, i: latest.update(snap=s, img=i))
    on_tick = (lambda rec: behavior_log.write(json.dumps(rec) + "\n")) if behavior_log else None
    turn_range = (-60.0, 60.0)
    if neck is not None:
        t = ncfg.joints["turn"]
        turn_range = tuple(sorted((t.to_deg(t.raw_min), t.to_deg(t.raw_max))))
    runner = Runner(Engine(preset, turn_range=turn_range), perception, neck,
                    stale_after_s=pcfg.stale_after_s, on_tick=on_tick)
    stop = threading.Event()
    try:
        with perception:
            if args.view:
                loop = threading.Thread(target=runner.run, args=(stop,), daemon=True)
                loop.start()
                while True:
                    snap, img = latest["snap"], latest["img"]
                    if snap is not None and img is not None:
                        frame = draw_overlay(img.copy(), snap, 0.0, 0.0)
                        cv2.putText(frame, f"[{preset.name}] {runner.status_line()}", (10, 74),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.imshow("behave", frame)
                    key = cv2.waitKey(20) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("c"):
                        runner.clear_fault()
                stop.set()
                loop.join(timeout=1.0)
            else:
                print(f"[{preset.name}] running - Ctrl-C to stop")
                runner.run(stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if neck is not None:
            neck.close()
        cv2.destroyAllWindows()
        if behavior_log is not None:
            behavior_log.close()
            print("behavior log:", behavior_log.name)
        if neck_log is not None and neck_log.rows:
            print("neck log:", neck_log.save(REPO / "logs" / f"behave-neck-{stamp}.csv"))


if __name__ == "__main__":
    main()
