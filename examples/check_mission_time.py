"""Measure one episode against the clock Unity displays.

One question, one script: when an episode is given a budget of N seconds of
world time, does the simulator agree? The episode loop itself is the
environment's, so this only builds it, runs it to its end, and reads three
numbers on both sides of the run - the world clock, what the Unity bar shows,
and how much world time each control step actually covered.

Unity has to be reachable for the numbers to mean anything, so the script asks
the editor to enter Play itself and leaves it in standby when it is done. It
uses the same `unity` command line the rest of the tooling uses; if that is not
on the PATH, press Play by hand and pass `--no-editor`.

    python examples/check_mission_time.py --seconds 20 --time-scale 10

A step that covers more than `--control-period` of world time is the number to
watch: it is the policy's decision interval quietly growing past what it was
written for.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robotsnap import runs  # noqa: E402
from robotsnap.client import RobotSNAPClient  # noqa: E402
from robotsnap.envs import RobotSNAPEnv  # noqa: E402

#: Both clocks and the state of the hold, read in one call to the editor.
_PROBE = """
var doc = UnityEngine.Object.FindAnyObjectByType<UnityEngine.UIElements.UIDocument>();
var root = doc != null ? doc.rootVisualElement : null;
var label = root != null
    ? UnityEngine.UIElements.UQueryExtensions.Q<UnityEngine.UIElements.Label>(root, "StatusTime")
    : null;
var clock = RobotSNAP.Core.Clock.Instance;
var elapsed = clock != null ? clock.ElapsedSeconds : double.NaN;
return $"bar={(label != null ? label.text : "?")}"
     + $"|worldClock={elapsed:F3}"
     + $"|timeScale={UnityEngine.Time.timeScale:F1}";
"""


def _editor(project: str, *args: str) -> str:
    """Run one ``unity`` CLI command and return its stdout and stderr together."""
    done = subprocess.run(
        [
            "unity",
            "command",
            *args,
            "--caller",
            "plugin",
            "--skill",
            "unity-cli",
            "--format",
            "json",
            "--project-path",
            project,
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    return done.stdout + done.stderr


def _probe(project: str) -> str:
    """The bar, the world clock and the time scale, as one readable line."""
    answer = _editor(project, "eval", "--code", _PROBE)
    found = re.search(r'"result":\s*"((?:[^"\\]|\\.)*)"', answer)
    return found.group(1) if found else f"(no answer: {answer.strip()[:120]})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--seconds", type=float, default=20.0, help="episode budget, in world seconds")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Unity time scale")
    parser.add_argument("--control-period", type=float, default=0.2, help="world seconds per step")
    parser.add_argument("--pacing", default="lockstep", choices=("lockstep", "free"))
    parser.add_argument("--unity-project", default="/home/adam/robotsnap-unity")
    parser.add_argument(
        "--no-editor",
        action="store_true",
        help="do not drive the editor: press Play yourself before running",
    )
    args = parser.parse_args()

    if not args.no_editor:
        _editor(args.unity_project, "editor_stop")
        time.sleep(2.0)

    client = RobotSNAPClient(port=args.port, autostart=True)
    print(f"bridge listening on {args.port}", flush=True)
    if not args.no_editor:
        time.sleep(0.5)
        _editor(args.unity_project, "editor_play")

    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline and not client.is_connected:
        time.sleep(0.5)
    print(f"peer connected: {client.is_connected}", flush=True)
    if not client.is_connected:
        return 2

    environment = RobotSNAPEnv(
        scenario="mission_time_probe",
        scenario_fields=runs.random_episode_fields(),
        scenario_directory=runs._scenario_directory(args.unity_project),
        client=client,
        time_scale=args.time_scale,
        pacing=args.pacing,
        max_episode_seconds=args.seconds,
        control_period=args.control_period,
    )
    environment.reset()
    print(f"before   {_probe(args.unity_project)}", flush=True)

    rng = np.random.default_rng(0)
    started = time.monotonic()
    steps = 0
    while True:
        _, _, terminated, truncated, info = environment.step(
            rng.uniform(-1.0, 1.0, size=2).astype("float32")
        )
        steps += 1
        if terminated or truncated:
            break
    wall = time.monotonic() - started

    print(f"after    {_probe(args.unity_project)}", flush=True)
    episode = float(info["episode_seconds"])
    print(f"budget   {args.seconds:g} s of world time")
    print(f"episode  {episode:.3f} s over {steps} steps ({'truncated' if truncated else 'terminated'})")
    print(f"per step {episode / steps:.4f} s of world time (asked {args.control_period:g})")
    print(f"wall     {wall:.2f} s, so {episode / wall:.2f}x real time")

    environment.close()
    print(f"stop -> {client.stop_simulation()}", flush=True)
    if not args.no_editor:
        _editor(args.unity_project, "editor_stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
