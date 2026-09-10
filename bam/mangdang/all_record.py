# Copyright 2026 Mangdang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Batch runner that sweeps P-gains and trajectories for the MD01.

Mirrors ``bam/feetech/all_record.py``: for every (kp, trajectory) pair it
prints/announces the current run and shells out to ``bam.mangdang.record``.

Two details are specific to the MD01 rig:

- ``--vin`` defaults to the 12 V servo rail. The driver board has no rail ADC,
  so it is a constant, not a measurement; override it with the bench supply's
  real output, and keep it consistent with ``MD01Actuator.__init__``.
- the sweep holds one gain out for validation (``bam.fit --validation_kp``), so
  keep at least four gains in :data:`KPS`. The defaults bracket the firmware's
  usable range around the 32 documented for the bridge; confirm them on the
  bench once the rig is instrumented.

Each child run is checked: a non-zero exit or a missing log aborts the sweep,
so a broken rig does not silently churn through the whole matrix.
"""

import argparse
import glob
import os
import subprocess
import sys
import time

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--mass", type=float, required=True)
arg_parser.add_argument("--length", type=float, required=True)
arg_parser.add_argument(
    "--arm-mass",
    "--arm_mass",
    dest="arm_mass",
    type=float,
    default=0.0,
    help="arm mass [kg]",
)
arg_parser.add_argument("--motor", type=str, default="md01")
arg_parser.add_argument("--port", type=str, default=None)
arg_parser.add_argument("--id", type=int, default=1)
arg_parser.add_argument("--logdir", type=str, required=True)
arg_parser.add_argument("--vin", type=float, default=12.0)
arg_parser.add_argument("--speak", action="store_true")
arg_parser.add_argument(
    "--dry-run",
    action="store_true",
    help="exercise the sweep without hardware",
)
args = arg_parser.parse_args()

#: P-gain sweep. At least one value is held out for validation in bam.fit.
KPS = [8, 16, 24, 32]

#: Trajectory sweep, restricted to the names registered in bam.trajectory.
trajectories = [
    "lift_and_drop",
    "sin_sin",
    "sin_time_square",
    "up_and_down",
    "steps",
    "half_sine",
]


def run_all() -> None:
    """Sweep every (kp, trajectory) pair by invoking ``bam.mangdang.record``."""
    command_base = [
        sys.executable,
        "-m",
        "bam.mangdang.record",
        "--mass",
        str(args.mass),
        "--arm-mass",
        str(args.arm_mass),
        "--length",
        str(args.length),
        "--id",
        str(args.id),
        "--motor",
        args.motor,
        "--vin",
        str(args.vin),
        "--logdir",
        args.logdir,
    ]
    if args.port:
        command_base += ["--port", args.port]
    if args.dry_run:
        command_base.append("--dry-run")

    for kp in KPS:
        for trajectory in trajectories:
            sentence = f"Kp {kp}, trajectory {trajectory.replace('_', ' ')}"
            print(sentence)

            if args.speak:
                from gtts import gTTS

                myobj = gTTS(text=sentence, lang="en", slow=False)
                myobj.save("/tmp/message.mp3")
                os.system("mpg321 /tmp/message.mp3")

            command = command_base + [
                "--kp",
                str(kp),
                "--trajectory",
                trajectory,
            ]
            result = subprocess.run(command, check=False)

            if result.returncode != 0:
                raise SystemExit(
                    f"run failed (exit {result.returncode}): {' '.join(command)}\n"
                    "Aborting the sweep so a broken rig does not burn through "
                    "the whole matrix."
                )

            produced = glob.glob(f"{args.logdir}/*.json")
            if not produced:
                raise SystemExit(
                    f"no log produced in {args.logdir} for kp={kp}, "
                    f"trajectory={trajectory}; aborting"
                )

            if trajectory == "sin_time_square":
                time.sleep(3)


if __name__ == "__main__":
    run_all()
