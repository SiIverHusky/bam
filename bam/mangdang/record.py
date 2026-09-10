# Copyright 2026 Mangdang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Identification-rig recorder for the Mangdang MD01 servo-actuator.

The MD01 is driven through an AT32F413 driver board behind the ESP32-S3 SPI
bridge, whose host protocol lives in :mod:`bam.mangdang.spi`. This module plays
the generic BAM trajectories on one servo and logs the response to JSON in the
schema ``bam.fit`` expects (see ``docs/identification/acquisition.rst``).

Typical session (driven by :mod:`bam.mangdang.all_record`)::

    uv run python -m bam.mangdang.record \
        --port /dev/ttyUSB0 --id 1 --mass 0.5 --arm-mass 0.02 --length 0.15 \
        --vin 12.0 --kp 32 --trajectory lift_and_drop --logdir data_raw

Units: positions are radians and speeds rad/s. The driver reports ``0.0`` at
the centre of the potentiometer travel, and that same zero is used for the log,
so **mount the arm so the potentiometer centre is the pendulum's downward
pose** (the zero of :class:`~bam.testbench.Pendulum`). The offset between the
two is measured during the pre-roll hold and stored as ``q_offset``.

Two quantities the fit consumes are estimated rather than measured:

- the control signal sent to the motor, reconstructed from the position error
  as ``duty = clip(kp * error_gain * (goal - position), -max_pwm, max_pwm)``
  and reported as ``duty_cycle`` (unitless), and
- the mechanical zero ``q_offset``, measured before the trajectory.

Pass ``--error-gain`` (and optionally ``--max-pwm``) if you have measured them
with an oscilloscope (see ``ADDING_A_MOTOR.md`` §3.2); otherwise the defaults
are used, which is the same trade-off the Feetech recorder makes by not
recording the duty cycle at all. Use ``--dry-run`` to exercise the whole
pipeline without hardware.
"""

import argparse
import datetime
import json
import os
import time

import numpy as np

from bam.mangdang.spi import (
    DEFAULT_CUR_MA,
    DEFAULT_SCALE_DD,
    DEFAULT_VIN,
    SpiMd01IO,
    pick_port,
)
from bam.trajectory import trajectories

#: Fraction of the PWM range the controller is assumed to reach, inherited from
#: the measurements on the other BAM voltage-controlled servos.
DEFAULT_MAX_PWM = 0.97

#: Position-error to duty-cycle gain, matching ``MD01Actuator``. Measured with
#: an oscilloscope (``ADDING_A_MOTOR.md`` §3.2); the default is inherited from
#: the Feetech STS3215 and is only a starting point for the MD01.
DEFAULT_ERROR_GAIN = 0.0104

#: Duration [s] of the pre-roll hold used to settle the rig and estimate the
#: mechanical zero, mirroring the 1 s hold in ``bam/feetech/record.py``.
PREROLL = 1.0

#: How often the control signal and torque-enable flag are recomputed. The
#: bridge runs no local trajectory generator, so a delay here shows up directly
#: in the fit's ``command_delay``; keep it at or below the logging period.
CONTROL_PERIOD = 0.001

#: Safety limits. The MD01 rating is not in the datasheet excerpt used here, so
#: these only catch an obviously wrong rig; tighten them once known.
MAX_CURRENT_MA = 2000
MAX_ABSOLUTE_SPEED = 60.0  # rad/s

def build_parser() -> argparse.ArgumentParser:
    """Build the recorder's argument parser."""
    parser = argparse.ArgumentParser(
        description="Record a BAM identification trajectory from an MD01"
    )
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument(
        "--arm-mass",
        "--arm_mass",
        dest="arm_mass",
        type=float,
        default=0.0,
        help="arm mass [kg] (the recorder writes the arm_mass key)",
    )
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--trajectory", type=str, default="lift_and_drop")
    parser.add_argument("--motor", type=str, default="md01")
    parser.add_argument("--kp", type=float, default=32)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument("--vin", type=float, default=DEFAULT_VIN)
    parser.add_argument("--id", type=int, required=True, help="servo id 1..12")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--cur",
        type=int,
        default=DEFAULT_CUR_MA,
        help="firmware current cap in position mode [mA]",
    )
    parser.add_argument(
        "--scale-dd",
        type=float,
        default=DEFAULT_SCALE_DD,
        help="full-scale travel in deci-degrees (3100 = 310 deg)",
    )
    parser.add_argument(
        "--error-gain",
        type=float,
        default=DEFAULT_ERROR_GAIN,
        help="kp * error * error_gain -> duty cycle",
    )
    parser.add_argument("--max-pwm", type=float, default=DEFAULT_MAX_PWM)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not touch hardware; synthesise feedback",
    )
    return parser


#: CLI arguments. Parsed at import time so ``python -m bam.mangdang.record``
#: behaves like every other recorder; tests monkeypatch it instead of passing
#: argv. Set to ``None`` by ``main()`` after parsing to expose a clean import.
args = build_parser().parse_args()

os.makedirs(args.logdir, exist_ok=True)

if args.trajectory not in trajectories:
    raise ValueError(f"Unknown trajectory: {args.trajectory}")
if not 1 <= args.id <= 12:
    raise ValueError(f"--id must be in 1..12, got {args.id}")


class Recorder:
    """Turns a driver into the sampled state and entries the fit consumes."""

    def __init__(
        self,
        io: SpiMd01IO,
        servo: int,
        kp: float,
        error_gain: float,
        max_pwm: float,
    ):
        self.io = io
        self.servo = servo
        self.kp = kp
        self.error_gain = error_gain
        self.max_pwm = max_pwm
        #: Reconstructed duty cycle of the last commanded sample, in [-1, 1].
        self.duty_cycle = 0.0
        #: Mechanical zero estimated during the pre-roll hold [rad].
        self.q_offset = 0.0

    def compute_duty_cycle(self, goal_position: float, position: float) -> float:
        """Reconstruct the firmware duty cycle from the position error.

        The bridge does not report the duty cycle, so it is recomputed from the
        same P law the firmware runs (``MD01Actuator`` uses
        ``duty = kp * error_gain * Δq``)::

            duty = clip(kp * error_gain * (goal - position), -max_pwm, max_pwm)

        This is an open-loop estimate — it cannot see the firmware's current
        limiter or any saturation on the board. Read ``pwm_duty`` over SPI
        instead if a measured command is needed (one extra transaction per
        sample, so acquisition is slower).
        """
        error = goal_position - position
        duty = self.kp * self.error_gain * error
        self.duty_cycle = float(np.clip(duty, -self.max_pwm, self.max_pwm))
        return self.duty_cycle

    def read_data(self) -> dict:
        """Read one sample from the MD01.

        :returns: Dict with ``position`` [rad], ``speed`` [rad/s],
            ``duty_cycle``, ``load`` [mA of motor current], ``input_volts`` [V]
            and ``temp``. The ``timestamp``, ``goal_position`` and
            ``torque_enable`` keys are added by the acquisition loop, which
            straddles the sample with two clock reads.

        Position, current and velocity come from a single SPI transaction
        (:meth:`~bam.mangdang.spi.SpiMd01IO.read_sample`) — the firmware returns
        position and current together, and the velocity is differentiated from
        the position. Issuing one round-trip instead of three roughly triples
        the achievable logging rate, which keeps the fit's ``command_delay``
        estimate meaningful.
        """
        sample = self.io.read_sample(self.servo)
        volts = self.io.get_present_voltage([self.servo])[0]

        return {
            "position": float(sample["position"]),
            "speed": float(sample["speed"]),
            # BAM's voltage-controlled models consume `duty_cycle`; recording it
            # (rather than a voltage) keeps `vin` an independent fittable input.
            "duty_cycle": float(self.duty_cycle),
            "load": float(sample["current_mA"]),
            "input_volts": float(volts),
            "temp": 0.0,
        }


def open_io(port: str | None):
    """Open the MD01 bridge and return the driver, with the scale synced.

    :param port: Serial device path, or ``None`` to auto-detect it.
    :returns: A :class:`~bam.mangdang.spi.SpiMd01IO` ready for position mode.
    """
    io = SpiMd01IO(port, baud=args.baud, scale_dd=args.scale_dd, dry_run=args.dry_run)
    if not args.dry_run:
        # Adopt the AT32's range_position_deg so the host and board agree on
        # what "310 degrees" means; a mismatch would silently squash the range.
        try:
            io.sync_scale_from_board(args.id)
        except (OSError, TimeoutError) as exc:
            print(
                f"warning: could not read the board scale ({exc}); "
                f"using {io.scale_dd / 10.0:.1f} deg"
            )
    return io


def settle_and_measure_zero(io, servo: int, trajectory) -> float:
    """Hold the start pose, set the gains, and estimate the mechanical zero.

    Commands the trajectory's ``t = 0`` pose while the rig settles, applies the
    firmware gains, then averages the position over the hold. Averaging over a
    full second rejects both the single-sample noise and the small droop that
    follows the first application of the setpoint.

    :returns: The estimated mechanical zero [rad].
    """
    goal_position, _ = trajectory(0.0)

    io.set_mode({servo: 1})
    io.set_P_coefficient({servo: args.kp})
    io.set_D_coefficient({servo: args.kd})
    io.set_goal_position({servo: goal_position})
    io.enable_torque([servo])

    samples = []
    start = time.perf_counter()
    while time.perf_counter() - start < PREROLL:
        samples.append(io.get_present_position([servo])[0])
        time.sleep(CONTROL_PERIOD)

    # The rig's zero is the position it rests at while held at the trajectory
    # origin, which is how the pendulum testbench (arm down, q = 0) is defined.
    return float(np.mean(samples)) - goal_position


def run_trajectory(io, servo: int, recorder: Recorder, trajectory) -> list[dict]:
    """Play ``trajectory`` and return the timestamped entries.

    Torque is toggled when the trajectory's enable flag flips, and the goal
    position is only commanded while torque is on — exactly like the Feetech
    recorder, so ``lift_and_drop`` still produces a free gravity drop.
    """
    entries: list[dict] = []
    start = time.perf_counter()
    torque_enable = False
    last_control_t = 0.0
    goal_position, _ = trajectory(0.0)

    while True:
        t = time.perf_counter() - start
        if t >= trajectory.duration:
            break

        new_goal, new_enable = trajectory(t)

        if new_enable != torque_enable:
            if new_enable:
                io.set_goal_position({servo: new_goal})
                io.enable_torque([servo])
            else:
                io.disable_torque([servo])
            torque_enable = new_enable
            time.sleep(CONTROL_PERIOD)

        goal_position = new_goal
        if torque_enable and t - last_control_t >= CONTROL_PERIOD:
            io.set_goal_position({servo: goal_position})
            last_control_t = t

        # Straddle the sample with clock reads so the timestamp sits in the
        # middle of the readout, as bam.process and bam.fit assume.
        t0 = time.perf_counter() - start
        entry = recorder.read_data()
        t1 = time.perf_counter() - start

        entry["timestamp"] = (t0 + t1) / 2.0
        entry["goal_position"] = float(goal_position)
        entry["torque_enable"] = bool(torque_enable)
        recorder.compute_duty_cycle(goal_position, entry["position"])
        entries.append(entry)

        if abs(entry["speed"]) > MAX_ABSOLUTE_SPEED:
            print(
                f"warning: speed {entry['speed']:.1f} rad/s exceeds the "
                f"{MAX_ABSOLUTE_SPEED} rad/s safety limit"
            )
        if abs(entry["load"]) > MAX_CURRENT_MA:
            raise RuntimeError(
                f"current {entry['load']:.0f} mA exceeds the {MAX_CURRENT_MA} mA "
                "safety limit; aborting and releasing torque"
            )

        time.sleep(CONTROL_PERIOD)

    return entries


def return_to_zero(io, servo: int, recorder: Recorder) -> None:
    """Ramp the arm back to zero, then release torque."""
    goal = io.get_present_position([servo])[0] - recorder.q_offset
    dt = 0.01
    max_step = dt * 1.0  # rad per step

    while abs(goal) > max_step:
        goal -= np.sign(goal) * max_step
        io.set_goal_position({servo: goal + recorder.q_offset})
        time.sleep(dt)

    io.set_goal_position({servo: recorder.q_offset})
    time.sleep(0.2)
    io.disable_torque([servo])


def main() -> None:
    """Run one identification trajectory and dump the log to ``--logdir``."""
    port = None if args.dry_run else pick_port(args.port)
    io = open_io(port)
    recorder = Recorder(
        io, args.id, kp=args.kp, error_gain=args.error_gain, max_pwm=args.max_pwm
    )

    trajectory = trajectories[args.trajectory]
    date = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
    filename = f"{args.logdir}/{date}.json"

    entries: list[dict] = []
    try:
        recorder.q_offset = settle_and_measure_zero(io, args.id, trajectory)
        print(
            f"servo {args.id}: zero offset {recorder.q_offset:+.4f} rad, "
            f"scale {io.scale_dd / 10.0:.1f} deg"
        )

        entries = run_trajectory(io, args.id, recorder, trajectory)
        return_to_zero(io, args.id, recorder)
    finally:
        # Always release the motor and flush whatever was captured, so an
        # interrupt still leaves a usable (truncated) log behind.
        try:
            io.disable_torque([args.id])
        except (OSError, TimeoutError) as exc:
            print(f"warning: could not disable torque: {exc}")

        data = {
            "mass": args.mass,
            "arm_mass": args.arm_mass,
            "length": args.length,
            "kp": args.kp,
            "kd": args.kd,
            "vin": args.vin,
            "motor": args.motor,
            "trajectory": args.trajectory,
            "q_offset": recorder.q_offset,
            "error_gain": args.error_gain,
            "max_pwm": args.max_pwm,
            "servo_id": args.id,
            "entries": entries,
        }
        with open(filename, "w") as logfile:
            json.dump(data, logfile)
        io.close()

    print(f"recorded {len(entries)} entries -> {filename}")


if __name__ == "__main__":
    main()
