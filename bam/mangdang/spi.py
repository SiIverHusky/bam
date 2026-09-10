# Copyright 2026 Mangdang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Low-level driver for the Mangdang MD01 servo-actuator.

The MD01 is not driven directly: it sits behind an AT32F413 driver board that
speaks SPI, and the boards are reached through a small ESP32-S3 bridge that
exposes them on a USB serial port. This module implements the host half of that
bridge protocol and is the only place in BAM that knows about deci-degrees,
frame layouts or CRC16.

Protocol (see ``bam-testfirmware/main/bam_bridge.c`` for the firmware half and
``bam-testfirmware/tools/bam_bridge.py`` for the reference implementation this
file is ported from)::

    request   10 bytes: 42 cmd servo 00 arg1:u16 arg2:u16 crc:u16
    response  14 bytes: 62 cmd servo status pos_dd:u16 cur_mA:i16 val:f32 crc:u16

Everything on the wire is little-endian and CRC16-CCITT-FALSE is computed over
every byte but the trailing CRC itself. Servo ids run from 1 to 12 and board
``(id - 1) // 3`` owns each servo.

Units: the AT32 reports positions in **deci-degrees** (0.1 deg per count) over
``0 .. scale_dd``. The scale must equal ``range_position_deg * 10`` as configured
on the AT32 (default ``3100`` = 310.0 deg) and its centre is the middle of the
potentiometer travel. :class:`SpiMd01IO` exposes that raw world, while its
BAM-shaped methods convert to radians with ``0.0`` at the centre of travel.

Instantiating with ``dry_run=True`` returns a driver that answers every command
with plausible synthetic data and never touches a serial port, so the recorder
can be exercised without hardware (see ``record.py --dry-run``).
"""

from __future__ import annotations

import math
import struct
import time

MAGIC_REQ = 0x42
MAGIC_RSP = 0x62

CMD_PING = 0
CMD_SET_POS = 1
CMD_IDLE = 2
CMD_TORQUE = 3
CMD_SET_KP = 4
CMD_SET_KD = 5
CMD_GET_PARAM = 6
CMD_GET_LIVE = 7
CMD_SET_SCALE = 8
CMD_RAW_REPLY = 9

BAM_OK = 0

#: Human-readable status codes returned by the firmware.
STATUS = {
    0: "OK",
    1: "ERR_SERVO",
    2: "ERR_ARG",
    3: "ERR_BOARD",
    4: "ERR_CRC",
    5: "ERR_CMD",
}

#: AT32 ``sms_config`` parameter ids (mirrors ``driver_board.h``).
PARAM = {
    "reverse_position_sensor": 0,
    "min_position_adc": 1,
    "max_position_adc": 2,
    "range_position_deg": 3,
    "reverse_motor": 4,
    "kp_position": 5,
    "kd_position": 6,
    "kp_current": 7,
    "kff_current": 8,
    "max_pwm_duty_cycle": 9,
}
PARAM_BY_ID = {v: k for k, v in PARAM.items()}

#: AT32 live control-loop value ids (mirrors ``driver_board.h``).
LIVE = {
    "pos_adc": 0,
    "cur_adc": 1,
    "setpoint_pos_deg": 2,
    "present_pos_deg": 3,
    "error_pos_deg": 4,
    "max_current_mA": 5,
    "setpoint_cur_mA": 6,
    "present_cur_mA": 7,
    "error_cur_mA": 8,
    "pwm_duty": 9,
    "mode": 10,
    "loop_counter": 11,
}
LIVE_BY_ID = {v: k for k, v in LIVE.items()}

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

#: ``range_position_deg * 10`` on the AT32 (310.0 deg). Kept in sync with the
#: board by :meth:`SpiMd01IO.sync_scale_from_board`.
DEFAULT_SCALE_DD = 3100.0

#: Current cap [mA] used by the position commands. In position mode the AT32
#: reads the torque field as a *cap*, not a forced draw.
DEFAULT_CUR_MA = 900

#: Servo ids addressable through the bridge (4 boards x 3 channels).
SERVO_IDS = tuple(range(1, 13))

#: Nominal servo rail. The driver board has no ADC, so this is a constant.
DEFAULT_VIN = 12.0


def crc16(data: bytes) -> int:
    """CRC16-CCITT-FALSE, init ``0xFFFF``, poly ``0x1021``, MSB-first."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def pack_float(value: float) -> tuple[int, int]:
    """Pack a float into the two 16-bit argument words the firmware expects."""
    return struct.unpack("<HH", struct.pack("<f", value))


class BridgeError(IOError):
    """Raised when the firmware answers with a non-OK status."""


def pick_port(port: str | None) -> str:
    """Return ``port`` if given, else auto-detect a single USB-serial port.

    USB bridges (CH340K, CP210x, FTDI) are preferred over the virtual ports that
    sometimes clutter ``/dev`` under Linux.

    :param port: Explicit device path, or ``None`` to auto-detect.
    :returns: The device path to open.
    :raises SystemExit: If no port, or several candidate ports, are found.
    """
    if port:
        return port
    try:
        from serial.tools import list_ports
    except ImportError:
        raise SystemExit("pyserial is not installed. Run: pip install pyserial")

    ports = list(list_ports.comports())
    usb = [
        p
        for p in ports
        if (p.vid is not None)
        or "USB" in (p.device or "").upper()
        or "USB" in (p.description or "").upper()
    ]
    candidates = usb or ports

    if not candidates:
        raise SystemExit(
            "No serial port found. Is the bridge plugged in? "
            "Pass --port explicitly, e.g. --port /dev/ttyUSB0"
        )
    if len(candidates) > 1:
        print("Several serial ports found:")
        for p in candidates:
            print(f"  {p.device}  ({p.description})")
        raise SystemExit(
            "Re-run with --port <one of the above>, e.g. --port %s" % candidates[0].device
        )
    return candidates[0].device


class SpiMd01IO:
    """Host-side driver for one or more MD01s behind the ESP32 SPI bridge.

    The object is a thin, stateful wrapper around the serial link. It caches the
    last commanded goal per servo so that :meth:`enable_torque` can re-command
    the same goal (which is how the firmware re-enters position mode), and it
    keeps the previous sample to derive a velocity by differentiation.

    :param port: Serial device path (e.g. ``/dev/ttyUSB0``).
    :param baud: Bridge baud rate; must match ``CONFIG_BAM_UART_BAUD``.
    :param timeout: Per-response timeout [s].
    :param scale_dd: Full-scale travel in deci-degrees. Must equal
        ``range_position_deg * 10`` on the AT32 boards. Call
        :meth:`sync_scale_from_board` once connected to adopt the board value.
    :param dry_run: If ``True``, never open the serial port and synthesise
        responses instead (used for offline testing of the recorder).
    """

    def __init__(
        self,
        port: str,
        baud: int = 921600,
        timeout: float = 1.0,
        scale_dd: float = DEFAULT_SCALE_DD,
        dry_run: bool = False,
    ):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.scale_dd = float(scale_dd)
        self.dry_run = dry_run

        #: Last commanded deci-degrees per servo, used by enable_torque.
        self._goal_dd: dict[int, int] = {}
        #: Last (time, pos_dd) sample, used by read_present_velocity.
        self._prev_t = 0.0
        self._prev_dd: int | None = None
        self._velocity = 0.0

        self.ser = None
        if not dry_run:
            import serial  # imported lazily so dry-run needs no pyserial

            self.ser = serial.Serial(port, baud, timeout=timeout)
            time.sleep(0.2)  # let the firmware finish booting
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

    # ---- lifecycle -------------------------------------------------------
    def close(self) -> None:
        """Release the serial port. Safe to call more than once."""
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def __enter__(self) -> SpiMd01IO:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- transport -------------------------------------------------------
    def _request(self, cmd: int, servo: int = 0, arg1: int = 0, arg2: int = 0) -> dict:
        """Send one command frame and return the decoded response.

        A boot banner byte or a lost frame can desync the stream once, so the
        frame is written a second time before giving up; resyncing from a fresh
        write always recovers.

        :raises TimeoutError: If no response arrives.
        :raises BridgeError: If the firmware answers with a non-OK status.
        """
        if self.dry_run:
            return self._dry_response(cmd, servo, arg1, arg2)

        head = struct.pack(
            "<BBBBHH", MAGIC_REQ, cmd, servo, 0, arg1 & 0xFFFF, arg2 & 0xFFFF
        )
        frame = head + struct.pack("<H", crc16(head))

        last_err: Exception | None = None
        for _ in range(2):
            self.ser.write(frame)
            try:
                return self._check(self._read_response())
            except (TimeoutError, OSError) as exc:
                last_err = exc
                self.ser.reset_input_buffer()
        raise last_err  # type: ignore[misc]

    def _read_response(self) -> dict:
        """Read and validate one 14-byte response frame."""
        buf = bytearray()
        deadline = time.monotonic() + self.timeout

        # Skip the firmware's human-readable "# ..." boot lines.
        while time.monotonic() < deadline:
            byte = self.ser.read(1)
            if not byte:
                continue
            if byte[0] == MAGIC_RSP:
                buf += byte
                break
            if byte[0] == ord("#"):
                while time.monotonic() < deadline:
                    c = self.ser.read(1)
                    if c in (b"", b"\n"):
                        break

        if not buf:
            raise TimeoutError("no response (wrong port? wrong firmware?)")

        while len(buf) < 14 and time.monotonic() < deadline:
            buf += self.ser.read(14 - len(buf))
        if len(buf) < 14:
            raise TimeoutError("short response: %d bytes" % len(buf))

        raw = bytes(buf[:14])
        _magic, cmd, servo, status, pos_dd, cur_ma, val, crc = struct.unpack(
            "<BBBBHhfH", raw
        )
        if crc16(raw[:12]) != crc:
            raise OSError("response CRC mismatch")
        return {
            "cmd": cmd,
            "servo": servo,
            "status": status,
            "pos_dd": pos_dd,
            "cur_mA": cur_ma,
            "val": val,
        }

    @staticmethod
    def _check(response: dict) -> dict:
        """Raise :class:`BridgeError` when the firmware reported a failure."""
        if response["status"] != BAM_OK:
            raise BridgeError(
                "firmware error: %s" % STATUS.get(response["status"], response["status"])
            )
        return response

    # ---- raw firmware operations (mirror the C command set) --------------
    def ping(self, servo: int) -> dict:
        """Refresh one servo's feedback without changing its setpoint."""
        return self._request(CMD_PING, servo)

    def set_pos_dd(self, servo: int, pos_dd: int, cur_ma: int = DEFAULT_CUR_MA) -> dict:
        """Enter position mode and command a raw deci-degree setpoint.

        :param pos_dd: Raw AT32 position, clamped to ``0 .. scale_dd``.
        :param cur_ma: Maximum current the firmware may apply [mA].
        """
        pos_dd = max(0, min(round(self.scale_dd), round(pos_dd)))
        self._goal_dd[servo] = pos_dd
        return self._request(CMD_SET_POS, servo, pos_dd, cur_ma)

    def set_idle(self, servo: int) -> dict:
        """Motor off: the AT32 enters idle mode and the output goes free."""
        return self._request(CMD_IDLE, servo)

    def set_torque_ma(self, servo: int, cur_ma: int) -> dict:
        """Raw current mode, signed mA (spin tests / identification)."""
        return self._request(CMD_TORQUE, servo, cur_ma & 0xFFFF)

    def set_p_gain(self, servo: int, kp: float) -> dict:
        """Write the firmware ``kp_position``."""
        lo, hi = pack_float(float(kp))
        return self._request(CMD_SET_KP, servo, lo, hi)

    def set_d_gain(self, servo: int, kd: float) -> dict:
        """Write the firmware ``kd_position``."""
        lo, hi = pack_float(float(kd))
        return self._request(CMD_SET_KD, servo, lo, hi)

    def get_param(self, servo: int, param_id: int) -> dict:
        """Read one ``sms_config`` parameter from the AT32."""
        return self._request(CMD_GET_PARAM, servo, param_id)

    def get_live(self, servo: int, live_id: int) -> dict:
        """Read one live control-loop value from the AT32."""
        return self._request(CMD_GET_LIVE, servo, live_id)

    def set_scale_dd(self, scale_dd: float) -> float:
        """Set the host-side and firmware-side full-scale travel [deci-deg]."""
        response = self._request(CMD_SET_SCALE, 0, int(scale_dd))
        self.scale_dd = float(response["val"])
        return self.scale_dd

    def raw_reply(self, servo: int) -> dict:
        """Diagnostic: byte statistics of the 36 raw MISO bytes.

        ``ff_count == 36`` means no board is driving MISO (line pulled up, i.e.
        board absent or unpowered); a varying count means the line is floating.
        """
        response = self._request(CMD_RAW_REPLY, servo)
        packed = struct.unpack("<I", struct.pack("<f", response["val"]))[0]
        return {
            "ff_count": packed & 0xFF,
            "b0": (packed >> 8) & 0xFF,
            "b1": (packed >> 16) & 0xFF,
            "b2": (packed >> 24) & 0xFF,
        }

    def sync_scale_from_board(self, servo: int) -> float:
        """Adopt the AT32 ``range_position_deg`` as the host full-scale."""
        reading = self.get_param(servo, PARAM["range_position_deg"])
        return self.set_scale_dd(reading["val"] * 10.0)

    # ---- unit helpers ----------------------------------------------------
    def rad_to_dd(self, rad: float) -> int:
        """Convert a joint angle [rad] to raw deci-degrees.

        ``0.0 rad`` maps to the centre of the potentiometer travel.
        """
        return round(rad * RAD2DEG * 10.0 + self.scale_dd / 2.0)

    def dd_to_rad(self, dd: float) -> float:
        """Convert raw deci-degrees to a joint angle [rad], zero at centre."""
        return (dd - self.scale_dd / 2.0) * 0.1 * DEG2RAD

    def dd_to_deg(self, dd: float) -> float:
        """Convert raw deci-degrees to degrees, zero at centre."""
        return (dd - self.scale_dd / 2.0) * 0.1

    # ---- BAM-driver-shaped API (what record.py calls) --------------------
    def set_mode(self, mode: dict) -> None:
        """Mirror the Feetech driver's ``set_mode``.

        The bridge has no explicit mode register: the mode is implied by the
        frame that is sent (position / torque / idle). Only position mode (the
        BAM recorder's mode) is accepted here; anything else is rejected so a
        silent mismatch cannot slip through.

        :param mode: Mapping ``{servo_id: mode}``.
        """
        for value in mode.values():
            if value not in (1, "position"):
                raise ValueError(
                    "SpiMd01IO.set_mode only supports position mode (1); the "
                    "bridge infers idle/torque from the command that is sent"
                )
        self._mode_set = True

    def enable_torque(self, ids) -> None:
        """Re-command each servo's last goal, which powers the motor."""
        for servo in ids:
            goal = self._goal_dd.get(servo, int(self.scale_dd / 2))
            self.set_pos_dd(servo, goal)

    def disable_torque(self, ids) -> None:
        """Enter idle mode on each servo (motor off, output free)."""
        for servo in ids:
            self.set_idle(servo)

    def set_goal_position(self, goal: dict) -> None:
        """Command joint angles.

        :param goal: Mapping ``{servo_id: angle}``. Angles are radians unless
            :meth:`set_goal_position_unit` selected degrees.
        """
        for servo, angle in goal.items():
            radians = angle if self._goal_unit == "rad" else angle * DEG2RAD
            self.set_pos_dd(servo, self.rad_to_dd(radians))

    def set_goal_position_unit(self, unit: str) -> None:
        """Select the unit of :meth:`set_goal_position` (``"rad"`` or ``"deg"``).

        .. note::

            BAM records everything in radians, so the default is ``"rad"``. The
            Feetech driver speaks degrees; pass ``"deg"`` when porting code that
            follows that convention.
        """
        if unit not in ("rad", "deg"):
            raise ValueError("unit must be 'rad' or 'deg'")
        self._goal_unit = unit

    def set_P_coefficient(self, gain: dict) -> None:
        """Write ``kp_position`` on each servo.

        :param gain: Mapping ``{servo_id: kp}``.
        """
        for servo, kp in gain.items():
            self.set_p_gain(servo, kp)

    def set_D_coefficient(self, gain: dict) -> None:
        """Write ``kd_position`` on each servo.

        :param gain: Mapping ``{servo_id: kd}``.
        """
        for servo, kd in gain.items():
            self.set_d_gain(servo, kd)

    def get_present_position(self, ids) -> list[float]:
        """Read joint angles [rad] (``0.0`` at the centre of travel)."""
        return [self.dd_to_rad(self.ping(servo)["pos_dd"]) for servo in ids]

    def get_present_position_dd(self, ids) -> list[int]:
        """Read raw deci-degrees, close to what the AT32 reports."""
        return [int(self.ping(servo)["pos_dd"]) for servo in ids]

    def get_present_speed(self, ids) -> list[float]:
        """Read velocities [rad/s], differentiated on the host.

        The AT32 has no velocity register, so this is a finite difference of the
        position feedback. Unlike :meth:`read_present_velocity` the raw
        difference is returned (no smoothing): the recorder applies its own.
        """
        velocities = []
        for servo in ids:
            response = self.ping(servo)
            t = time.monotonic()
            dd = response["pos_dd"]
            if self._prev_dd is not None and t > self._prev_t:
                velocities.append(
                    (dd - self._prev_dd) * 0.1 * DEG2RAD / (t - self._prev_t)
                )
                self._prev_dd, self._prev_t = dd, t
            else:
                self._prev_dd, self._prev_t = dd, t
                velocities.append(0.0)
        return velocities

    def get_present_current(self, ids) -> list[int]:
        """Read signed motor currents [mA]."""
        return [int(self.ping(servo)["cur_mA"]) for servo in ids]

    def read_sample(self, servo: int) -> dict:
        """Read position, current and velocity in a single SPI transaction.

        Every firmware response already carries both the position and the
        current, so calling :meth:`get_present_position`,
        :meth:`get_present_current` and :meth:`get_present_speed` separately
        would issue three redundant round-trips. This method issues one, and
        differentiates the position against the previous call to obtain the
        velocity. The recorder uses it on its hot path.

        :param servo: Servo id (1..12).
        :returns: Dict with ``pos_dd`` (raw), ``position`` [rad],
            ``current_mA``, ``speed`` [rad/s] and ``t`` (monotonic timestamp
            taken as soon as the response was decoded).
        """
        response = self.ping(servo)
        t = time.monotonic()
        dd = response["pos_dd"]

        speed = 0.0
        if self._prev_dd is not None and t > self._prev_t:
            speed = (dd - self._prev_dd) * 0.1 * DEG2RAD / (t - self._prev_t)
        self._prev_dd, self._prev_t = dd, t

        return {
            "pos_dd": int(dd),
            "position": self.dd_to_rad(dd),
            "current_mA": int(response["cur_mA"]),
            "speed": speed,
            "t": t,
        }

    def get_present_voltage(self, ids) -> list[float]:
        """Read supply voltages [V].

        The AT32 has no rail ADC: it is the bench supply (12 V default), so a
        constant is reported instead of a fake measurement.
        """
        return [DEFAULT_VIN for _ in ids]

    def get_present_temperature(self, ids) -> list[float]:
        """Temperature is not exposed by the board; kept for API symmetry."""
        return [0.0 for _ in ids]

    def get_present_pwm_duty(self, ids) -> list[float]:
        """Read the applied PWM duty cycle in ``[-1, 1]``.

        One ``DB_LIVE_PWM_DUTY`` transaction per servo (slower than a ping).
        """
        return [float(self.get_live(servo, LIVE["pwm_duty"])["val"]) for servo in ids]

    def read_present_velocity(self, servo: int, alpha: float = 0.2) -> float:
        """Velocity [rad/s] via host-side differentiation, exponentially smoothed.

        :param alpha: Per-sample EMA gain (``0.2`` is a mild low-pass).
        """
        response = self.ping(servo)
        t = time.monotonic()
        dd = response["pos_dd"]
        if self._prev_dd is not None and t > self._prev_t:
            raw = (dd - self._prev_dd) * 0.1 * DEG2RAD / (t - self._prev_t)
            self._velocity += alpha * (raw - self._velocity)
        self._prev_dd, self._prev_t = dd, t
        return self._velocity

    def scan(self) -> list[int]:
        """Probe every servo id and return those whose board answers."""
        found = []
        for servo in SERVO_IDS:
            try:
                self.ping(servo)
            except (OSError, TimeoutError):
                continue
            found.append(servo)
        return found

    # ---- synthetic responses (dry-run only) ------------------------------
    def _dry_response(self, cmd: int, servo: int, arg1: int, arg2: int) -> dict:
        """Fabricate a response for offline testing; never touches hardware.

        A position command is echoed back as the measured position, so the
        recorder sees a step response rather than a static readout, and written
        parameters are remembered so ``get_param`` round-trips like the real
        board.
        """
        if not hasattr(self, "_dry_state"):
            self._dry_state: dict = {"goals": {}, "params": {}}

        goals = self._dry_state["goals"]
        params = self._dry_state["params"]
        pos_dd = goals.setdefault(servo, int(self.scale_dd / 2))
        value = 0.0

        if cmd == CMD_SET_POS:
            pos_dd = max(0, min(int(self.scale_dd), arg1))
            goals[servo] = pos_dd
        elif cmd in (CMD_SET_KP, CMD_SET_KD):
            bits = arg1 | (arg2 << 16)
            stored = struct.unpack("<f", struct.pack("<I", bits))[0]
            params[(servo, PARAM["kp_position" if cmd == CMD_SET_KP else "kd_position"])] = stored
            value = stored
        elif cmd == CMD_GET_PARAM:
            value = params.get((servo, arg1), 0.0)
        elif cmd == CMD_GET_LIVE:
            value = 0.0

        return {
            "cmd": cmd,
            "servo": servo,
            "status": BAM_OK,
            "pos_dd": pos_dd,
            "cur_mA": 0,
            "val": value,
        }


# ``_goal_unit`` is set here rather than in ``__init__`` so subclasses that
# override ``__init__`` still get a sane default.
SpiMd01IO._goal_unit = "rad"
