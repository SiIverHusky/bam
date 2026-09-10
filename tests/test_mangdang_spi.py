# Copyright 2026 Mangdang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline tests for the MD01 driver and bridge frame codec.

No hardware and no serial port are required: the frame tests exercise
:meth:`SpiMd01IO._read_response` against a fake serial object, and the
higher-level tests use ``dry_run=True``. Run with::

    uv run python -m pytest tests/test_mangdang_spi.py
"""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bam.mangdang.spi import (  # noqa: E402
    CMD_GET_LIVE,
    CMD_PING,
    CMD_SET_POS,
    DEFAULT_SCALE_DD,
    BridgeError,
    SpiMd01IO,
    crc16,
    pack_float,
)


class FakeSerial:
    """Minimal stand-in for ``serial.Serial`` with a scripted byte stream."""

    def __init__(self, payload: bytes):
        self.payload = bytearray(payload)

    def read(self, n: int = 1) -> bytes:
        chunk, self.payload = self.payload[:n], self.payload[n:]
        return bytes(chunk)

    def write(self, data: bytes) -> int:
        return len(data)

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


def make_response(cmd=0, servo=1, status=0, pos_dd=1550, cur_ma=0, val=0.0) -> bytes:
    """Build one well-formed 14-byte response frame."""
    body = struct.pack("<BBBBHhf", 0x62, cmd, servo, status, pos_dd, cur_ma, val)
    return body + struct.pack("<H", crc16(body))


def raw_io(payload: bytes) -> SpiMd01IO:
    """Return a driver whose serial port replays ``payload``."""
    io = SpiMd01IO("/dev/null", dry_run=True)
    io.dry_run = False  # force the real codec path
    io.ser = FakeSerial(payload)
    return io


# ---- frame codec ---------------------------------------------------------


def test_crc16_known_vector():
    """CRC16-CCITT-FALSE of "123456789" is the standard 0x29B1 check value."""
    assert crc16(b"123456789") == 0x29B1


def test_pack_float_roundtrip():
    lo, hi = pack_float(32.0)
    bits = lo | (hi << 16)
    assert struct.unpack("<f", struct.pack("<I", bits))[0] == 32.0


def test_read_response_decodes_fields():
    io = raw_io(make_response(cmd=CMD_PING, servo=3, pos_dd=1234, cur_ma=-250))
    response = io._read_response()
    assert response["cmd"] == CMD_PING
    assert response["servo"] == 3
    assert response["pos_dd"] == 1234
    assert response["cur_mA"] == -250


def test_read_response_skips_boot_banner():
    """The firmware prints '# ...' lines before switching to the bridge baud."""
    banner = b"# BAM bridge booted. Bridge baud = 921600.\r\n# ready\r\n"
    io = raw_io(banner + make_response(pos_dd=1551))
    assert io._read_response()["pos_dd"] == 1551


def test_read_response_rejects_bad_crc():
    frame = bytearray(make_response())
    frame[0] = 0x62
    frame[-1] ^= 0xFF  # corrupt the CRC
    io = raw_io(bytes(frame))
    with pytest.raises(IOError, match="CRC mismatch"):
        io._read_response()


def test_read_response_times_out_without_magic():
    io = raw_io(b"")
    with pytest.raises(TimeoutError, match="no response"):
        io._read_response()


def test_check_raises_on_error_status():
    with pytest.raises(BridgeError, match="ERR_BOARD"):
        SpiMd01IO._check({"status": 3})


# ---- unit conversions ----------------------------------------------------


def test_rad_to_dd_maps_zero_to_travel_centre():
    io = SpiMd01IO("/dev/null", scale_dd=DEFAULT_SCALE_DD, dry_run=True)
    assert io.rad_to_dd(0.0) == 1550


def test_dd_to_rad_is_inverse_of_rad_to_dd():
    io = SpiMd01IO("/dev/null", scale_dd=DEFAULT_SCALE_DD, dry_run=True)
    for rad in (-1.2, -0.3, 0.0, 0.7, 2.5):
        assert io.dd_to_rad(io.rad_to_dd(rad)) == pytest.approx(rad, abs=1e-3)


def test_rad_to_dd_clamped_to_travel():
    io = SpiMd01IO("/dev/null", scale_dd=DEFAULT_SCALE_DD, dry_run=True)
    assert io.set_pos_dd(1, -9999)["pos_dd"] == 0
    assert io.set_pos_dd(1, 99999)["pos_dd"] == int(DEFAULT_SCALE_DD)


# ---- firmware-mirroring behaviour ---------------------------------------


def test_set_pos_records_goal_for_enable_torque():
    """enable_torque must re-command the last goal, not the travel centre."""
    io = SpiMd01IO("/dev/null", dry_run=True)
    io.set_goal_position({1: 0.5})
    expected = io.rad_to_dd(0.5)
    assert io.get_present_position_dd([1])[0] == expected

    # An idle in between must not lose the goal.
    io.disable_torque([1])
    io.enable_torque([1])
    assert io.get_present_position_dd([1])[0] == expected


def test_enable_torque_without_goal_uses_centre():
    io = SpiMd01IO("/dev/null", dry_run=True)
    io.enable_torque([1])
    assert io.get_present_position_dd([1])[0] == int(DEFAULT_SCALE_DD / 2)


def test_idle_leaves_position_unchanged():
    io = SpiMd01IO("/dev/null", dry_run=True)
    io.set_goal_position({1: 0.4})
    before = io.get_present_position_dd([1])[0]
    io.disable_torque([1])
    assert io.get_present_position_dd([1])[0] == before


def test_set_mode_rejects_non_position_modes():
    io = SpiMd01IO("/dev/null", dry_run=True)
    with pytest.raises(ValueError, match="position mode"):
        io.set_mode({1: 0})


def test_voltage_is_the_constant_rail():
    io = SpiMd01IO("/dev/null", dry_run=True)
    assert io.get_present_voltage([1, 2]) == [12.0, 12.0]


def test_scan_probes_all_twelve_servos():
    io = SpiMd01IO("/dev/null", dry_run=True)
    assert io.scan() == list(range(1, 13))


def test_get_gains_returns_written_value():
    io = SpiMd01IO("/dev/null", dry_run=True)
    io.set_P_coefficient({1: 32.0})
    io.set_D_coefficient({1: 1.5})
    assert io.get_param(1, 5)["val"] == 32.0  # kp_position
    assert io.get_param(1, 6)["val"] == 1.5  # kd_position
    # Gains are per servo, not global.
    assert io.get_param(2, 5)["val"] == 0.0


# ---- recorder helpers ----------------------------------------------------


def test_recorder_duty_cycle_clips_to_max_pwm(tmp_path, monkeypatch):
    """compute_duty_cycle must not depend on record.py's module-level args."""
    # record.py parses its CLI at import time (same convention as the other
    # BAM recorders), so the module needs a plausible argv to be importable.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record.py",
            "--mass", "0.5",
            "--length", "0.15",
            "--id", "1",
            "--logdir", str(tmp_path),
        ],
    )
    record = pytest.importorskip("bam.mangdang.record")

    io = SpiMd01IO("/dev/null", dry_run=True)
    recorder = record.Recorder(io, 1, kp=32.0, error_gain=0.0104, max_pwm=0.97)

    # Saturation needs kp * error_gain * error > max_pwm, i.e. error > ~2.9 rad.
    assert recorder.compute_duty_cycle(10.0, 0.0) == pytest.approx(0.97)
    assert recorder.compute_duty_cycle(0.0, 10.0) == pytest.approx(-0.97)

    # Small error stays on the linear part of the law.
    expected = 32.0 * 0.0104 * 0.01
    assert recorder.compute_duty_cycle(0.01, 0.0) == pytest.approx(expected)

    # A 1 rad error is still linear (32 * 0.0104 = 0.3328 < 0.97).
    assert recorder.compute_duty_cycle(1.0, 0.0) == pytest.approx(0.3328)


def test_set_goal_position_unit_degrees():
    """The degree mode must place 180 deg below centre, not wrap or saturate."""
    io = SpiMd01IO("/dev/null", scale_dd=DEFAULT_SCALE_DD, dry_run=True)
    io.set_goal_position_unit("deg")
    io.set_goal_position({1: 90.0})
    assert io.get_present_position_dd([1])[0] == 1550 + 900


def test_set_goal_position_defaults_to_radians():
    io = SpiMd01IO("/dev/null", scale_dd=DEFAULT_SCALE_DD, dry_run=True)
    io.set_goal_position({1: 1.0})
    assert io.get_present_position_dd([1])[0] == pytest.approx(1550 + 573, abs=1)


def test_set_goal_position_unit_rejects_unknown():
    io = SpiMd01IO("/dev/null", dry_run=True)
    with pytest.raises(ValueError, match="unit must be"):
        io.set_goal_position_unit("gradians")
