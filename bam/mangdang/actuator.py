# Copyright 2026 Mangdang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from bam.actuator import VoltageControlledActuator
from bam.parameter import Parameter
from bam.testbench import Testbench


class MD01Actuator(VoltageControlledActuator):
    """Mangdang MD01 voltage-controlled servo-actuator.

    Uses the standard BAM voltage-controlled control law
    (``duty_cycle = clip(kp * error_gain * Δq, -max_pwm, max_pwm)``) and the
    DC motor torque equation with back-EMF. The MD01 is reached through the
    AT32F413 driver board behind the ESP32-S3 SPI bridge implemented in
    :mod:`bam.mangdang.spi`; :mod:`bam.mangdang.record` is the matching
    recorder.

    The bridge firmware also exposes a derivative gain (``kd_position``), but
    BAM's voltage-controlled model has no damping term, so ``kd`` is recorded
    as ``0`` by default and is not fitted here.

    .. note::

        Values marked ``TODO`` must be measured on the bench and/or read from
        the MD01 datasheet before this actuator can be identified. See
        ``ADDING_A_MOTOR.md`` for where each quantity comes from.
    """

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            # Supply voltage [V] — nominal MD01 bus voltage. The driver board
            # has no rail ADC, so recordings carry this constant; override it
            # with the bench supply's real output when assembling a dataset.
            vin=12.0,
            # Default firmware P-gain, overridden per log by ``load_log``.
            kp=80.0,
            # Converts kp * Δq into a duty cycle in [-1, 1]. Measured with an
            # oscilloscope (ADDING_A_MOTOR.md §3.2); the recorder starts from
            # the same value and passes it through as log metadata.
            error_gain=0.0104,
            # Maximum duty-cycle magnitude; inherited from the measurements on
            # the other BAM voltage-controlled servos — TODO(md01): measure.
            max_pwm=1.0,
            # Firmware current limit [A]. The AT32 reads the torque field of a
            # position command as a max current cap, so this is a real cap; it
            # is conservative until the MD01 rating is confirmed.
            max_current=1.4,
        )

    def initialize(self):
        # Torque constant [Nm/A] or [V/(rad/s)] — TODO: datasheet.
        self.model.kt = Parameter(0.223, 0.0, 1.0) 

        # Motor resistance [Ohm]; often estimable as vin / I_stall — TODO.
        self.model.R = Parameter(8.5, 7.5, 9.5)  # TODO

        # Rotor / apparent inertia [kg m^2] — TODO: datasheet or fit seed.
        self.model.armature = Parameter(2.25e-4, 1.1e-4, 6.8e-4) 

        # Optional: fit a ratio on top of error_gain (see ST3025Actuator).
        # self.model.error_gain_ratio = Parameter(1.0, 0.1, 10.0)

    def load_log(self, log: dict):
        """Load per-log settings, tolerating the recorder's extra metadata.

        :class:`~bam.actuator.DCMotorActuator` reads ``kp`` and ``vin``; the
        MD01 recorder also writes ``error_gain`` and ``max_pwm``, which take
        precedence over the class defaults when present, so a dataset stays
        self-consistent with how it was recorded.
        """
        super().load_log(log)
        if "error_gain" in log:
            self.error_gain = log["error_gain"]
        if "max_pwm" in log:
            self.max_pwm = log["max_pwm"]

    def get_extra_inertia(self) -> float:
        return self.model.armature.value
