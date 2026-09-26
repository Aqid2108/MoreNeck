"""ServoBus: the only code that touches the serial port.

Wraps python-st3215's low-level group read/write so all servos are read in one
transaction and written in one transaction per tick. Never writes EEPROM.
"""
from __future__ import annotations

import logging
import termios
from dataclasses import dataclass

import serial
from serial.tools import list_ports
from st3215.group_sync_read import GroupSyncRead
from st3215.group_sync_write import GroupSyncWrite
from st3215.port_handler import PortHandler
from st3215.protocol_packet_handler import protocol_packet_handler
from st3215.values import (
    COMM_SUCCESS, STS_ACC, STS_GOAL_POSITION_L, STS_GOAL_SPEED_L, STS_PRESENT_POSITION_L,
    STS_TORQUE_ENABLE,
)

log = logging.getLogger("neck.bus")

REG_TORQUE_LIMIT = 48          # RAM, 2 bytes, 0-1000 (not defined in st3215.values)
READ_START = STS_PRESENT_POSITION_L   # 0x38
READ_LEN = 15                  # registers 0x38..0x46
CURRENT_MA_PER_COUNT = 6.5
STREAM_ACC = 254               # servo-side ramp effectively off; controller shapes motion
STREAM_SPEED = 3400            # servo-side speed cap effectively off


# pyserial raises SerialException/OSError; on macOS a vanished USB device also raises
# termios.error (not an OSError subclass) from the library's ser.flush().
SERIAL_ERRORS = (serial.SerialException, OSError, termios.error)


class BusError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reading:
    position: int
    speed: int
    load: int
    voltage: float
    temperature: int
    status: int
    moving: bool
    current_ma: float
    error: int


def _sign_magnitude(value: int, sign_bit: int) -> int:
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if value & (1 << sign_bit) else magnitude


def decode_reading(packet: list[int]) -> Reading:
    """packet[0] = reply error byte; packet[1:16] = registers 0x38..0x46."""
    d = packet[1:]

    def u16(i):
        return d[i] | (d[i + 1] << 8)

    return Reading(
        position=u16(0),
        speed=_sign_magnitude(u16(2), 15),
        load=_sign_magnitude(u16(4), 10),
        voltage=d[6] / 10,
        temperature=d[7],
        status=d[9],
        moving=bool(d[10]),
        current_ma=abs(_sign_magnitude(u16(13), 15)) * CURRENT_MA_PER_COUNT,
        error=packet[0],
    )


def goal_bytes(raw: float) -> list[int]:
    v = min(max(int(round(raw)), 0), 4095)
    return [v & 0xFF, v >> 8]


def find_port(vid: int, pid: int) -> str:
    for p in list_ports.comports():
        if p.vid == vid and p.pid == pid:
            return p.device
    raise BusError(f"no USB serial device {vid:04x}:{pid:04x}; is the bus adapter plugged in?")


class ServoBus:
    def __init__(self, ids: dict[str, int], vid: int = 0x1A86, pid: int = 0x55D3,
                 baudrate: int = 1_000_000, port: str | None = None):
        self.ids = dict(ids)
        self._vid = vid
        self._pid = pid
        self._baudrate = baudrate
        self._port = port
        self._port_handler = None
        self._ph = None
        self._reader = None

    @classmethod
    def from_config(cls, cfg) -> "ServoBus":
        return cls({n: j.id for n, j in cfg.joints.items()}, vid=cfg.port_vid, pid=cfg.port_pid,
                   baudrate=cfg.baudrate)

    def open(self) -> None:
        port = self._port or find_port(self._vid, self._pid)
        self._port_handler = PortHandler(port)
        self._port_handler.baudrate = self._baudrate
        if not self._port_handler.openPort():
            raise BusError(f"could not open {port}")
        self._ph = protocol_packet_handler(self._port_handler)
        self._reader = GroupSyncRead(self._ph, READ_START, READ_LEN)
        for sid in self.ids.values():
            self._reader.addParam(sid)
        log.info("bus open on %s @ %d", port, self._baudrate)

    def close(self) -> None:
        if self._port_handler is not None and self._port_handler.is_open:
            self._port_handler.closePort()

    def _begin(self) -> None:
        self._port_handler.is_using = False

    def ping(self) -> set[str]:
        found = set()
        for name, sid in self.ids.items():
            self._begin()
            _, comm, _ = self._ph.ping(sid)
            if comm == COMM_SUCCESS:
                found.add(name)
        return found

    def read_all(self) -> dict[str, Reading | None]:
        for sid in self.ids.values():
            self._reader.data_dict[sid] = []
        self._begin()
        try:
            # syncReadRx reads a fixed byte count, so one stray byte would shift every later frame.
            self._port_handler.ser.reset_input_buffer()
            self._reader.txRxPacket()
        except SERIAL_ERRORS as exc:
            log.warning("group read failed: %r", exc)
            return {name: None for name in self.ids}
        out = {}
        for name, sid in self.ids.items():
            data = self._reader.data_dict.get(sid)
            out[name] = decode_reading(data) if data and len(data) == READ_LEN + 1 else None
        return out

    def _sync_write(self, address: int, length: int, per_id: dict[int, list[int]]) -> bool:
        writer = GroupSyncWrite(self._ph, address, length)
        for sid, data in per_id.items():
            writer.addParam(sid, data)
        self._begin()
        try:
            return writer.txPacket() == COMM_SUCCESS
        except SERIAL_ERRORS as exc:
            log.warning("sync write to %d failed: %r", address, exc)
            return False

    def write_goals(self, goals: dict[str, float]) -> bool:
        return self._sync_write(STS_GOAL_POSITION_L, 2,
                                {self.ids[n]: goal_bytes(raw) for n, raw in goals.items()})

    def read_goals(self) -> dict[str, int | None]:
        """Read back each servo's goal position register (acknowledged, per servo)."""
        out = {}
        for name, sid in self.ids.items():
            self._begin()
            try:
                value, comm, _ = self._ph.read2ByteTxRx(sid, STS_GOAL_POSITION_L)
            except SERIAL_ERRORS as exc:
                log.warning("goal read from %s failed: %r", name, exc)
                value, comm = 0, -1
            out[name] = value if comm == COMM_SUCCESS else None
        return out

    def set_torque(self, on: bool) -> bool:
        return self._sync_write(STS_TORQUE_ENABLE, 1, {sid: [1 if on else 0] for sid in self.ids.values()})

    def _write_verified(self, address: int, value: int, two_bytes: bool) -> bool:
        """Acknowledged per-servo write plus read-back. Broadcast settings can be lost silently
        (on the robot, "acceleration 254" never reached nod and it kept an old value)."""
        ok = True
        for name, sid in self.ids.items():
            write = self._ph.write2ByteTxRx if two_bytes else self._ph.write1ByteTxRx
            read = self._ph.read2ByteTxRx if two_bytes else self._ph.read1ByteTxRx
            try:
                self._begin()
                write(sid, address, value)
                self._begin()
                got, comm, _ = read(sid, address)
            except SERIAL_ERRORS as exc:
                log.warning("setting register %d on %s failed: %r", address, name, exc)
                got, comm = None, -1
            if comm != COMM_SUCCESS or got != value:
                log.error("%s: register %d is %s, expected %d", name, address, got, value)
                ok = False
        return ok

    def set_torque_limit(self, fraction: float) -> bool:
        v = int(round(min(max(fraction, 0.0), 1.0) * 1000))
        return self._write_verified(REG_TORQUE_LIMIT, v, two_bytes=True)

    def configure_for_streaming(self) -> bool:
        acc = self._write_verified(STS_ACC, STREAM_ACC, two_bytes=False)
        spd = self._write_verified(STS_GOAL_SPEED_L, STREAM_SPEED, two_bytes=True)
        return acc and spd

    def read_u16(self, name: str, address: int) -> int | None:
        self._begin()
        value, comm, _ = self._ph.read2ByteTxRx(self.ids[name], address)
        return value if comm == COMM_SUCCESS else None
