import termios

import pytest
import serial

from neck.bus import BusError, Reading, ServoBus, decode_reading, find_port, goal_bytes


def packet(position=805, speed=0, load=0, voltage=74, temp=27, status=0, moving=0,
           current=0, error=0):
    d = [0] * 15
    d[0], d[1] = position & 0xFF, position >> 8
    d[2], d[3] = speed & 0xFF, speed >> 8
    d[4], d[5] = load & 0xFF, load >> 8
    d[6] = voltage
    d[7] = temp
    d[9] = status
    d[10] = moving
    d[13], d[14] = current & 0xFF, current >> 8
    return [error] + d


def test_decode_basic_fields():
    r = decode_reading(packet(position=3132, voltage=74, temp=26, status=0x20, moving=1, error=0x20))
    assert r == Reading(position=3132, speed=0, load=0, voltage=7.4, temperature=26,
                        status=0x20, moving=True, current_ma=0.0, error=0x20)


def test_decode_current_uses_both_bytes():
    # 384 counts = 0x0180: a low-byte-only reader would see 128 (832 mA)
    assert decode_reading(packet(current=384)).current_ma == 384 * 6.5


def test_decode_current_is_magnitude_of_sign_magnitude():
    assert decode_reading(packet(current=0x8000 | 100)).current_ma == 650.0


def test_decode_signed_speed_and_load():
    r = decode_reading(packet(speed=0x8000 | 100, load=0x0400 | 500))
    assert r.speed == -100
    assert r.load == -500


def test_goal_bytes_rounds_and_clamps():
    assert goal_bytes(805.4) == [0x25, 0x03]
    assert goal_bytes(-3) == [0, 0]
    assert goal_bytes(9999) == [0xFF, 0x0F]


class StubSerial:
    def __init__(self):
        self.input_resets = 0

    def reset_input_buffer(self):
        self.input_resets += 1


class StubPort:
    def __init__(self):
        self.is_using = True  # simulate a flag left set by an earlier failed transaction
        self.ser = StubSerial()


class StubReader:
    def __init__(self):
        self.data_dict = {1: [], 3: []}
        self.replies = {}
        self.raise_exc = None

    def txRxPacket(self):
        if self.raise_exc:
            raise self.raise_exc
        for sid, data in self.replies.items():
            self.data_dict[sid] = data
        return 0


def stub_bus():
    bus = ServoBus({"lean": 1, "nod": 3})
    bus._port_handler = StubPort()
    bus._reader = StubReader()
    return bus


def test_read_all_decodes_and_resets_busy_flag():
    bus = stub_bus()
    bus._reader.replies = {1: packet(position=900), 3: packet(position=3000)}
    out = bus.read_all()
    assert out["lean"].position == 900
    assert out["nod"].position == 3000
    assert bus._port_handler.is_using is False


def test_read_all_never_returns_stale_data():
    bus = stub_bus()
    bus._reader.replies = {1: packet(position=900), 3: packet(position=3000)}
    bus.read_all()
    bus._reader.replies = {3: packet(position=3001)}  # lean stops answering
    out = bus.read_all()
    assert out["lean"] is None
    assert out["nod"].position == 3001


def test_read_all_returns_none_on_serial_exception():
    bus = stub_bus()
    bus._reader.raise_exc = serial.SerialException("device disconnected")
    assert bus.read_all() == {"lean": None, "nod": None}


def test_read_all_discards_stale_input_bytes_first():
    bus = stub_bus()
    bus._reader.replies = {1: packet(), 3: packet()}
    bus.read_all()
    assert bus._port_handler.ser.input_resets == 1


def test_read_all_returns_none_on_termios_error():
    # macOS raises termios.error (not OSError) from tcdrain when the USB device is gone
    bus = stub_bus()
    bus._reader.raise_exc = termios.error(6, "Device not configured")
    assert bus.read_all() == {"lean": None, "nod": None}


class StubHandler:
    def __init__(self, goals=None, raise_exc=None):
        self.goals = goals or {}
        self.raise_exc = raise_exc

    def syncWriteTxOnly(self, *args):
        raise self.raise_exc

    def read2ByteTxRx(self, sid, address):
        assert address == 42
        if sid in self.goals:
            return self.goals[sid], 0, 0
        return 0, -6, 0


def test_write_goals_returns_false_on_termios_error():
    bus = stub_bus()
    bus._ph = StubHandler(raise_exc=termios.error(6, "Device not configured"))
    assert bus.write_goals({"lean": 900}) is False


def test_read_goals_reads_goal_register_per_servo():
    bus = stub_bus()
    bus._ph = StubHandler(goals={1: 905})
    assert bus.read_goals() == {"lean": 905, "nod": None}


class FakePortInfo:
    def __init__(self, device, vid, pid):
        self.device, self.vid, self.pid = device, vid, pid


def test_find_port_matches_vendor_and_product(monkeypatch):
    ports = [FakePortInfo("/dev/cu.eyes", 0x1A86, 0x7523), FakePortInfo("/dev/cu.servos", 0x1A86, 0x55D3)]
    monkeypatch.setattr("neck.bus.list_ports.comports", lambda: ports)
    assert find_port(0x1A86, 0x55D3) == "/dev/cu.servos"


def test_find_port_raises_when_adapter_absent(monkeypatch):
    monkeypatch.setattr("neck.bus.list_ports.comports", lambda: [FakePortInfo("/dev/cu.eyes", 0x1A86, 0x7523)])
    with pytest.raises(BusError, match="55d3"):
        find_port(0x1A86, 0x55D3)
