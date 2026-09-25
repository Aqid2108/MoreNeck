import serial

from neck.bus import Reading, ServoBus, decode_reading, goal_bytes


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


class StubPort:
    is_using = True  # simulate a flag left set by an earlier failed transaction


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
