"""Hardware tests. Run ONLY on request, user at the robot: pytest -m hardware -v"""
import pytest

from neck.bus import REG_TORQUE_LIMIT, ServoBus
from neck.config import load_config

pytestmark = pytest.mark.hardware


@pytest.fixture
def bus():
    b = ServoBus.from_config(load_config())
    b.open()
    yield b
    b.close()


def test_all_joints_answer_ping(bus):
    assert bus.ping() == {"turn", "nod", "lean"}


def test_read_all_returns_sane_readings(bus):
    readings = bus.read_all()
    assert set(readings) == {"turn", "nod", "lean"}
    for name, r in readings.items():
        assert r is not None, name
        assert 0 <= r.position <= 4095
        assert 6.5 <= r.voltage <= 8.0
        assert 10 <= r.temperature <= 60
        assert r.status == 0


def test_torque_limit_register_address_is_correct(bus):
    # At power-up RAM torque limit (48) is loaded from EEPROM max torque, which read 1000 on all servos.
    # If this fails, STOP: REG_TORQUE_LIMIT is wrong and rest() would write the wrong register.
    for name in ("turn", "nod", "lean"):
        assert bus.read_u16(name, REG_TORQUE_LIMIT) == 1000, name


def test_goal_register_reads_back(bus):
    # Startup verifies goal == present via this read before enabling torque.
    goals = bus.read_goals()
    assert set(goals) == {"turn", "nod", "lean"}
    for name, g in goals.items():
        assert g is not None and 0 <= g <= 4095, name
