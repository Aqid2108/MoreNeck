from behavior.engine import State
from tests.behavior_sim import Sim, SimPerson


def test_centres_a_still_person_without_overshoot(preset):
    sim = Sim(preset, [SimPerson(1, yaw=25.0)])
    sim.run(2.0)
    assert abs(sim.turn() - 25.0) < 2.0
    assert max(pose[0] for _, _, pose, _ in sim.log) < 27.0


def test_follows_a_walking_person_with_small_lag(preset):
    walker = SimPerson(1, yaw=0.0, yaw_rate=20.0)
    sim = Sim(preset, [walker])
    sim.run(1.5)
    lags = []
    for _ in range(20):
        sim.run(0.05)
        lags.append(abs(walker.yaw - sim.turn()))
    assert max(lags) < 12.0          # smoothness traded for lag (was 8; robot shuddered)


def test_switches_only_after_the_hold_time(preset):
    a = SimPerson(1, yaw=-10.0, facing_score=0.9)
    b = SimPerson(2, yaw=15.0, facing_score=0.3)
    sim = Sim(preset, [a, b])
    sim.run(2.0)
    assert sim.engine.attention.target_id == 1
    a.facing_score, b.facing_score = 0.2, 1.0
    sim.run(1.3)                                    # t = 3.3
    assert sim.engine.attention.target_id == 1
    sim.run(0.7)                                    # t = 4.0
    assert sim.engine.attention.target_id == 2


def test_newcomer_glance_returns_to_the_target(preset):
    sim = Sim(preset, [SimPerson(1, yaw=0.0)])
    sim.run(3.0)
    sim.people.append(SimPerson(3, yaw=20.0, first_seen=sim.t))
    sim.run(0.6)
    assert sim.engine.state == State.GLANCING and sim.turn() > 5.0
    sim.run(1.9)                                    # t = 5.5
    assert sim.engine.state == State.TRACKING and sim.engine.attention.target_id == 1
    assert abs(sim.turn()) < 3.0


def test_lost_then_idle_returns_to_neutral(preset):
    sim = Sim(preset, [SimPerson(1, yaw=20.0)])
    sim.run(3.0)
    sim.people.clear()
    sim.run(1.5)
    assert sim.engine.state == State.LOST and abs(sim.turn() - 20.0) < 2.5
    sim.run(2.0)                                    # past lost_hold_s (2 s)
    assert sim.engine.state == State.IDLE and abs(sim.turn()) < 2.0


def test_curious_scans_and_finds_someone_out_of_view(make_preset):
    curious = make_preset({"idle_scan": True, "follow_gain": 0.5, "glance_movement": True})
    sim = Sim(curious, [SimPerson(1, yaw=50.0)])
    sim.run(4.0)
    assert sim.engine.state == State.IDLE           # nobody visible yet
    sim.run(4.0)                                    # t = 8: scanning started at 5 s
    assert any(state == State.TRACKING for _, state, _, _ in sim.log)
    sim.run(2.0)
    assert abs(sim.turn() - 50.0) < 3.0


def test_neck_fault_pauses_until_it_runs_again(preset):
    sim = Sim(preset, [SimPerson(1, yaw=10.0)])
    sim.run(2.0)
    sim.neck.running = False
    sim.run(1.0)
    assert sim.engine.state == State.PAUSED
    looks = len(sim.neck.looks)
    sim.run(1.0)
    assert len(sim.neck.looks) == looks
    sim.neck.running = True
    sim.run(2.0)
    assert sim.engine.state == State.TRACKING and len(sim.neck.looks) > looks


def test_shy_looks_away_when_stared_at_then_comes_back(make_preset):
    shy = make_preset({"avert_after_s": 4.0, "glance_newcomer": False, "lean_deg": 0.0})
    sim = Sim(shy, [SimPerson(1, yaw=0.0, facing_score=0.9)])
    sim.run(4.6)
    assert sim.engine.state == State.AVERTING and abs(sim.turn()) >= 10.0
    sim.run(2.4)                                    # t = 7.0, averting lasted 2 s
    assert sim.engine.state == State.TRACKING and abs(sim.turn()) < 3.0


def test_camera_failure_behaves_like_nobody_there(preset):
    sim = Sim(preset, [SimPerson(1, yaw=20.0)])
    sim.run(3.0)
    sim.camera.failed = True
    sim.run(1.0)
    assert sim.engine.state == State.LOST
    sim.run(2.0)
    assert sim.engine.state == State.IDLE and abs(sim.turn()) < 2.0


def test_target_beyond_turn_limit_does_not_wind_up(preset):
    p = SimPerson(1, yaw=50.0)
    sim = Sim(preset, [p])
    sim.run(2.0)
    p.yaw = 75.0                                    # beyond the 60 deg limit, still in view at 60
    sim.run(3.0)
    assert sim.turn() <= 60.0
    p.yaw = 30.0
    sim.run(2.0)
    assert abs(sim.turn() - 30.0) < 3.0


def test_person_below_level_keeps_nod_at_zero(preset):
    sim = Sim(preset, [SimPerson(1, yaw=10.0, pitch=-10.0)])
    sim.run(3.0)
    assert sim.neck.pos["nod"] == 0.0
    assert sim.engine.state == State.TRACKING and abs(sim.turn() - 10.0) < 2.0


def test_following_a_walker_glides_instead_of_stop_and_go(preset):
    sim = Sim(preset, [SimPerson(1, yaw=-20.0, yaw_rate=10.0)])
    sim.camera.noise_deg = 1.9                      # as measured on the robot
    sim.run(4.0)
    assert sim.neck.stop_go_rate() < 2.0            # robot today: 5.1/s


def test_holding_on_a_still_noisy_person_stays_calm(preset):
    sim = Sim(preset, [SimPerson(1, yaw=15.0)])
    sim.camera.noise_deg = 1.9
    sim.run(4.0)
    assert sim.neck.stop_go_rate() < 2.0
    assert abs(sim.turn() - 15.0) < 3.0
