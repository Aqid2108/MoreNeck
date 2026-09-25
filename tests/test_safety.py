from neck.bus import Reading
from neck.safety import Level, check, new_history


def reading(position, **kw):
    base = dict(position=position, speed=0, load=0, voltage=7.4, temperature=30, status=0,
                moving=False, current_ma=100.0, error=0)
    base.update(kw)
    return Reading(**base)


def nominal(cfg, **per_joint):
    return {n: reading(**{"position": j.neutral, **per_joint.get(n, {})}) for n, j in cfg.joints.items()}


def chk_n(cfg, readings, n=3):
    """Run the same readings for n consecutive ticks on a fresh history; return the last verdict."""
    hist = new_history(cfg)
    for i in range(n):
        v = chk(cfg, hist, readings, now=i / 50)
    return v


def chk(cfg, hist, readings, now=0.0, setpoints=None, holding=None):
    setpoints = setpoints or {n: float(j.neutral) for n, j in cfg.joints.items()}
    holding = holding or {n: False for n in cfg.joints}
    return check(readings, setpoints, holding, cfg, hist, now)


def test_nominal_is_ok(cfg):
    v = chk(cfg, new_history(cfg), nominal(cfg))
    assert v.level == Level.OK
    assert v.reasons == []


def test_missed_reads_warn_then_hard_on_fifth(cfg):
    hist = new_history(cfg)
    r = nominal(cfg)
    r["nod"] = None
    for i in range(4):
        assert chk(cfg, hist, r, now=i).level == Level.WARN
    assert chk(cfg, hist, r, now=5).level == Level.HARD


def test_miss_counter_resets_on_reply(cfg):
    hist = new_history(cfg)
    missing = nominal(cfg)
    missing["nod"] = None
    for _ in range(4):
        chk(cfg, hist, missing)
    chk(cfg, hist, nominal(cfg))
    assert chk(cfg, hist, missing).level == Level.WARN


def test_status_bits_are_hard_after_debounce(cfg):
    assert chk_n(cfg, nominal(cfg, nod={"status": 0x20}), n=2).level == Level.OK
    assert chk_n(cfg, nominal(cfg, nod={"status": 0x20}), n=3).level == Level.HARD


def test_single_sample_temperature_spike_is_ignored(cfg):
    # Seen on hardware: nod read 28, 82, 28 C on consecutive ticks (a corrupted byte).
    hist = new_history(cfg)
    for i, temp in enumerate([28, 82, 28, 28, 28]):
        assert chk(cfg, hist, nominal(cfg, nod={"temperature": temp}), now=i / 50).level == Level.OK


def test_only_uncontrollable_faults_request_limp(cfg):
    hist = new_history(cfg)
    lost = nominal(cfg)
    lost["nod"] = None
    for i in range(5):
        v = chk(cfg, hist, lost, now=i / 50)
    assert v.level == Level.HARD and v.limp
    over = {n: reading(j.neutral, voltage=8.2) for n, j in cfg.joints.items()}
    assert chk_n(cfg, over).limp
    for hot in (nominal(cfg, nod={"temperature": 66}), nominal(cfg, nod={"status": 0x20}),
                {n: reading(j.neutral, voltage=5.9) for n, j in cfg.joints.items()}):
        v = chk_n(cfg, hot)
        assert v.level == Level.HARD and not v.limp


def test_voltage_thresholds(cfg):
    for volts, level in [(6.7, Level.WARN), (6.4, Level.SOFT), (5.9, Level.HARD), (8.1, Level.HARD),
                         (7.4, Level.OK)]:
        r = {n: reading(j.neutral, voltage=volts) for n, j in cfg.joints.items()}
        assert chk_n(cfg, r).level == level, volts


def test_temperature_thresholds_are_per_joint(make_cfg):
    cfg = make_cfg({"nod.temp_warn_c": 50, "nod.temp_soft_c": 55, "nod.temp_hard_c": 60})
    assert chk_n(cfg, nominal(cfg, nod={"temperature": 52})).level == Level.WARN
    assert chk_n(cfg, nominal(cfg, nod={"temperature": 56})).level == Level.SOFT
    assert chk_n(cfg, nominal(cfg, nod={"temperature": 60})).level == Level.HARD
    assert chk_n(cfg, nominal(cfg, turn={"temperature": 56})).level == Level.WARN


def test_current_spike_needs_three_consecutive_ticks(cfg):
    hist = new_history(cfg)
    hot = nominal(cfg, nod={"current_ma": 900.0})
    assert chk(cfg, hist, hot, now=0.00).level == Level.OK
    assert chk(cfg, hist, hot, now=0.02).level == Level.OK
    assert chk(cfg, hist, hot, now=0.04).level == Level.SOFT


def test_current_spike_counter_resets_on_dip(cfg):
    hist = new_history(cfg)
    hot = nominal(cfg, nod={"current_ma": 900.0})
    chk(cfg, hist, hot)
    chk(cfg, hist, hot)
    chk(cfg, hist, nominal(cfg))
    assert chk(cfg, hist, hot).level == Level.OK


def test_holding_mean_current_over_window(make_cfg):
    cfg = make_cfg({"nod.hold_current_limit_ma": 500})
    hist = new_history(cfg)
    r = nominal(cfg, nod={"current_ma": 600.0})
    holding = {n: True for n in cfg.joints}
    for i in range(100):                       # t = 0.00 .. 1.98 s
        assert chk(cfg, hist, r, now=i / 50, holding=holding).level == Level.OK
    assert chk(cfg, hist, r, now=100 / 50, holding=holding).level == Level.SOFT


def test_holding_window_restarts_after_motion(make_cfg):
    cfg = make_cfg({"nod.hold_current_limit_ma": 500})
    hist = new_history(cfg)
    r = nominal(cfg, nod={"current_ma": 600.0})
    hold = {n: True for n in cfg.joints}
    move = {n: False for n in cfg.joints}
    for i in range(90):
        chk(cfg, hist, r, now=i / 50, holding=hold)
    chk(cfg, hist, r, now=90 / 50, holding=move)
    for i in range(91, 140):                   # less than 2 s of continuous holding again
        assert chk(cfg, hist, r, now=i / 50, holding=hold).level == Level.OK


def test_tracking_error_must_persist(cfg):
    hist = new_history(cfg)
    turn = cfg.joints["turn"]
    r = nominal(cfg, turn={"position": turn.neutral + 100})   # ~8.8 degrees off
    assert chk(cfg, hist, r, now=0.00).level == Level.OK
    assert chk(cfg, hist, r, now=0.20).level == Level.OK
    assert chk(cfg, hist, r, now=0.32).level == Level.SOFT


def test_soft_limit_check_arms_only_after_entering_range(cfg):
    hist = new_history(cfg)
    lean = cfg.joints["lean"]
    at = lambda pos: {**nominal(cfg), "lean": reading(pos)}
    sp = lambda pos: {**{n: float(j.neutral) for n, j in cfg.joints.items()}, "lean": float(pos)}
    assert chk(cfg, hist, at(lean.raw_min - 34), setpoints=sp(lean.raw_min - 34)).level == Level.OK
    chk(cfg, hist, at(900), setpoints=sp(900))
    assert chk(cfg, hist, at(lean.raw_min - 34), setpoints=sp(lean.raw_min - 34)).level == Level.SOFT


def test_most_severe_wins_and_all_reasons_kept(cfg):
    r = nominal(cfg, nod={"temperature": 66, "voltage": 6.4})
    v = chk_n(cfg, r)
    assert v.level == Level.HARD
    assert any("temperature" in s for s in v.reasons)
    assert any("voltage" in s for s in v.reasons)
