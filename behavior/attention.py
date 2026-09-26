"""Who to look at: scoring, stickiness with a sustained-lead switch rule, and glances."""
from __future__ import annotations

def closeness(distance_m, preset) -> float:
    if distance_m is None:
        return 0.0
    span = preset.near_zero_m - preset.near_full_m
    return min(1.0, max(0.0, (preset.near_zero_m - distance_m) / span))


def score(person, preset, is_target: bool = False) -> float:
    s = (preset.w_facing * person.facing_score
         + preset.w_near * closeness(person.distance_m, preset)
         + preset.w_approach * (1.0 if person.motion == "approaching" else 0.0))
    if not person.visible:
        s *= preset.coast_factor
    if is_target:
        s += preset.stickiness
    return s


class Attention:
    def __init__(self, preset):
        self.p = preset
        self.target_id: int | None = None
        self._challenger: int | None = None
        self._challenger_since = 0.0
        self._last_glance_at = float("-inf")
        self._low_since: float | None = None      # when the current target's confidence went low

    def _eligible(self, people):
        return [x for x in people if x.confidence >= self.p.min_confidence]

    def update(self, people, now: float, can_acquire: bool = True) -> int | None:
        """can_acquire=False: keep following the current target, but take on nobody new."""
        eligible = self._eligible(people)
        # The current target stays while it is present at all: the confidence cutoff applies only
        # to acquiring and challenging, so a one-frame detector dip can't hand attention away.
        by_id = {x.id: x for x in people}
        if self.target_id in by_id:
            if by_id[self.target_id].confidence >= self.p.min_confidence:
                self._low_since = None
            elif self._low_since is None:
                self._low_since = now
            elif now - self._low_since > self.p.low_confidence_drop_s:
                # A one-frame dip is tolerated; a track that stays low (e.g. a ghost on the ceiling
                # the real person's track decayed into) is dropped so the head can give up.
                del by_id[self.target_id]
        if self.target_id not in by_id:
            self.target_id = None
            self._challenger = None
            self._low_since = None
        if not can_acquire:
            self._challenger = None
            return self.target_id
        if self.target_id is None:
            if eligible:
                self.target_id = max(eligible, key=lambda x: score(x, self.p)).id
            return self.target_id
        current = score(by_id[self.target_id], self.p, is_target=True)
        rivals = [x for x in eligible if x.id != self.target_id]
        best = max(rivals, key=lambda x: score(x, self.p), default=None)
        if best is not None and score(best, self.p) > current:
            if self._challenger != best.id:
                self._challenger, self._challenger_since = best.id, now
            elif now - self._challenger_since >= self.p.switch_hold_s:
                self.target_id, self._challenger = best.id, None
        else:
            self._challenger = None
        return self.target_id

    def glance_candidate(self, people, now: float) -> int | None:
        if now - self._last_glance_at < self.p.glance_cooldown_s:
            return None
        for x in self._eligible(people):
            if x.id == self.target_id:
                continue
            newcomer = self.p.glance_newcomer and now - x.first_seen <= self.p.newcomer_window_s
            moving = self.p.glance_movement and x.motion == "passing"
            if newcomer or moving:
                self._last_glance_at = now
                return x.id
        return None
