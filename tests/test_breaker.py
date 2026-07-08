"""CircuitBreaker: trip, cooldown gating, re-probe, reset — deterministic
via an injected clock (no sleeping)."""
from whisperflow_local.breaker import CircuitBreaker


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make(threshold=3, cooldown=300.0):
    clock = Clock()
    return CircuitBreaker(threshold=threshold, cooldown=cooldown, clock=clock), clock


def test_healthy_allows_remote():
    b, _ = make()
    assert b.allow_remote() is True


def test_trips_only_on_threshold_failure():
    b, _ = make(threshold=3)
    assert b.record_failure() is False   # 1
    assert b.record_failure() is False   # 2
    assert b.record_failure() is True    # 3 -> trips now
    assert b.allow_remote() is False     # cooling


def test_success_resets_and_reports_prior_trip():
    b, _ = make(threshold=3)
    for _ in range(3):
        b.record_failure()               # trip
    assert b.record_success() is True    # was tripped -> reconnected
    assert b.allow_remote() is True
    # A fresh success (never tripped) reports False.
    assert b.record_success() is False


def test_success_resets_consecutive_before_trip():
    b, _ = make(threshold=3)
    b.record_failure()
    b.record_failure()
    b.record_success()                   # reset the count
    assert b.record_failure() is False   # this is failure #1 again, not #3
    assert b.allow_remote() is True


def test_cooldown_gates_reprobe():
    b, clock = make(threshold=3, cooldown=300.0)
    for _ in range(3):
        b.record_failure()               # trip
    clock.advance(299)
    assert b.allow_remote() is False     # still cooling
    clock.advance(2)
    assert b.allow_remote() is True      # cooldown elapsed -> one probe allowed


def test_failure_while_tripped_rearms_cooldown():
    b, clock = make(threshold=3, cooldown=300.0)
    for _ in range(3):
        b.record_failure()               # trip
    clock.advance(301)
    assert b.allow_remote() is True      # probe window
    assert b.record_failure() is False   # probe failed (not a fresh trip)
    assert b.allow_remote() is False     # cooldown re-armed
    clock.advance(301)
    assert b.allow_remote() is True


def test_reset_clears_trip():
    b, _ = make(threshold=3)
    for _ in range(3):
        b.record_failure()
    b.reset()
    assert b.allow_remote() is True
    assert b.record_failure() is False   # counter cleared
