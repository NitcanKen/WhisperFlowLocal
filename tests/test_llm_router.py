"""LLMRouter: remote-primary + local fallback + circuit breaker + cooldown.

Deterministic — backends are lightweight in-test fakes and time is an injected
clock, so breaker/cooldown behaviour is exercised without real sockets or
sleeping. (No product-code mocks; the fakes live only here.)
"""
from whisperflow_local.llm import LLMUnavailable
from whisperflow_local.router import LLMRouter


class FakeBackend:
    def __init__(self, name, healthy=True, reply="OUT"):
        self.model = name
        self.healthy = healthy
        self.reply = reply
        self.calls = 0

    def _run(self):
        self.calls += 1
        if not self.healthy:
            raise LLMUnavailable(f"{self.model} down")
        return self.reply

    def format_text(self, text, profile, vocab=None):
        return self._run()

    def propose_edits(self, text, vocab=None):
        self._run()
        return []

    def run_command(self, command, text):
        return self._run()

    def ping(self):
        return self.healthy


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make(local_healthy=True, remote_healthy=True, backend="auto",
         threshold=3, cooldown=300.0, notify=None):
    clock = Clock()
    local = FakeBackend("local-4b", healthy=local_healthy, reply="LOCAL")
    remote = FakeBackend("remote-35b", healthy=remote_healthy, reply="REMOTE")
    router = LLMRouter(local=local, remote=remote, backend=backend,
                       threshold=threshold, cooldown=cooldown,
                       clock=clock, notify=notify)
    return router, local, remote, clock


def test_remote_used_when_healthy():
    router, local, remote, _ = make()
    assert router.format_text("hi", "Clean") == "REMOTE"
    assert remote.calls == 1 and local.calls == 0
    assert router.model == "remote-35b"


def test_fallback_on_remote_failure():
    router, local, remote, _ = make(remote_healthy=False)
    assert router.format_text("hi", "Clean") == "LOCAL"
    assert remote.calls == 1 and local.calls == 1
    assert router.model == "local-4b"


def test_success_resets_consecutive_counter():
    router, local, remote, _ = make(remote_healthy=False, threshold=3)
    router.format_text("a", "Clean")   # fail 1
    router.format_text("b", "Clean")   # fail 2
    remote.healthy = True
    assert router.format_text("c", "Clean") == "REMOTE"  # success -> reset
    remote.healthy = False
    router.format_text("d", "Clean")   # fail 1 (not 3)
    router.format_text("e", "Clean")   # fail 2
    before = remote.calls
    router.format_text("f", "Clean")   # fail 3 -> trips on this call
    router.format_text("g", "Clean")   # tripped: no remote attempt
    assert remote.calls == before + 1


def test_breaker_trips_after_threshold_then_skips_remote():
    router, local, remote, _ = make(remote_healthy=False, threshold=3)
    for _ in range(3):
        router.format_text("x", "Clean")
    assert remote.calls == 3
    router.format_text("x", "Clean")   # tripped -> straight to local
    router.format_text("x", "Clean")
    assert remote.calls == 3
    assert local.calls == 5


def test_cooldown_reprobes_and_recovers():
    router, local, remote, clock = make(remote_healthy=False, threshold=3,
                                        cooldown=300.0)
    for _ in range(3):
        router.format_text("x", "Clean")  # trip
    assert remote.calls == 3
    clock.advance(299)
    router.format_text("x", "Clean")      # still cooling: no probe
    assert remote.calls == 3
    clock.advance(2)
    remote.healthy = True
    assert router.format_text("x", "Clean") == "REMOTE"  # probe -> recover
    assert remote.calls == 4
    assert router.model == "remote-35b"
    router.format_text("x", "Clean")      # normal: remote directly
    assert remote.calls == 5


def test_probe_failure_restays_local_and_rearms_cooldown():
    router, local, remote, clock = make(remote_healthy=False, threshold=3,
                                        cooldown=300.0)
    for _ in range(3):
        router.format_text("x", "Clean")  # trip
    clock.advance(301)
    router.format_text("x", "Clean")      # probe -> fails
    assert remote.calls == 4
    router.format_text("x", "Clean")      # cooldown re-armed: no probe
    assert remote.calls == 4


def test_local_pinned_never_calls_remote():
    router, local, remote, _ = make(backend="local")
    assert router.format_text("hi", "Clean") == "LOCAL"
    router.propose_edits("hi")
    router.run_command("Summarize", "hi")
    assert remote.calls == 0
    assert router.model == "local-4b"


def test_set_backend_local_then_auto_resets_breaker():
    router, local, remote, _ = make(remote_healthy=False, threshold=3)
    for _ in range(3):
        router.format_text("x", "Clean")  # trip
    router.set_backend("local")
    router.set_backend("auto")            # reset breaker
    remote.healthy = True
    assert router.format_text("x", "Clean") == "REMOTE"
    assert remote.calls == 4              # tried remote right after reset


def test_notify_fires_on_trip_and_reconnect():
    events = []
    router, local, remote, clock = make(
        remote_healthy=False, threshold=3, cooldown=300.0,
        notify=lambda ev, m: events.append((ev, m)))
    for _ in range(3):
        router.format_text("x", "Clean")  # trip -> "fallback"
    assert events == [("fallback", "local-4b")]
    clock.advance(301)
    remote.healthy = True
    router.format_text("x", "Clean")      # probe succeeds -> "reconnected"
    assert events == [("fallback", "local-4b"), ("reconnected", "remote-35b")]
