"""ASRRouter: remote-primary + local SenseVoice fallback + circuit breaker.

Deterministic — engines are lightweight in-test fakes and time is an injected
clock, so breaker/cooldown behaviour is exercised without real sockets or
sleeping. (No product-code mocks; the fakes live only here.)
"""
from whisperflow_local.asr import ASRUnavailable
from whisperflow_local.asr_router import ASRRouter


class FakeEngine:
    def __init__(self, name, healthy=True, reply="OUT"):
        self.name = name
        self.healthy = healthy
        self.reply = reply
        self.calls = 0

    def transcribe(self, wav_path, language="auto", context=None):
        self.calls += 1
        if not self.healthy:
            raise ASRUnavailable(f"{self.name} down")
        return self.reply

    def ensure_loaded(self, progress_cb=None):
        pass

    def ping(self):
        return self.healthy


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make(remote_healthy=True, backend="auto", threshold=3, cooldown=300.0,
         notify=None):
    clock = Clock()
    local = FakeEngine("sensevoice", healthy=True, reply="LOCAL")
    remote = FakeEngine("qwen3", healthy=remote_healthy, reply="REMOTE")
    router = ASRRouter(local=local, remote=remote, backend=backend,
                       threshold=threshold, cooldown=cooldown,
                       clock=clock, notify=notify)
    return router, local, remote, clock


def test_remote_used_when_healthy():
    router, local, remote, _ = make()
    assert router.transcribe("a.wav") == "REMOTE"
    assert remote.calls == 1 and local.calls == 0
    assert router.engine_name == "qwen3"


def test_fallback_on_remote_failure():
    router, local, remote, _ = make(remote_healthy=False)
    assert router.transcribe("a.wav", "yue", ["WhisperFlow"]) == "LOCAL"
    assert remote.calls == 1 and local.calls == 1
    assert router.engine_name == "sensevoice"   # reflects the actual server


def test_success_resets_consecutive_counter():
    router, local, remote, _ = make(remote_healthy=False, threshold=3)
    router.transcribe("a.wav")   # fail 1
    router.transcribe("b.wav")   # fail 2
    remote.healthy = True
    assert router.transcribe("c.wav") == "REMOTE"  # success -> reset
    remote.healthy = False
    router.transcribe("d.wav")   # fail 1 (not 3)
    router.transcribe("e.wav")   # fail 2
    before = remote.calls
    router.transcribe("f.wav")   # fail 3 -> trips on this call
    router.transcribe("g.wav")   # tripped: no remote attempt
    assert remote.calls == before + 1


def test_breaker_trips_after_threshold_then_skips_remote():
    router, local, remote, _ = make(remote_healthy=False, threshold=3)
    for _ in range(3):
        router.transcribe("x.wav")
    assert remote.calls == 3
    router.transcribe("x.wav")   # tripped -> straight to local
    router.transcribe("x.wav")
    assert remote.calls == 3
    assert local.calls == 5


def test_cooldown_reprobes_and_recovers():
    router, local, remote, clock = make(remote_healthy=False, threshold=3,
                                        cooldown=300.0)
    for _ in range(3):
        router.transcribe("x.wav")   # trip
    assert remote.calls == 3
    clock.advance(299)
    router.transcribe("x.wav")       # still cooling: no probe
    assert remote.calls == 3
    clock.advance(2)
    remote.healthy = True
    assert router.transcribe("x.wav") == "REMOTE"  # probe -> recover
    assert remote.calls == 4
    assert router.engine_name == "qwen3"


def test_probe_failure_restays_local_and_rearms_cooldown():
    router, local, remote, clock = make(remote_healthy=False, threshold=3,
                                        cooldown=300.0)
    for _ in range(3):
        router.transcribe("x.wav")   # trip
    clock.advance(301)
    router.transcribe("x.wav")       # probe -> fails
    assert remote.calls == 4
    router.transcribe("x.wav")       # cooldown re-armed: no probe
    assert remote.calls == 4


def test_local_pinned_never_calls_remote():
    router, local, remote, _ = make(backend="local")
    assert router.transcribe("a.wav") == "LOCAL"
    assert remote.calls == 0
    assert router.engine_name == "sensevoice"
    assert router.ping() is True   # local always reachable


def test_set_engine_maps_and_resets_breaker():
    router, local, remote, _ = make(remote_healthy=False, threshold=3)
    for _ in range(3):
        router.transcribe("x.wav")   # trip
    router.set_engine("sensevoice")  # -> local-only
    assert router.transcribe("x.wav") == "LOCAL"
    assert remote.calls == 3          # pinned local, remote untouched
    router.set_engine("qwen3")        # -> auto, breaker reset
    remote.healthy = True
    assert router.transcribe("x.wav") == "REMOTE"
    assert remote.calls == 4           # tried remote right after reset


def test_notify_fires_on_trip_and_reconnect():
    events = []
    router, local, remote, clock = make(
        remote_healthy=False, threshold=3, cooldown=300.0,
        notify=lambda ev, eng: events.append((ev, eng)))
    for _ in range(3):
        router.transcribe("x.wav")   # trip -> "fallback"
    assert events == [("fallback", "sensevoice")]
    clock.advance(301)
    remote.healthy = True
    router.transcribe("x.wav")       # probe succeeds -> "reconnected"
    assert events == [("fallback", "sensevoice"), ("reconnected", "qwen3")]
