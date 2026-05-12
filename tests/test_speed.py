"""Direct unit tests for the windowed live-speed and ETA math on ``CopyJob``."""

from __future__ import annotations

import math
from collections import deque

import pytest

from ocopy.verified_copy import SPEED_WINDOW_S, CopyJob


@pytest.fixture
def job(tmp_path):
    """A ``CopyJob`` that does not run; we only exercise its sampling math."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.bin").write_bytes(b"x" * 1024)
    dst = tmp_path / "dst"
    dst.mkdir()
    return CopyJob(src, [dst], auto_start=False)


def test_windowed_rate_empty_returns_none(job):
    assert CopyJob._windowed_rate(deque()) is None


def test_windowed_rate_single_sample_returns_none(job):
    samples: deque[tuple[float, float]] = deque([(0.0, 0.0)])
    assert CopyJob._windowed_rate(samples) is None


def test_windowed_rate_two_samples(job):
    samples: deque[tuple[float, float]] = deque([(100.0, 0.0), (101.0, 100_000_000.0)])
    assert CopyJob._windowed_rate(samples) == pytest.approx(100_000_000.0)


def test_windowed_rate_tight_window_returns_none(job):
    # dt < 0.1s → fallback signal
    samples: deque[tuple[float, float]] = deque([(100.000, 0.0), (100.05, 1_000_000.0)])
    assert CopyJob._windowed_rate(samples) is None


def test_prune_drops_old_samples(job):
    now = 1000.0
    samples: deque[tuple[float, float]] = deque(
        [
            (now - SPEED_WINDOW_S - 1.0, 1.0),  # dropped
            (now - SPEED_WINDOW_S - 0.1, 2.0),  # dropped
            (now - 1.0, 3.0),  # kept
            (now, 4.0),  # kept
        ]
    )
    CopyJob._prune(samples, now)
    assert list(samples) == [(now - 1.0, 3.0), (now, 4.0)]


def test_live_speed_copy_fallback_to_cumulative_when_no_samples(job):
    # No samples yet; falls back to cumulative copy bytes / elapsed.
    job._copy_bytes_done = 50_000_000.0
    # Force elapsed to a known value by shifting start time.
    job._start_time = job._start_time - 5.0  # 5 seconds elapsed
    assert job.live_speed_copy == pytest.approx(10_000_000.0, rel=0.01)


def test_live_speed_uses_current_phase(job):
    now = 1000.0
    job._samples_copy.extend([(now, 0.0), (now + 1.0, 200_000_000.0)])
    job._samples_verify.extend([(now, 0.0), (now + 1.0, 500_000_000.0)])
    job.current_phase = "copy"
    assert job.live_speed == job.live_speed_copy
    job.current_phase = "verify"
    assert job.live_speed == job.live_speed_verify


def test_live_speed_verify_falls_back_to_copy_speed(job):
    now = 1000.0
    job._samples_copy.extend([(now, 0.0), (now + 1.0, 200_000_000.0)])
    # No verify samples → fallback.
    assert job.live_speed_verify == pytest.approx(job.live_speed_copy)


def test_eta_seconds_no_verify(job):
    # 80MB remaining at 10MB/s → 8s
    job.verify = False
    job.todo_size = 100_000_000
    job.total_size = 100_000_000
    job._copy_bytes_done = 20_000_000.0
    job.total_done = 20_000_000.0
    now = 1000.0
    # Two samples one second apart with 10MB delta give rate 10MB/s.
    job._samples_copy.extend([(now, 10_000_000.0), (now + 1.0, 20_000_000.0)])
    assert job.eta_seconds == pytest.approx(8.0)


def test_eta_seconds_with_verify_copy_phase(job):
    # In copy phase: copy_remaining/copy_speed + verify_total/verify_speed_fallback (= copy_speed).
    # 80MB remaining copy at 10MB/s = 8s; 100MB verify at 10MB/s fallback = 10s → total 18s.
    job.verify = True
    job.total_size = 100_000_000
    job.todo_size = 200_000_000
    job._copy_bytes_done = 20_000_000.0
    job._verify_bytes_done = 0.0
    job.total_done = 20_000_000.0
    job.current_phase = "copy"
    now = 1000.0
    job._samples_copy.extend([(now, 10_000_000.0), (now + 1.0, 20_000_000.0)])
    assert job.eta_seconds == pytest.approx(18.0)


def test_eta_seconds_with_verify_phase_uses_verify_rate(job):
    # Copy is done; verify in progress at 20MB/s. 60MB verify remaining → 3s.
    job.verify = True
    job.total_size = 100_000_000
    job.todo_size = 200_000_000
    job._copy_bytes_done = 100_000_000.0
    job._verify_bytes_done = 40_000_000.0
    job.total_done = 140_000_000.0
    job.current_phase = "verify"
    now = 1000.0
    job._samples_verify.extend([(now, 20_000_000.0), (now + 1.0, 40_000_000.0)])
    # copy_remaining is 0, so ETA = 60MB / 20MB/s = 3s.
    assert job.eta_seconds == pytest.approx(3.0)


def test_eta_seconds_zero_when_finished(job):
    job.todo_size = 100_000_000
    job.total_done = 100_000_000
    assert job.eta_seconds == 0.0


def test_eta_seconds_inf_when_no_speed_data(job):
    # No samples, no bytes copied, elapsed > 0 → live_speed_copy is 0 → ETA unknown.
    job.verify = True
    job.total_size = 100_000_000
    job.todo_size = 200_000_000
    job._copy_bytes_done = 0.0
    job.total_done = 0.0
    job._start_time = job._start_time - 0.5  # tiny elapsed so cumulative is still 0
    assert math.isinf(job.eta_seconds)


def test_cold_start_progressively_fills_window(job):
    """During the first ~5s while the deque fills, speed and ETA must remain sane.

    Sane = non-negative finite values OR ``math.inf`` (unknown). No NaN, no negative,
    no absurdly large numbers driven by tiny-denominator divisions.
    """
    job.verify = False
    job.total_size = 1_000_000_000  # 1 GB
    job.todo_size = 1_000_000_000

    # Walk a simulated chunk stream at a steady 200 MB/s for 8 seconds.
    # Sample every 50ms (≈ 10MB chunks at this rate) into the deque, matching what
    # _progress_reader would do, including the prune step.
    chunk_dt = 0.05
    rate = 200_000_000.0  # 200 MB/s
    base = 1000.0  # fake "now" baseline
    job._start_time = base  # so .elapsed is consistent with our fake clock

    observed_speeds: list[float] = []
    observed_etas: list[float] = []

    for i in range(1, 161):  # 160 samples * 50ms = 8s total
        now = base + i * chunk_dt
        job._copy_bytes_done = i * chunk_dt * rate
        job.total_done = job._copy_bytes_done
        job._samples_copy.append((now, job._copy_bytes_done))
        CopyJob._prune(job._samples_copy, now)

        # We don't have a fake clock for ``elapsed``; the only thing that depends
        # on it (live_speed fallback) won't be exercised here because by sample 2
        # we always have a valid window. Read the properties and assert sanity.
        speed = job.live_speed_copy
        eta = job.eta_seconds
        observed_speeds.append(speed)
        observed_etas.append(eta)

        # Speed must be finite and non-negative at every step.
        assert math.isfinite(speed)
        assert speed >= 0
        # ETA must be 0, +inf, or a finite non-negative number — never negative, never NaN.
        assert eta == 0 or math.isinf(eta) or (math.isfinite(eta) and eta >= 0)

    # After the window has had time to settle (samples 20+ ≈ t=1s+), speeds
    # should be within 20% of the simulated 200 MB/s.
    settled = observed_speeds[20:]
    for s in settled:
        assert 0.8 * rate <= s <= 1.2 * rate, f"settled speed {s} far from {rate}"

    # The deque must have pruned to within SPEED_WINDOW_S worth of samples after t=5s.
    # At 50ms chunk_dt + 5s window, we expect ~100 samples in the deque.
    assert 90 <= len(job._samples_copy) <= 110
