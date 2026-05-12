"""End-to-end checks for the integrity invariants of :func:`verified_copy`.

These tests target *correctness guarantees*, not coverage points: the assertions
in here are the reason ocopy exists. They exercise paths that other suites only
touch in their happy direction:

- conflicting trusted hashes between destinations must raise (not silently pick one).
- a source-side manifest that disagrees with the freshly-computed digest must raise.
- verification mismatch under ``overwrite=True`` retries exactly once, then raises.
- the ``--dont-verify --no-mhl`` fresh-copy path returns the streamed copy digest
  without a pool re-read (i.e. the user actually gets what they paid for in speed).
- the resume-checkpoint reader must drop malformed lines without losing earlier
  good records (crash-safe partial writes are the whole point of fsync'd JSONL).
"""

from __future__ import annotations

import importlib
import os

import pytest

from ocopy.ascmhl_seal import seal_ascmhl_at_destination
from ocopy.checkpoint import Checkpoint
from ocopy.file_info import FileInfo
from ocopy.hash import get_hash
from ocopy.verified_copy import VerificationError, verified_copy


def test_conflicting_trusted_hashes_between_destinations_raises(tmp_path):
    """Two destinations claiming different trusted hashes for the same source must error.

    Picking either hash silently would mean a tampered checkpoint can validate a
    matching-bytes destination as the wrong content. The classifier collects all
    trusted hashes into a set and rejects len > 1.
    """
    src = tmp_path / "src"
    src.mkdir()
    payload = b"shared-bytes"
    f = src / "f.bin"
    f.write_bytes(payload)
    mtime = f.stat().st_mtime

    d1_root = tmp_path / "d1" / "src"
    d2_root = tmp_path / "d2" / "src"
    d1_root.mkdir(parents=True)
    d2_root.mkdir(parents=True)
    dst1 = d1_root / "f.bin"
    dst2 = d2_root / "f.bin"
    for dst in (dst1, dst2):
        dst.write_bytes(payload)
        os.utime(dst, (mtime, mtime))

    # Two checkpoints, both metadata-matching, but with conflicting digests.
    Checkpoint(d1_root).record("f.bin", len(payload), mtime, "xxh64", "a" * 16)
    Checkpoint(d2_root).record("f.bin", len(payload), mtime, "xxh64", "b" * 16)

    with pytest.raises(VerificationError, match="Conflicting trusted hashes"):
        verified_copy(f, [dst1, dst2], skip_existing=True)


def test_source_medium_hash_mismatch_raises(tmp_path):
    """A source manifest that contradicts the freshly-computed digest must raise.

    Models the "bad bytes on a card whose MHL was generated before the corruption"
    case. The destination is a clean fresh copy — both src and tmp hash the same
    bytes, so the pool agrees — but the source-side MHL is from an earlier truth.
    """
    src = tmp_path / "src"
    src.mkdir()
    f = src / "f.bin"
    original = b"original-bytes"
    f.write_bytes(original)
    orig_mtime = f.stat().st_mtime

    # Seal an ASC MHL at the source recording the *original* digest.
    src_hash = get_hash(f)
    seal_ascmhl_at_destination(
        src,
        src,
        [FileInfo(f, src_hash, len(original), orig_mtime)],
    )

    # Tamper with the file but preserve mtime, so the manifest is now lying.
    tampered = b"tampered-bytes"
    assert len(tampered) == len(original)  # keep size stable
    f.write_bytes(tampered)
    os.utime(f, (orig_mtime, orig_mtime))

    dst_root = tmp_path / "dst" / "src"
    dst_root.mkdir(parents=True)
    dst_file = dst_root / "f.bin"

    with pytest.raises(VerificationError, match="not correct"):
        verified_copy(f, [dst_file])

    # Nothing should remain in flight at the destination.
    assert not dst_file.exists()
    assert not dst_file.with_name(dst_file.name + ".copy_in_progress").exists()


def test_verification_retry_exhausts_with_overwrite(tmp_path, mocker):
    """Persistent verification failure under ``overwrite=True`` must raise on attempt 2.

    The repair-retry loop is bounded at 2 attempts. After a second mismatch the
    error must surface (no infinite loop, no silent skip).
    """
    mocker.patch(
        "ocopy.verified_copy.multi_xxhash_check",
        return_value="hashes_do_not_match",
    )

    src = tmp_path / "src"
    src.mkdir()
    f = src / "f.bin"
    f.write_bytes(b"abc")

    dst_root = tmp_path / "dst" / "src"
    dst_root.mkdir(parents=True)
    dst_file = dst_root / "f.bin"

    with pytest.raises(VerificationError, match="Verification failed"):
        verified_copy(f, [dst_file], overwrite=True)

    # Both attempts must have run; mock fired twice.
    from ocopy.verified_copy import multi_xxhash_check as patched

    assert patched.call_count == 2  # type: ignore[attr-defined]
    # No tmp or final dst should survive a failed retry.
    assert not dst_file.exists()
    assert not dst_file.with_name(dst_file.name + ".copy_in_progress").exists()


def test_fresh_copy_no_integrity_returns_streamed_digest_without_pool_reread(tmp_path, mocker):
    """``verify=False`` + no MHL on a fresh destination must not trigger a pool re-read.

    The whole point of the no-integrity fast path is that the user opted out of
    verification reads. If we silently re-read for the pool anyway we'd burn the
    IO they were trying to avoid.
    """
    pool_spy = mocker.spy(
        importlib.import_module("ocopy.verified_copy"),
        "multi_xxhash_check",
    )

    src = tmp_path / "src"
    src.mkdir()
    f = src / "f.bin"
    payload = b"streamed-payload"
    f.write_bytes(payload)

    dst_root = tmp_path / "dst" / "src"
    dst_root.mkdir(parents=True)
    dst_file = dst_root / "f.bin"

    digest = verified_copy(f, [dst_file], verify=False)

    assert pool_spy.call_count == 0
    assert dst_file.read_bytes() == payload
    # The returned digest is what the streamed copy produced; equals an independent hash.
    assert digest == get_hash(f)


@pytest.mark.parametrize(
    ("bad_line", "label"),
    [
        (b'{"rel_path":"x","size":"7","mtime":1.0,"algorithm":"xxh64","hash":"a"}\n', "size_as_string"),
        (b'{"rel_path":"x","size":1,"mtime":null,"algorithm":"xxh64","hash":"a"}\n', "mtime_null"),
        (b'{"rel_path":123,"size":1,"mtime":1.0,"algorithm":"xxh64","hash":"a"}\n', "rel_path_not_string"),
        (b'{"rel_path":"x","size":1,"mtime":"not-a-float","algorithm":"xxh64","hash":"a"}\n', "mtime_not_floatable"),
        (b'{"rel_path":"x","size":1,"mtime":1.0,"hash":"a"}\n', "missing_algorithm_no_legacy"),
        (b'{"rel_path":"x","size":1,"mtime":1.0,"algorithm":"xxh64","hash":""}\n', "empty_hash"),
        (b"not-json-at-all\n", "garbage_line"),
        (b"\n", "blank_line"),
    ],
)
def test_checkpoint_reader_skips_malformed_lines_without_losing_good_records(tmp_path, bad_line, label):
    """A malformed JSONL line must be silently skipped; preceding good records survive.

    The reader is what runs on resume — if any malformation aborted parsing, a
    user with a crash-truncated checkpoint would lose every record after the
    bad line. We append a known-good record after the bad one to prove it.
    """
    root = tmp_path / f"r_{label}"
    cp = Checkpoint(root)
    cp.ensure_exists()

    # Good record first, then the malformed one, then another good one.
    cp.record("good_a.bin", 4, 100.0, "xxh64", "g" * 16)
    cp.path.write_bytes(cp.path.read_bytes() + bad_line)
    cp.record("good_b.bin", 8, 200.0, "xxh64", "h" * 16)

    assert cp.lookup("good_a.bin", 4, 100.0, "xxh64") == "g" * 16
    assert cp.lookup("good_b.bin", 8, 200.0, "xxh64") == "h" * 16
