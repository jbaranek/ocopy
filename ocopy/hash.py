from concurrent import futures
from functools import lru_cache, partial
from pathlib import Path
from queue import Queue

from ascmhl import errors as ascmhl_errors
from ascmhl.__version__ import ascmhl_folder_name
from ascmhl.hasher import Hasher, HashType
from ascmhl.history import MHLHistory

from ocopy.checkpoint import Checkpoint
from ocopy.mhl import find_mhl, xxh64_from_legacy_mhl_path
from ocopy.progress import ProgressPhase, ProgressUpdate, get_progress_queue

SUPPORTED_ALGORITHMS: tuple[str, ...] = tuple(t.name for t in HashType)
DEFAULT_ALGORITHM = "xxh64"


def _new_hasher(algorithm: str) -> Hasher:
    """Return an ASC MHL Hasher (``.update(bytes)`` / ``.string_digest()``).

    Reusing ``ascmhl.hasher.HashType`` keeps the algorithm catalog in sync with
    what ASC MHL itself accepts (incl. C4's custom base58 encoding) without
    reimplementing it here.
    """
    try:
        return HashType[algorithm].value()
    except KeyError as e:
        raise ValueError(f"unsupported hash algorithm: {algorithm!r}") from e


def get_hash(
    file_path: Path,
    progress_queue: Queue[ProgressUpdate] | None = None,
    total_files: int = 1,
    algorithm: str = DEFAULT_ALGORITHM,
) -> str:
    x = _new_hasher(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            x.update(chunk)
            if progress_queue:
                progress_queue.put(
                    ProgressUpdate(
                        ProgressPhase.VERIFY,
                        file_path,
                        len(chunk),
                        parallel_verify_readers=total_files,
                    ),
                )

    return x.string_digest()


def multi_xxhash_check(filenames: list[Path], algorithm: str = DEFAULT_ALGORITHM) -> str:
    with futures.ThreadPoolExecutor(max_workers=len(filenames)) as executor:
        unique_file_hashes = {
            file_hash
            for file_hash in executor.map(
                partial(
                    get_hash,
                    progress_queue=get_progress_queue(),
                    total_files=len(filenames),
                    algorithm=algorithm,
                ),
                filenames,
            )
        }

    return unique_file_hashes.pop() if len(unique_file_hashes) == 1 else "hashes_do_not_match"


def _innermost_root_with_marker(file_path: Path, marker: str, *, is_dir: bool) -> Path | None:
    """Walk from ``file_path`` upwards and return the deepest ancestor containing ``marker``.

    ``marker`` is checked against ``parent / marker`` with ``is_dir`` distinguishing
    a directory marker (e.g. ``ascmhl``) from a file marker (e.g. ``.ocopy-checkpoint``).
    Returns ``None`` when no ancestor qualifies or when ``file_path`` cannot be
    resolved as relative to any candidate (e.g. symlink crossings).
    """
    p = file_path.resolve()
    candidates: list[Path] = []
    if p.is_dir():
        candidates.append(p)
    candidates.extend(p.parents)
    best: Path | None = None
    for parent in candidates:
        marker_path = parent / marker
        if is_dir:
            if not marker_path.is_dir():
                continue
        elif not marker_path.is_file():
            continue
        try:
            p.relative_to(parent)
        except ValueError:
            continue
        if best is None or len(parent.parts) > len(best.parts):
            best = parent
    return best


def _hash_from_checkpoint(content_root: Path, file_path: Path, algorithm: str) -> str | None:
    try:
        rel = file_path.resolve().relative_to(content_root.resolve()).as_posix()
    except ValueError:
        return None
    try:
        st = file_path.stat()
    except OSError:
        return None
    return Checkpoint(content_root).lookup(rel, st.st_size, st.st_mtime, algorithm)


@lru_cache(maxsize=1024)
def _cached_load_ascmhl(content_root_str: str, _mtime_ns: int) -> MHLHistory | None:
    """Load ASC MHL history; keyed by ``(content_root_str, mtime_ns)`` of ``ascmhl/``.

    ``maxsize=1024``: typical runs touch one or a few content roots; re-seals add new
    ``mtime_ns`` keys. This bounds retained ``MHLHistory`` objects in long-lived
    processes (e.g. GUI) while leaving plenty of headroom for many volumes and
    generations of invalidation keys.
    """
    try:
        return MHLHistory.load_from_path(content_root_str)
    except (
        ascmhl_errors.NoMHLChainException,
        ascmhl_errors.ModifiedMHLManifestFileException,
        ascmhl_errors.MissingMHLManifestException,
        OSError,
    ):
        return None


def _load_ascmhl_history(content_root: Path) -> MHLHistory | None:
    marker = content_root / ascmhl_folder_name
    try:
        mtime_ns = marker.stat().st_mtime_ns
    except OSError:
        return None
    return _cached_load_ascmhl(str(content_root.resolve()), mtime_ns)


def _hash_latest_from_ascmhl(content_root: Path, file_path: Path, algorithm: str) -> str | None:
    history = _load_ascmhl_history(content_root)
    if history is None:
        return None

    try:
        rel = file_path.resolve().relative_to(content_root.resolve())
    except ValueError:
        return None

    # ``media_hashes_path_map`` keys are produced via ``convert_posix_to_local_path``
    # in the ASC MHL XML parser, which calls ``PureWindowsPath`` on Windows - so the
    # stored key has backslash separators there. Using ``str(rel)`` gives the native
    # separator on each OS and matches the library's convention; ``.as_posix()``
    # would miss on Windows. We try the POSIX form as a fallback for histories that
    # were written on a POSIX host and then consulted on Windows (same filesystem,
    # cross-platform read).
    for hash_list in reversed(history.hash_lists):
        for form in (str(rel), rel.as_posix()):
            media_hash = hash_list.find_media_hash_for_path(form)
            if media_hash is None or media_hash.is_directory:
                continue
            entry = media_hash.find_hash_entry_for_format(algorithm)
            if entry is not None and entry.hash_string:
                return entry.hash_string
    return None


def find_hash(file_path: Path, algorithm: str = DEFAULT_ALGORITHM) -> str | None:
    ck_root = _innermost_root_with_marker(file_path, Checkpoint.FILENAME, is_dir=False)
    if ck_root is not None:
        ck_hash = _hash_from_checkpoint(ck_root, file_path, algorithm)
        if ck_hash:
            return ck_hash

    asc_root = _innermost_root_with_marker(file_path, ascmhl_folder_name, is_dir=True)
    if asc_root is not None:
        asc_hash = _hash_latest_from_ascmhl(asc_root, file_path, algorithm)
        if asc_hash:
            return asc_hash

    # Legacy flat ``*.mhl`` v1.1 only defines ``<xxhash64be>``; no other algorithms.
    if algorithm == "xxh64":
        dot_mhl = find_mhl(file_path)
        if dot_mhl:
            try:
                rel = file_path.resolve().relative_to(dot_mhl.parent.resolve())
            except ValueError:
                return None
            file_hash = xxh64_from_legacy_mhl_path(dot_mhl, rel)
            if file_hash:
                return file_hash

    return None
