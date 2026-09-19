"""Per-session-name lifecycle lock serializing session CREATE against TEARDOWN.

Ported from upstream awslabs/cli-agent-orchestrator PR #498
(fix(sessions): make session teardown atomic so tmux and the registry cannot
diverge) onto this project's spaceandparcel-2.2.0-based fork. Upstream's own
create/teardown call sites had diverged too far (async cancellation-safe
worker tasks, a ``backends/`` abstraction layer that predates this fork's
branch point) for a line-level port of the surrounding code, so only this
self-contained lock primitive is taken verbatim; the call sites in
``terminal_service.py``/``session_service.py`` are a fork-appropriate
reimplementation of the same design.

Why this exists (#498): tmux and the terminal registry (SQLite) are two
separate stores, and a session's lifecycle transitions write BOTH. Without
mutual exclusion, a create landing mid-teardown (or two concurrent teardowns)
interleaves arbitrarily and orphans state in one store or the other — this
project's own launch-error transcripts (``Failed to delete session ... not
found`` immediately followed by ``Startup prompt handler timed out`` /
``Claude Code initialization timed out after 30 seconds``) reproduce exactly
this failure shape when a client retries a create for a session name whose
prior attempt's teardown/cleanup is still in flight server-side.

Design notes
------------
``threading.Lock``, not ``asyncio.Lock``. Session creation in this fork's
``terminal_service.create_terminal`` and teardown in
``session_service.delete_session`` are both plain synchronous functions
called from ``async def`` FastAPI handlers via ``asyncio.to_thread`` (see
``api/main.py``) — a threading primitive is the only one reachable from
both worker-thread contexts, and there is no single-event-loop assumption to
rely on.

To keep this safe, the lock is held only across each path's SHORT
state-transition critical section — never across the long
``provider.initialize()`` call (tens of seconds). See the call sites for
exactly what each critical section spans.

Locks are refcounted and dropped at zero, so the registry cannot grow without
bound as sessions come and go.
"""

import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Tuple

# Guards the registry below. Only ever held for the few dict operations in
# ``session_lifecycle_lock``'s acquire/release bookkeeping — never across the
# caller's critical section, so it can't serialize different session names.
_registry_guard = threading.Lock()

# session name -> (lock, number of holders+waiters currently interested in it).
# The count is what makes eviction safe: the entry is removed only once nobody
# is using it, so two threads contending for the same name always end up on the
# SAME lock object (a plain "pop on release" would let a waiter be handed a
# fresh, uncontended lock and defeat the mutual exclusion entirely).
_session_locks: Dict[str, Tuple[threading.Lock, int]] = {}


@contextmanager
def session_lifecycle_lock(session_name: str) -> Iterator[None]:
    """Hold the lifecycle lock for ``session_name`` across the with-block.

    Different session names never contend (the whole point — a global lock
    would serialize every session operation on the server); the same name is
    strictly serialized, for create-vs-teardown and teardown-vs-teardown
    alike.

    Release is guaranteed on every exit path, exceptions included.

    NOT reentrant: a call path already holding the lock for ``name`` must not
    re-enter it for the same ``name``, or it self-deadlocks.
    """
    with _registry_guard:
        lock, holders = _session_locks.get(session_name, (threading.Lock(), 0))
        _session_locks[session_name] = (lock, holders + 1)

    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _registry_guard:
            # Re-read rather than trusting the count captured above: other
            # threads have adjusted it in the meantime.
            entry = _session_locks.get(session_name)
            if entry is not None:
                held_lock, holders = entry
                if holders <= 1:
                    del _session_locks[session_name]
                else:
                    _session_locks[session_name] = (held_lock, holders - 1)
