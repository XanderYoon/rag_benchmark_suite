from __future__ import annotations

from UI.state import session_state as state


def test_benchmark_snapshot_set_get_and_clear(monkeypatch) -> None:
    fake_session_state: dict = {}
    monkeypatch.setattr(state.st, "session_state", fake_session_state)

    snapshot = {"run_id": "run_001", "baseline": {"hit_at_1": 0.75}}
    state.set_benchmark_snapshot(snapshot=snapshot)

    loaded_snapshot = state.get_benchmark_snapshot()
    assert loaded_snapshot == snapshot
    assert loaded_snapshot is not snapshot

    state.clear_benchmark_snapshot()
    assert state.get_benchmark_snapshot() is None


def test_set_benchmark_snapshot_rejects_non_dict(monkeypatch) -> None:
    fake_session_state: dict = {}
    monkeypatch.setattr(state.st, "session_state", fake_session_state)

    try:
        state.set_benchmark_snapshot(snapshot="invalid")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Expected a dictionary" in str(exc)
        return

    raise AssertionError("Expected ValueError for non-dictionary snapshot.")


def test_get_benchmark_snapshot_rejects_invalid_stored_type(monkeypatch) -> None:
    fake_session_state = {state.BENCHMARK_SNAPSHOT_KEY: "invalid"}
    monkeypatch.setattr(state.st, "session_state", fake_session_state)

    try:
        state.get_benchmark_snapshot()
    except ValueError as exc:
        assert state.BENCHMARK_SNAPSHOT_KEY in str(exc)
        assert "Expected a dictionary" in str(exc)
        return

    raise AssertionError("Expected ValueError for invalid snapshot type in session state.")


def test_resolve_corpus_dir_preserves_absolute_paths(tmp_path) -> None:
    resolved = state._resolve_corpus_dir(tmp_path)

    assert resolved == tmp_path


def test_resolve_corpus_dir_resolves_relative_to_project_root() -> None:
    resolved = state._resolve_corpus_dir("data/rag_corpus_pdf")

    assert resolved == (state.PROJECT_ROOT / "data/rag_corpus_pdf").resolve()
