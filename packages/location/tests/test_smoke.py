"""Trivial import-only smoke tests. No API calls."""


def test_package_imports():
    import prap_location  # noqa: F401


def test_run_is_callable():
    from prap_location import run

    assert callable(run)


def test_schemas_import():
    from prap_location.schemas import (
        CaseRecord,
        LocationResult,
        RunResult,
        ValidationResult,
    )

    assert CaseRecord and LocationResult and RunResult and ValidationResult
