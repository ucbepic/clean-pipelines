"""The `prepare` step: a mentions CSV -> jsonl of OfficerMention records.

Keeps input prep free of model/LLM/API imports so it's cheap to run and test.
"""

from __future__ import annotations

from .io import read_mentions


def prepare(
    input_path,
    output_path,
    *,
    default_state: str | None = None,
    sample_n: int | None = None,
    sample_seed: int | None = None,
) -> int:
    """Read a mentions CSV and write one jsonl line per OfficerMention.

    Returns the number of records written.
    """
    from prap_core.io import write_jsonl

    mentions = read_mentions(
        str(input_path),
        default_state=default_state,
        sample_n=sample_n,
        sample_seed=sample_seed,
    )
    return write_jsonl(output_path, [m.model_dump(mode="json") for m in mentions])
