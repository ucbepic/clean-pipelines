"""Two-step LLM-based filtering of document summaries.

Pipelines supply their own filter prompt template (the criteria for
"important" differ by task).

The filter prompt template must use `$count`, `$num_summaries`, and
`$formatted_text` placeholders (`string.Template` syntax).
"""

from __future__ import annotations

import logging
import re
from string import Template
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .llm import LLM

logger = logging.getLogger("prap.summary_filter")


_INDEX_EXTRACTION_PROMPT = Template(
    """
    <task>
    Your task is to extract only the document indices from the following text.
    The original LLM was asked to identify the $count most important documents out of $num_summaries total documents.
    </task>

    <input_text>
    $llm_response
    </input_text>

    <instructions>
    1. Look for numbers in the text that represent document indices
    2. Only consider indices that are valid (between 0 and $max_index inclusive)
    3. Return exactly $count indices if possible
    4. If there are more than $count indices mentioned, prioritize the ones that appear to be most important
    5. If there are fewer than $count valid indices mentioned, only return those that are valid
    </instructions>

    <output_format>
    Return ONLY a comma-separated list of indices, with no brackets or other text.
    For example: "0, 3, 5"
    </output_format>
    """
)


def _extract_clean_indices(
    llm: LLM,
    llm_response: str,
    count: int,
    num_summaries: int,
    *,
    log: logging.Logger | None = None,
    case_name: str | None = None,
    row_index: int = 0,
    chunk_idx: Any = None,
) -> list[int]:
    chunk_label = f" (Chunk {chunk_idx})" if chunk_idx is not None else ""
    prompt = _INDEX_EXTRACTION_PROMPT.safe_substitute(
        llm_response=llm_response,
        count=count,
        num_summaries=num_summaries,
        max_index=num_summaries - 1,
    )
    try:
        clean = llm.complete(prompt).text
        matches = re.findall(r"\b(\d+)\b", clean)
        indices = [int(i) for i in matches if i.isdigit() and int(i) < num_summaries]
        indices = indices[:count]
        if log:
            if len(indices) == count:
                log.info(
                    f"Case {case_name} (Row {row_index}){chunk_label}: "
                    f"Successfully extracted {count} indices: {indices}"
                )
            else:
                log.warning(
                    f"Case {case_name} (Row {row_index}){chunk_label}: "
                    f"Extracted {len(indices)} indices instead of {count}: {indices}"
                )
        return indices
    except Exception as e:
        if log:
            log.error(
                f"Case {case_name} (Row {row_index}){chunk_label}: Error in index extraction: {e}"
            )
        return []


def filter_chunk(
    llm: LLM,
    summaries: list[str],
    count: int,
    filter_prompt_template: str,
    *,
    log: logging.Logger | None = None,
    case_name: str | None = None,
    row_index: int = 0,
    chunk_idx: Any = None,
    add_missing_fallback_indices: bool = False,
) -> list[str]:
    """Pick the `count` most important summaries via a two-step LLM call.

    `filter_prompt_template` is task-specific. It must reference
    `$count`, `$num_summaries`, and `$formatted_text`.
    `add_missing_fallback_indices` reproduces the extract_case_type
    pre-validation fallback.
    """
    if len(summaries) <= count:
        return summaries

    chunk_label = f" (Chunk {chunk_idx})" if chunk_idx is not None else ""

    formatted_summaries = [
        f"\n\n===== DOCUMENT {i} START =====\n\n{s}\n\n===== DOCUMENT {i} END =====\n\n"
        for i, s in enumerate(summaries)
    ]
    formatted_text = "".join(formatted_summaries)

    prompt = Template(filter_prompt_template).safe_substitute(
        formatted_text=formatted_text,
        num_summaries=len(summaries),
        count=count,
    )
    llm_response = llm.complete(prompt).text

    indices = _extract_clean_indices(
        llm,
        llm_response=llm_response,
        count=count,
        num_summaries=len(summaries),
        log=log,
        case_name=case_name,
        row_index=row_index,
        chunk_idx=chunk_idx,
    )

    if add_missing_fallback_indices and len(indices) < count:
        if log:
            log.warning(
                f"Case {case_name} (Row {row_index}){chunk_label}: "
                f"Not enough valid indices extracted. Adding fallback indices."
            )
        missing = count - len(indices)
        available = [i for i in range(len(summaries)) if i not in indices]
        indices.extend(available[:missing])

    indices = [i for i in indices if 0 <= i < len(summaries)]
    indices = list(dict.fromkeys(indices))

    if len(indices) < count and len(summaries) > count:
        available = [i for i in range(len(summaries)) if i not in indices]
        indices.extend(available[: count - len(indices)])

    return [summaries[i] for i in indices]


def filter_important_summaries(
    llm: LLM,
    summaries: list[str],
    filter_prompt_template: str,
    *,
    log: logging.Logger | None = None,
    case_name: str | None = None,
    row_index: int = 0,
    chunk_size: int = 4,
    final_count: int = 2,
    max_allowed_summaries: int = 10,
    add_missing_fallback_indices: bool = False,
    cap_combined: bool = False,
) -> list[str]:
    """Chunk-then-filter strategy for large summary sets.

    `cap_combined`: when chunked results land below `max_allowed_summaries`,
    still truncate to that limit (incident_date behavior).
    """
    try:
        total = len(summaries)
        if total <= final_count:
            if log:
                log.info(
                    f"Case {case_name} (Row {row_index}): "
                    f"Only {total} summaries available, skipping filtering"
                )
            return summaries

        if total <= chunk_size:
            return filter_chunk(
                llm,
                summaries,
                final_count,
                filter_prompt_template,
                log=log,
                case_name=case_name,
                row_index=row_index,
                add_missing_fallback_indices=add_missing_fallback_indices,
            )

        if log:
            log.info(
                f"Case {case_name} (Row {row_index}): "
                f"Processing {total} summaries in chunks of {chunk_size}"
            )

        num_chunks = (total + chunk_size - 1) // chunk_size
        important_from_chunks: list[str] = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            chunk = summaries[start:end]
            if log:
                log.info(
                    f"Case {case_name} (Row {row_index}): "
                    f"Processing chunk {i + 1}/{num_chunks} with {len(chunk)} summaries"
                )
            chunk_important = filter_chunk(
                llm,
                chunk,
                min(final_count, len(chunk)),
                filter_prompt_template,
                log=log,
                case_name=case_name,
                row_index=row_index,
                chunk_idx=i,
                add_missing_fallback_indices=add_missing_fallback_indices,
            )
            important_from_chunks.extend(chunk_important)

        if len(important_from_chunks) > max_allowed_summaries:
            if log:
                log.info(
                    f"Case {case_name} (Row {row_index}): "
                    f"Filtering combined results from {len(important_from_chunks)} "
                    f"to {max_allowed_summaries}"
                )
            final_important = filter_chunk(
                llm,
                important_from_chunks,
                max_allowed_summaries,
                filter_prompt_template,
                log=log,
                case_name=case_name,
                row_index=row_index,
                chunk_idx="final",
                add_missing_fallback_indices=add_missing_fallback_indices,
            )
        elif cap_combined:
            final_important = important_from_chunks[:max_allowed_summaries]
        else:
            final_important = important_from_chunks

        return final_important

    except Exception as e:
        if log:
            log.error(f"Error filtering summaries for case {case_name} (Row {row_index}): {e}")
        return summaries[:final_count]
