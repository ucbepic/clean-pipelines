"""Summary-filtering helpers and agency-summary cleaner."""

from __future__ import annotations

import logging
import re
from importlib import resources
from typing import TYPE_CHECKING

import tiktoken
from jinja2 import Template

if TYPE_CHECKING:
    from prap_core.llm import LLM

logger = logging.getLogger("prap.involved_agency.helpers")


def _load_prompt(name: str) -> str:
    return (
        resources.files("prap_involved_agency.prompts")
        .joinpath(f"{name}.txt")
        .read_text(encoding="utf-8")
    )


# ============================================================================
# clean_summaries (from preprocess.py)
# ============================================================================


def clean_summaries(summaries: dict[str, str]) -> dict[str, str]:
    """Clean all page-level summaries. Returns only non-empty results."""
    cleaned = {}
    for slug, text in summaries.items():
        result = _clean(text)
        if result.strip():
            cleaned[slug] = result
    return cleaned


def _clean(text: str) -> str:
    # Remove "Updated summary:" prefixes
    text = re.sub(r"Updated summary:\s*", "", text)

    # Remove "No agencies identified" lines (various phrasings)
    text = re.sub(
        r"(?i)(?:RESPONDING|INVESTIGATING) AGENCIES:\s*\n\s*No (?:responding|investigating) "
        r"agencies identified on this page\.?\s*\n?",
        "",
        text,
    )
    text = re.sub(
        r"(?i)No (?:responding|investigating) agencies identified on this page\.?\s*\n?",
        "",
        text,
    )
    text = re.sub(
        r"(?i)No other agencies are explicitly named or described with investigative or "
        r"responding roles on this page\.?\s*\n?",
        "",
        text,
    )

    # Remove bare section headers with nothing after them
    text = re.sub(r"(?i)RESPONDING AGENCIES:\s*\n\s*(?=\n|INVESTIGATING|$)", "", text)
    text = re.sub(r"(?i)INVESTIGATING AGENCIES:\s*\n\s*(?=\n|RESPONDING|$)", "", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================================
# filter_important_summaries (from helpers.py)
# ============================================================================


def filter_important_summaries(
    llm: LLM,
    summaries: list[str],
    logger=None,
    case_name=None,
    row_index=0,
    chunk_size: int = 4,
    final_count: int = 2,
) -> list[str]:
    """Identify and return the most important summaries using a chunking strategy."""
    try:
        total_summaries = len(summaries)

        if total_summaries <= final_count:
            if logger:
                logger.info(
                    f"Case {case_name} (Row {row_index}): Only {total_summaries} summaries "
                    f"available, skipping filtering"
                )
            return summaries

        if total_summaries <= chunk_size:
            return filter_chunk(llm, summaries, final_count, logger, case_name, row_index)

        if logger:
            logger.info(
                f"Case {case_name} (Row {row_index}): Processing {total_summaries} "
                f"summaries in chunks of {chunk_size}"
            )

        num_chunks = (total_summaries + chunk_size - 1) // chunk_size

        important_from_chunks: list[str] = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_summaries)
            chunk = summaries[start_idx:end_idx]

            if logger:
                logger.info(
                    f"Case {case_name} (Row {row_index}): Processing chunk {i + 1}/{num_chunks} "
                    f"with {len(chunk)} summaries"
                )

            chunk_important = filter_chunk(
                llm,
                chunk,
                min(final_count, len(chunk)),
                logger,
                case_name,
                row_index,
                chunk_idx=i,
            )
            important_from_chunks.extend(chunk_important)

        max_allowed_summaries = 10
        if len(important_from_chunks) > max_allowed_summaries:
            if logger:
                logger.info(
                    f"Case {case_name} (Row {row_index}): Filtering combined results from "
                    f"{len(important_from_chunks)} to {max_allowed_summaries}"
                )

            final_important = filter_chunk(
                llm,
                important_from_chunks,
                max_allowed_summaries,
                logger,
                case_name,
                row_index,
                chunk_idx="final",
            )
        else:
            final_important = important_from_chunks

        if logger:
            encoding = tiktoken.get_encoding("cl100k_base")
            original_total_tokens = sum(len(encoding.encode(s)) for s in summaries)
            filtered_total_tokens = sum(len(encoding.encode(s)) for s in final_important)
            logger.info(
                f"Case {case_name} (Row {row_index}): Reduced from {original_total_tokens} "
                f"to {filtered_total_tokens} tokens "
                f"({filtered_total_tokens / original_total_tokens:.2%} of original)"
            )

        return final_important

    except Exception as e:
        error_message = str(e)
        if logger:
            logger.error(
                f"Error filtering summaries for case {case_name} (Row {row_index}): {error_message}"
            )

        return summaries[:final_count]


def extract_clean_indices(
    llm: LLM,
    llm_response: str,
    count: int,
    num_summaries: int,
    logger=None,
    case_name=None,
    row_index=0,
    chunk_idx=None,
) -> list:
    """Use an llm to extract only the indices from a potentially messy llm call."""

    chunk_label = f" (Chunk {chunk_idx})" if chunk_idx is not None else ""

    extraction_template = Template(_load_prompt("extract_clean_indices"))

    extraction_prompt = extraction_template.render(
        llm_response=llm_response,
        count=count,
        num_summaries=num_summaries,
        max_index=num_summaries - 1,
    )

    try:
        clean_indices_str = llm.complete(extraction_prompt).text

        index_matches = re.findall(r"\b(\d+)\b", clean_indices_str)
        indices = [int(idx) for idx in index_matches if idx.isdigit() and int(idx) < num_summaries]

        indices = indices[:count]

        if logger:
            if len(indices) == count:
                logger.info(
                    f"Case {case_name} (Row {row_index}){chunk_label}: Successfully "
                    f"extracted {count} indices: {indices}"
                )
            else:
                logger.warning(
                    f"Case {case_name} (Row {row_index}){chunk_label}: Extracted {len(indices)} "
                    f"indices instead of {count}: {indices}"
                )

        return indices

    except Exception as e:
        if logger:
            logger.error(
                f"Case {case_name} (Row {row_index}){chunk_label}: Error in index extraction: "
                f"{str(e)}"
            )
        return []


def filter_chunk(
    llm: LLM,
    summaries: list,
    count: int,
    logger=None,
    case_name=None,
    row_index=0,
    chunk_idx=None,
) -> list:
    """First llm identifies important summaries, second llm extracts clean indices."""
    if len(summaries) <= count:
        return summaries

    chunk_label = f" (Chunk {chunk_idx})" if chunk_idx is not None else ""

    formatted_summaries = []
    for i, summary in enumerate(summaries):
        formatted_summary = (
            f"\n\n===== DOCUMENT {i} START =====\n\n{summary}\n\n===== DOCUMENT {i} END =====\n\n"
        )
        formatted_summaries.append(formatted_summary)

    formatted_text = "".join(formatted_summaries)

    prompt_template = Template(_load_prompt("filter_summaries"))

    llm_prompt = prompt_template.render(
        formatted_text=formatted_text, num_summaries=len(summaries), count=count
    )

    llm_response = llm.complete(llm_prompt).text

    indices = extract_clean_indices(
        llm,
        llm_response=llm_response,
        count=count,
        num_summaries=len(summaries),
        logger=logger,
        case_name=case_name,
        row_index=row_index,
        chunk_idx=chunk_idx,
    )

    if len(indices) < count:
        if logger:
            logger.warning(
                f"Case {case_name} (Row {row_index}){chunk_label}: Not enough valid "
                f"indices extracted. Adding fallback indices."
            )

        missing = count - len(indices)
        available_indices = [i for i in range(len(summaries)) if i not in indices]
        indices.extend(available_indices[:missing])

    indices = [i for i in indices if 0 <= i < len(summaries)]
    indices = list(dict.fromkeys(indices))

    if len(indices) < count and len(summaries) > count:
        available_indices = [i for i in range(len(summaries)) if i not in indices]
        indices.extend(available_indices[: count - len(indices)])

    important_summaries = [summaries[i] for i in indices]

    return important_summaries
