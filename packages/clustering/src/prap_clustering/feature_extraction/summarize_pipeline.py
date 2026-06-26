"""
Generic summarization pipeline for feature extraction.

Simplified from incident_date_extraction/summarize/src/src.py
Accepts OCR text and feature-specific prompts, returns bulletpoint summary.
"""

import concurrent.futures
import logging
import time
from typing import Any

from jinja2 import Template

from prap_clustering._llm import get_llm

logger = logging.getLogger(__name__)


def concatenate_pages(pages: list[str], start_idx: int, num_pages: int) -> tuple:
    """
    Concatenate multiple pages starting from start_idx.

    Args:
        pages: List of page text strings
        start_idx: Starting index
        num_pages: Number of pages to concatenate

    Returns:
        Tuple of (combined_content, pages_consumed)
    """
    combined_content = ""
    current_idx = start_idx
    pages_concatenated = 0
    total_pages = len(pages)

    while pages_concatenated < num_pages and current_idx < total_pages:
        page_content = pages[current_idx].replace("\n", " ")

        # Skip empty pages
        if page_content.strip():
            combined_content += page_content + " "
            pages_concatenated += 1

        current_idx += 1

    return combined_content.strip(), current_idx - start_idx


def create_batches(pages: list[str], batch_size: int) -> list[list[str]]:
    """Create batches of pages for parallel processing."""
    return [pages[i : i + batch_size] for i in range(0, len(pages), batch_size)]


def process_batch(
    batch: list[str], num_pages_to_concat: int, prompts: dict[str, str]
) -> list[dict[str, Any]]:
    """
    Process a batch of pages by concatenating and summarizing them.

    Args:
        batch: List of page text strings
        num_pages_to_concat: Number of pages to concatenate per summary
        prompts: Dictionary of prompt templates from feature-specific prompt module

    Returns:
        List of summary dictionaries with page_content and page_numbers
    """
    results = []
    i = 0

    while i < len(batch):
        concat_pages = []
        page_numbers = []

        # Concatenate num_pages_to_concat non-empty pages
        while len(concat_pages) < num_pages_to_concat and i < len(batch):
            current_page = batch[i].replace("\n", " ")
            if current_page.strip():
                concat_pages.append(current_page)
                page_numbers.append(i)
            i += 1

        if concat_pages:
            original_document = " ".join(concat_pages)
            logger.debug(
                f"Batch: Summarizing pages {page_numbers} (combined length: {len(original_document)} chars)"
            )

            # Initial summary
            template = Template(prompts["summarization"]["page_summary"])
            prompt = template.render(current_page=original_document)
            time.sleep(0.2)  # Rate limiting
            initial_summary = get_llm().complete(prompt).text
            logger.debug(
                f"Batch pages {page_numbers}: Initial summary generated (length: {len(initial_summary)} chars)"
            )

            # Verification step
            template = Template(prompts["summarization"]["page_verification"])
            prompt = template.render(
                original_document=original_document, current_summary=initial_summary
            )
            time.sleep(0.2)  # Rate limiting
            verified_summary = get_llm().complete(prompt).text
            logger.debug(
                f"Batch pages {page_numbers}: Verified summary (length: {len(verified_summary)} chars)"
            )

            results.append(
                {
                    "page_content": verified_summary,
                    "page_numbers": page_numbers,
                }
            )

        if not concat_pages:
            break

    logger.debug(f"Batch processing complete: {len(results)} summaries generated")
    return results


def process_batch_wrapper(args):
    """Wrapper for parallel processing - unpacks arguments."""
    batch, num_pages_to_concat, prompts = args
    return process_batch(batch, num_pages_to_concat, prompts)


def generate_page_summaries(
    pages: list[str],
    prompts: dict[str, str],
    batch_size: int = 20,
    num_pages_to_concat: int = 5,
    max_workers: int = 3,
) -> list[dict[str, Any]]:
    """
    Generate summaries for all pages using parallel processing.

    Args:
        pages: List of page text strings
        prompts: Dictionary of prompt templates from feature-specific prompt module
        batch_size: Number of pages per batch for parallel processing
        num_pages_to_concat: Number of pages to concatenate per summary
        max_workers: Number of parallel workers

    Returns:
        List of page summary dictionaries
    """
    logger.info(
        f"Generating summaries for {len(pages)} pages (batch_size={batch_size}, concat={num_pages_to_concat})"
    )
    batches = create_batches(pages, batch_size)
    logger.info(f"Created {len(batches)} batches for parallel processing")

    results = []
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for each batch
        batch_args = [(batch, num_pages_to_concat, prompts) for batch in batches]

        # Submit all jobs
        futures = [executor.submit(process_batch_wrapper, args) for args in batch_args]
        logger.info(f"Submitted {len(futures)} batch jobs")

        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_results = future.result()
                results.extend(batch_results)
                completed += 1
                logger.info(
                    f"Batch {completed}/{len(futures)} completed ({len(batch_results)} summaries)"
                )
            except Exception as e:
                logger.error(f"Batch processing failed: {e}", exc_info=True)
                completed += 1

    elapsed = time.time() - start_time
    logger.info(f"Summary generation complete: {len(results)} summaries in {elapsed:.2f}s")
    return results


def combine_and_verify_pair(
    summary_1: dict, summary_2: dict, prompts: dict[str, str]
) -> dict[str, Any]:
    """
    Combine two summaries and verify the result.

    Args:
        summary_1: First summary dict with page_content and page_numbers
        summary_2: Second summary dict with page_content and page_numbers
        prompts: Dictionary of prompt templates

    Returns:
        Combined and verified summary dict
    """
    logger.debug(
        f"Combining summaries for pages {summary_1['page_numbers']} and {summary_2['page_numbers']}"
    )

    # Combine
    template = Template(prompts["summarization"]["combine"])
    prompt = template.render(
        summary_1=summary_1["page_content"], summary_2=summary_2["page_content"]
    )
    time.sleep(0.2)  # Rate limiting
    combined_summary = get_llm().complete(prompt).text
    logger.debug(f"Combined summary generated (length: {len(combined_summary)} chars)")

    # Verify
    template = Template(prompts["summarization"]["verification"])
    prompt = template.render(
        current_combined_summary=combined_summary,
        summary_1=summary_1["page_content"],
        summary_2=summary_2["page_content"],
    )
    time.sleep(0.2)  # Rate limiting
    verified_summary = get_llm().complete(prompt).text
    logger.debug(f"Verified combined summary (length: {len(verified_summary)} chars)")

    result = {
        "page_content": verified_summary,
        "page_numbers": summary_1["page_numbers"] + summary_2["page_numbers"],
    }
    logger.debug(f"Combined pages {result['page_numbers']}")
    return result


def combine_and_verify_pair_wrapper(args):
    """Wrapper for parallel combining."""
    summary_1, summary_2, prompts = args
    return combine_and_verify_pair(summary_1, summary_2, prompts)


def parallel_combine(
    summaries: list[dict], prompts: dict[str, str], depth: int = 0, max_workers: int = 3
) -> dict[str, Any]:
    """
    Recursively combine summaries in parallel using a tree structure.

    Args:
        summaries: List of summary dictionaries
        prompts: Dictionary of prompt templates
        depth: Current recursion depth (for logging)
        max_workers: Number of parallel workers for combining

    Returns:
        Single combined summary dictionary
    """
    indent = "  " * depth
    logger.info(f"{indent}Parallel combine: {len(summaries)} summaries at depth {depth}")

    if len(summaries) == 1:
        logger.info(f"{indent}Single summary remaining, returning")
        return summaries[0]
    elif len(summaries) == 2:
        logger.info(f"{indent}Two summaries, combining directly")
        return combine_and_verify_pair(summaries[0], summaries[1], prompts)

    # Sort summaries based on the first page number in each summary
    sorted_summaries = sorted(summaries, key=lambda x: min(x["page_numbers"]))
    logger.debug(f"{indent}Sorted {len(sorted_summaries)} summaries by page number")

    # Pair adjacent summaries
    pairs = [
        (sorted_summaries[i], sorted_summaries[i + 1])
        for i in range(0, len(sorted_summaries) - 1, 2)
    ]
    logger.info(f"{indent}Created {len(pairs)} pairs for parallel processing")

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments
        pair_args = [(pair[0], pair[1], prompts) for pair in pairs]

        futures = [executor.submit(combine_and_verify_pair_wrapper, args) for args in pair_args]
        combined_results = []

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                combined_results.append(result)
                completed += 1
                logger.debug(f"{indent}Pair {completed}/{len(pairs)} combined")
            except Exception as e:
                logger.error(f"{indent}Pair combining failed: {e}", exc_info=True)

    elapsed = time.time() - start_time
    logger.info(
        f"{indent}Parallel combining complete: {len(combined_results)} results in {elapsed:.2f}s"
    )

    # If there's an odd number of summaries, add the last one to the combined results
    if len(sorted_summaries) % 2 != 0:
        logger.info(f"{indent}Adding unpaired summary to results")
        combined_results.append(sorted_summaries[-1])

    # Recursively combine the results
    logger.info(f"{indent}Recursing with {len(combined_results)} summaries")
    return parallel_combine(combined_results, prompts, depth + 1, max_workers)


def combine_summaries(summaries: list[dict], prompts: dict[str, str]) -> str:
    """
    Combine all summaries into a single summary string.

    Args:
        summaries: List of page summary dictionaries
        prompts: Dictionary of prompt templates

    Returns:
        Final combined summary as string
    """
    if not summaries:
        logger.warning("No summaries to combine")
        return ""

    logger.info(f"Starting summary combination with {len(summaries)} summaries")
    final_summary = parallel_combine(summaries, prompts)
    logger.info("Summary combination complete")
    return final_summary.get("page_content", "")


def process_interval(
    interval_index: int,
    interval_pages: list[str],
    prompts: dict[str, str],
    batch_size: int = 20,
    num_pages_to_concat: int = 5,
) -> dict[str, Any]:
    """
    Process a single interval of pages.

    Args:
        interval_index: Index of this interval (for logging)
        interval_pages: List of page strings for this interval
        prompts: Dictionary of prompt templates
        batch_size: Number of pages per batch for parallel processing
        num_pages_to_concat: Number of pages to concatenate per summary

    Returns:
        Combined summary dictionary for this interval
    """
    logger.info(f"Interval {interval_index}: Processing {len(interval_pages)} pages")
    start_time = time.time()

    # Step 1: Generate page summaries in parallel
    page_summaries = generate_page_summaries(
        interval_pages, prompts, batch_size=batch_size, num_pages_to_concat=num_pages_to_concat
    )
    logger.info(f"Interval {interval_index}: Generated {len(page_summaries)} page summaries")

    # Step 2: Combine summaries hierarchically
    combined_summary_text = combine_summaries(page_summaries, prompts)
    logger.info(f"Interval {interval_index}: Combined summaries")

    elapsed = time.time() - start_time
    logger.info(f"Interval {interval_index}: Complete in {elapsed:.2f}s")

    return {"page_content": combined_summary_text, "interval_index": interval_index}


def process_interval_wrapper(args):
    """Wrapper for multiprocessing - unpacks arguments."""
    interval_index, interval_pages, prompts, batch_size, num_pages_to_concat = args
    return process_interval(
        interval_index, interval_pages, prompts, batch_size, num_pages_to_concat
    )


def concatenate_interval_summaries(interval_summaries: list[dict]) -> str:
    """
    Concatenate text content from all interval summaries.

    Args:
        interval_summaries: List of interval summary dictionaries

    Returns:
        Concatenated summary text
    """
    logger.info(f"Concatenating {len(interval_summaries)} interval summaries")

    # Sort by interval index to maintain order
    sorted_summaries = sorted(interval_summaries, key=lambda x: x.get("interval_index", 0))

    all_content = []
    for interval_summary in sorted_summaries:
        content = interval_summary.get("page_content", "")
        if content and content.strip():
            all_content.append(content.strip())
            logger.debug(f"Interval {interval_summary.get('interval_index')}: {len(content)} chars")

    concatenated = "\n\n".join(all_content)
    logger.info(
        f"Concatenated summary: {len(concatenated)} chars from {len(all_content)} intervals"
    )

    return concatenated


def summarize_document(
    ocr_text: str,
    prompts: dict[str, str],
    batch_size: int = 20,
    num_pages_to_concat: int = 5,
    num_intervals: int = 4,
) -> str:
    """
    Main entry point: Summarize a document's OCR text into bulletpoint summary.

    This pipeline:
    1. Splits OCR text into pages
    2. Divides pages into intervals (default: 4)
    3. Processes each interval in parallel (page summaries + combining)
    4. Concatenates all interval summaries into final summary

    Args:
        ocr_text: Full OCR text of document (can be single string or already split)
        prompts: Dictionary of prompt templates from feature-specific prompt module
                 Must have 'summarization' key with: page_summary, page_verification,
                 combine, verification
        batch_size: Number of pages per batch for parallel processing
        num_pages_to_concat: Number of pages to concatenate per summary
        num_intervals: Number of intervals to divide document into (default: 4)

    Returns:
        Final bulletpoint summary as string

    Example:
        >>> from prompts.dates import PROMPTS as DATE_PROMPTS
        >>> summary = summarize_document(ocr_text, DATE_PROMPTS)
    """
    logger.info("Starting document summarization pipeline")

    # Handle input: if string, treat as single page; if list, use as-is
    if isinstance(ocr_text, str):
        pages = [ocr_text]
    else:
        pages = ocr_text

    total_pages = len(pages)
    logger.info(f"Processing {total_pages} pages")

    # Handle empty document
    if total_pages == 0:
        logger.warning("Document has no pages")
        return ""

    # Divide into intervals
    base_interval_size = total_pages // num_intervals
    extra_pages = total_pages % num_intervals
    interval_sizes = [
        base_interval_size + (1 if i < extra_pages else 0) for i in range(num_intervals)
    ]

    logger.info(f"Divided into {num_intervals} intervals: {interval_sizes}")

    # Create interval page lists
    interval_pages_list = []
    start_idx = 0
    for i, size in enumerate(interval_sizes):
        end_idx = start_idx + size
        interval_pages_list.append((i, pages[start_idx:end_idx]))
        start_idx = end_idx

    # Process intervals in parallel using ThreadPoolExecutor
    logger.info("Starting parallel interval processing")
    start_time = time.time()

    interval_summaries = []
    max_workers = 1  # Max 2 parallel intervals to manage API rate limits

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for each interval
        interval_args = [
            (interval_idx, interval_pages, prompts, batch_size, num_pages_to_concat)
            for interval_idx, interval_pages in interval_pages_list
        ]

        # Submit all interval jobs
        futures = [executor.submit(process_interval_wrapper, args) for args in interval_args]
        logger.info(
            f"Submitted {len(futures)} interval jobs to ThreadPoolExecutor (max {max_workers} parallel)"
        )

        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                interval_summaries.append(result)
                completed += 1
                logger.info(f"Interval {completed}/{num_intervals} completed")
            except Exception as e:
                logger.error(f"Interval processing failed: {e}", exc_info=True)
                completed += 1

    elapsed = time.time() - start_time
    logger.info(f"All intervals processed in {elapsed:.2f}s")

    # Concatenate interval summaries
    if not interval_summaries:
        logger.warning("No interval summaries generated")
        return ""

    final_summary = concatenate_interval_summaries(interval_summaries)

    logger.info(f"Summarization complete (final summary: {len(final_summary)} chars)")
    return final_summary


# For backwards compatibility and testing
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test with dates prompts
    from prompts.dates import PROMPTS as DATE_PROMPTS

    sample_ocr = """
    On January 26, 2014, at approximately 6:56 a.m., Officer Butera responded to a call.
    The incident occurred when the suspect fled in a vehicle. At 7:37 a.m., the officer
    discharged their firearm. The investigation was initiated on January 28, 2014.
    """

    summary = summarize_document(sample_ocr, DATE_PROMPTS)
    print("Summary:", summary)
