"""Sensitive persons classifier.

LLM calls go through prap_core.llm.LLM (threaded in from pipeline.run()).
Prompts are loaded from prap_redactions/prompts/ via prap_core.prompts.PromptDir.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources

import pandas as pd
from fuzzywuzzy import fuzz
from prap_core.llm import LLM
from prap_core.prompts import PromptDir

from ..schemas import (
    AddressResponse,
    FormattedOutput,
    FormattedPerson,
    NameExtractionResponse,
    VerificationResponse,
)

logger = logging.getLogger("prap.redactions.sensitive_persons")


def _prompt_dir() -> PromptDir:
    return PromptDir(str(resources.files("prap_redactions").joinpath("prompts")))


def identify_names(llm: LLM, text: str) -> NameExtractionResponse:
    """Identify potential names of victims and witnesses from the text with improved accuracy."""
    prompt = _prompt_dir().render("identify_names", source_text=text)
    return llm.complete(prompt, response_format=NameExtractionResponse)


def remove_irrelevant_persons(
    llm: LLM, text: str, verified_names_response: NameExtractionResponse
) -> VerificationResponse:
    """Filter out individuals who are not relevant to victim/witness tracking."""
    verified_names_json = verified_names_response.model_dump_json(indent=2)
    prompt = _prompt_dir().render(
        "remove_irrelevant_persons",
        source_text=text,
        verified_names=verified_names_json,
    )
    return llm.complete(prompt, response_format=VerificationResponse)


def format_output(llm: LLM, verification_response: VerificationResponse) -> FormattedOutput:
    """Format the verified names into the required output structure with enhanced categories."""
    verified_names_json = verification_response.model_dump_json(indent=2)
    prompt = _prompt_dir().render("format_output", verified_names=verified_names_json)
    return llm.complete(prompt, response_format=FormattedOutput)


def find_addresses(llm: LLM, persons: list[FormattedPerson], page_text: str) -> AddressResponse:
    """Use LLM to determine if addresses for multiple persons are in the text"""
    names_list = [person.name for person in persons]
    names_formatted = ", ".join(names_list)
    prompt = _prompt_dir().render("find_addresses", names=names_formatted, page_text=page_text)
    return llm.complete(prompt, response_format=AddressResponse)


def deduplicate_by_fuzzy_matching(results_df, threshold=85):
    """
    Deduplicate name records using fuzzy matching while preserving all important information
    by concatenating unique values for each field.
    """
    print("Starting fuzzy deduplication of name results...")

    df = results_df.copy()

    # If DataFrame is empty or has only one row, return as is
    if len(df) <= 1:
        return df

    # Ensure person_type is always a list
    df["person_type"] = df["person_type"].apply(
        lambda x: x if isinstance(x, list) else [x] if isinstance(x, str) else []
    )

    # Step 1: Normalize names (lowercase, remove extra spaces, etc.)
    df["normalized_name"] = df["name"].apply(
        lambda x: " ".join(str(x).lower().strip().split()) if pd.notna(x) else ""
    )

    # Step 2: Create clusters of similar names
    name_clusters = {}
    processed_indices = set()

    # For each name, find similar names and create clusters
    for i, row in df.iterrows():
        if i in processed_indices:
            continue

        name = row["normalized_name"]
        if not name:
            continue

        cluster = [i]
        processed_indices.add(i)

        # Compare with all other names
        for j, other_row in df.iterrows():
            if j in processed_indices:
                continue

            other_name = other_row["normalized_name"]
            if not other_name:
                continue

            # Use token_sort_ratio to handle word order differences
            similarity = fuzz.token_sort_ratio(name, other_name)

            if similarity >= threshold:
                cluster.append(j)
                processed_indices.add(j)

        name_clusters[name] = cluster

    print(f"Created {len(name_clusters)} name clusters from {len(df)} original records")

    # Step 3: Merge clusters into deduplicated records
    deduplicated = []

    # Define which fields should be collected as unique values across records
    fields_to_concatenate = [
        "document_name",
        "gdrive_id",
        "reasoning",
        "address",
        "address_context",
        "sha1",
    ]

    # Define which fields should be collected as sets of unique values
    fields_for_sets = ["person_type", "provisional_case_name", "page_number"]

    # Define which fields should use the best/highest value
    best_value_fields = {
        "confidence": {"high": 3, "medium": 2, "low": 1},  # Higher is better
        "address_found": {True: 1, False: 0},  # True is better than False
    }

    for representative_name, indices in name_clusters.items():
        cluster_df = df.iloc[indices].copy()

        # Choose the most frequent name form as canonical
        name_counts = cluster_df["name"].value_counts()
        canonical_name = name_counts.index[0] if not name_counts.empty else representative_name

        # Initialize the combined record with the canonical name
        combined_record = {"name": canonical_name, "original_count": len(indices)}

        # Process fields that should be unique sets
        for field in fields_for_sets:
            if field in cluster_df.columns:
                if field == "person_type":
                    # Special handling for person_type which is already a list
                    all_values = []
                    for values in cluster_df[field]:
                        if isinstance(values, list):
                            all_values.extend(values)
                        elif pd.notna(values):
                            all_values.append(values)
                    combined_record[field] = list(set(all_values))
                else:
                    # For other fields, gather unique values
                    unique_values = set(cluster_df[field].dropna())
                    if field == "page_number":
                        # For page numbers, keep them as a list of integers
                        combined_record[field] = sorted(list(unique_values))
                    else:
                        combined_record[field] = list(unique_values)

        # Process fields that should be concatenated
        for field in fields_to_concatenate:
            if field in cluster_df.columns:
                # Gather non-empty values
                values = [str(v) for v in cluster_df[field].dropna() if str(v).strip()]
                # Remove duplicates
                unique_values = list(set(values))
                # Join with semicolons
                combined_record[field] = "; ".join(unique_values) if unique_values else ""

        # Process fields that should use the best value
        for field, value_map in best_value_fields.items():
            if field in cluster_df.columns:
                # Map values to their numeric equivalents
                mapped_values = cluster_df[field].map(
                    lambda x, value_map=value_map: value_map.get(x, 0) if pd.notna(x) else 0
                )
                # Find the best value
                if not mapped_values.empty:
                    best_idx = mapped_values.idxmax()
                    # Use the original value, not the mapped one
                    combined_record[field] = cluster_df.loc[best_idx, field]
                else:
                    # Default values if the field is empty
                    combined_record[field] = (
                        False
                        if field == "address_found"
                        else "low"
                        if field == "confidence"
                        else None
                    )

        # Create a source identifiers list for traceability
        sources = cluster_df.apply(
            lambda x: (
                f"{x.get('provisional_case_name', 'unknown')}:"
                f"{x.get('gdrive_id', 'unknown')}:"
                f"{x.get('page_number', 'unknown')}"
            ),
            axis=1,
        ).tolist()
        combined_record["sources"] = list(set(sources))

        deduplicated.append(combined_record)

    print(f"Reduced to {len(deduplicated)} unique persons after deduplication")

    # Convert to DataFrame
    deduped_df = pd.DataFrame(deduplicated)

    # Fix person_type format - ensure it's a string
    if "person_type" in deduped_df.columns:
        deduped_df["person_type"] = deduped_df["person_type"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    # Fix page_number format - ensure it's a string
    if "page_number" in deduped_df.columns:
        deduped_df["page_number"] = deduped_df["page_number"].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
        )

    # Fix provisional_case_name format - ensure it's a string
    if "provisional_case_name" in deduped_df.columns:
        deduped_df["provisional_case_name"] = deduped_df["provisional_case_name"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    # Ensure all columns from original DataFrame are present
    for col in results_df.columns:
        if col not in deduped_df.columns and col != "normalized_name":
            deduped_df[col] = None

    return deduped_df


def _process_single_page(llm: LLM, page):
    """
    Process a single page to extract names. Helper function for parallel processing.
    """
    if "text" not in page or not page["text"]:
        return []

    page_text = page["text"]
    page_number = page.get("page_number", 0)
    document_name = page.get("document_name", "Unknown")
    gdrive_id = page.get("gdrive_id", "Unknown")
    sha1 = page.get("sha1", "")
    provisional_case_name = page.get("provisional_case_name", "")

    page_results = []

    try:
        # Step 1: Identify potential names on this page
        identified_names = identify_names(llm, page_text)

        # Step 2: Remove irrelevant persons
        irrelevant_removed_names = remove_irrelevant_persons(llm, page_text, identified_names)

        logger.debug(
            f"Page {page_number}: Remove irrelevant persons output: {irrelevant_removed_names}"
        )

        # Step 3: Format the output
        formatted_output = format_output(llm, irrelevant_removed_names)
        logger.debug(f"Page {page_number}: Formatted output: {formatted_output}")

        # Extract the persons from the formatted output
        if formatted_output.extracted_persons:
            persons = formatted_output.extracted_persons

            # If there are persons found, look for addresses
            if persons:
                addresses_response = find_addresses(llm, persons, page_text)
                addresses_by_name = {entry.name: entry for entry in addresses_response.addresses}
            else:
                addresses_by_name = {}

            # Add page metadata to each person and append to results
            for person in persons:
                person_entry = {
                    "name": person.name,
                    "person_type": person.person_type,
                    "page_number": page_number,
                    "document_name": document_name,
                    "gdrive_id": gdrive_id,
                    "sha1": sha1,
                    "provisional_case_name": provisional_case_name,
                    "reasoning": person.reasoning,
                    "confidence": person.confidence,
                }

                # Add address information if available
                address_info = addresses_by_name.get(person.name)
                if address_info:
                    person_entry["address_found"] = address_info.address_found
                    person_entry["address"] = address_info.address or ""
                    person_entry["address_context"] = address_info.context or ""
                else:
                    person_entry["address_found"] = False
                    person_entry["address"] = ""
                    person_entry["address_context"] = ""

                page_results.append(person_entry)

    except Exception as e:
        logger.error(f"Error processing page {page_number} in {document_name}: {str(e)}")

    return page_results


def extract_names(llm: LLM, ocr_text_pages, max_workers=10):
    """
    Extract names from OCR text pages in parallel and return a structured DataFrame.

    Args:
        llm: prap_core.llm.LLM instance
        ocr_text_pages: List of page dictionaries with OCR text
        max_workers: Maximum number of parallel threads (default: 10)
    """
    # Initialize an empty list to store results from all pages
    all_results = []

    # Filter out empty pages
    valid_pages = [page for page in ocr_text_pages if page.get("text")]

    if not valid_pages:
        logger.info("No valid pages to process")
        columns = [
            "name",
            "person_type",
            "page_number",
            "document_name",
            "gdrive_id",
            "sha1",
            "provisional_case_name",
            "reasoning",
            "confidence",
            "address_found",
            "address",
            "address_context",
        ]
        return pd.DataFrame(columns=columns)

    logger.info(f"Processing {len(valid_pages)} pages in parallel with {max_workers} workers...")

    max_workers = 7

    # Process pages in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_page = {
            executor.submit(_process_single_page, llm, page): page for page in valid_pages
        }

        # Collect results as they complete
        for future in as_completed(future_to_page):
            page = future_to_page[future]
            try:
                page_results = future.result()
                all_results.extend(page_results)
                logger.info(
                    f"Completed page {page.get('page_number', '?')}: "
                    f"found {len(page_results)} persons"
                )
            except Exception as e:
                page_num = page.get("page_number", "?")
                doc_name = page.get("document_name", "Unknown")
                logger.error(f"Failed to process page {page_num} in {doc_name}: {str(e)}")

    # If no results, return an empty DataFrame with the proper columns
    if not all_results:
        columns = [
            "name",
            "person_type",
            "page_number",
            "document_name",
            "gdrive_id",
            "sha1",
            "provisional_case_name",
            "reasoning",
            "confidence",
            "address_found",
            "address",
            "address_context",
        ]
        return pd.DataFrame(columns=columns)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Deduplicate the results using the existing function
    if len(results_df) > 1:
        results_df = deduplicate_by_fuzzy_matching(results_df)

    # Ensure all expected columns are present
    expected_columns = [
        "name",
        "person_type",
        "page_number",
        "document_name",
        "gdrive_id",
        "sha1",
        "provisional_case_name",
        "reasoning",
        "confidence",
        "address_found",
        "address",
        "address_context",
    ]

    for col in expected_columns:
        if col not in results_df.columns:
            results_df[col] = None

    return results_df
