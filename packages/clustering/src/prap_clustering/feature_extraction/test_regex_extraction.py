"""
Test script for rerun_regex_extraction.py

Validates that the regex patterns correctly extract and filter case IDs and names.
"""

import sys

from .regex_extract_fp_fn import (
    extract_ids_from_metadata,
    extract_names_from_metadata,
    should_exclude_case_id,
    should_exclude_name,
)


def test_case_id_exclusions():
    """Test that problematic case IDs are correctly excluded."""

    print("\n" + "=" * 80)
    print("TESTING CASE ID EXCLUSIONS")
    print("=" * 80)

    # IDs that SHOULD be blocked
    should_block = [
        ("TF967B", "Traffic form with letter suffix"),
        ("TF1050A", "Traffic form with letter suffix"),
        ("TF967", "Traffic form without letter"),
        ("F313", "Form number"),
        ("F990", "Form number"),
        ("FMT5", "Form code"),
        ("109", "Too short - 3 digits"),
        ("1208", "Too short - 4 digits"),
        ("190416", "Too short - 6 digits"),
        ("V1", "Single letter + digit"),
        ("V2", "Single letter + digit"),
        ("OISLL", "Abbreviation without numbers"),
        ("FILE20150803011837", "File timestamp"),
    ]

    # IDs that SHOULD be extracted
    should_extract = [
        ("IA2018-0167", "Specific IA case with year"),
        ("IAD552", "IAD with substantial number"),
        ("LOP170405000550", "Long alphanumeric ID"),
        ("OIS-2018", "OIS with year"),
        ("22-1460", "Numeric with year and case number"),
        ("2018-4567", "Numeric with year and case number"),
    ]

    print("\nCase IDs that SHOULD be blocked:")
    print("-" * 80)
    failures = []
    for case_id, reason in should_block:
        is_blocked = should_exclude_case_id(case_id)
        status = "✓ BLOCKED" if is_blocked else "✗ NOT BLOCKED (FAIL)"
        print(f"  {case_id:30s} | {reason:35s} | {status}")
        if not is_blocked:
            failures.append(f"FAILED to block: {case_id} ({reason})")

    print("\nCase IDs that SHOULD be extracted:")
    print("-" * 80)
    for case_id, reason in should_extract:
        is_blocked = should_exclude_case_id(case_id)
        status = "✓ EXTRACTED" if not is_blocked else "✗ BLOCKED (FAIL)"
        print(f"  {case_id:30s} | {reason:35s} | {status}")
        if is_blocked:
            failures.append(f"FAILED to extract: {case_id} ({reason})")

    return failures


def test_name_exclusions():
    """Test that problematic names are correctly excluded."""

    print("\n" + "=" * 80)
    print("TESTING NAME EXCLUSIONS")
    print("=" * 80)

    # Names that SHOULD be blocked
    should_block = [
        ("b 6 b", "Redacted name"),
        ("832 7 b 6 b", "Redacted with statute code"),
        ("832 7 b 6 a", "Redacted with statute code variant"),
        ("subject b 6 b", "Redacted subject"),
        ("witness 1", "Generic witness label"),
        ("witness 3", "Generic witness label"),
        ("unnamed suspect", "Generic label"),
        ("victim", "Generic label"),
        ("suspect", "Generic label"),
        ("name redacted", "Redacted name"),
        ("confidential informant ci", "Generic CI label"),
        ("garcia", "Single word last name only"),
        ("loman", "Single word last name only"),
        ("joyner", "Single word last name only"),
    ]

    # Names that SHOULD be extracted
    should_extract = [
        ("John Smith", "Full name"),
        ("Officer John Smith", "Name with title"),
        ("Hourigan, K", "Last name, initial format"),
        ("Michael Wade Butler Randle", "Multi-word name"),
        ("Det. Jane Doe", "Name with detective title"),
        ("Garcia, M", "Last name with initial - OK"),
        ("Benjamin Olson", "Full name"),
    ]

    print("\nNames that SHOULD be blocked:")
    print("-" * 80)
    failures = []
    for name, reason in should_block:
        is_blocked = should_exclude_name(name)
        status = "✓ BLOCKED" if is_blocked else "✗ NOT BLOCKED (FAIL)"
        print(f"  {name:30s} | {reason:35s} | {status}")
        if not is_blocked:
            failures.append(f"FAILED to block: {name} ({reason})")

    print("\nNames that SHOULD be extracted:")
    print("-" * 80)
    for name, reason in should_extract:
        is_blocked = should_exclude_name(name)
        status = "✓ EXTRACTED" if not is_blocked else "✗ BLOCKED (FAIL)"
        print(f"  {name:30s} | {reason:35s} | {status}")
        if is_blocked:
            failures.append(f"FAILED to extract: {name} ({reason})")

    return failures


def test_extraction_from_filepath():
    """Test extraction from realistic filepath examples."""

    print("\n" + "=" * 80)
    print("TESTING EXTRACTION FROM FILEPATHS")
    print("=" * 80)

    test_cases = [
        {
            "path": "/folder/IA2018-0167_Report/document.pdf",
            "expected_ids": ["IA2018-0167"],
            "expected_names": [],
            "reason": "IA case in folder name",
        },
        {
            "path": "/folder/TF967B_Traffic_Report.pdf",
            "expected_ids": [],  # Should be blocked
            "expected_names": [],
            "reason": "TF code should be blocked",
        },
        {
            "path": "/cases/190416/document.pdf",
            "expected_ids": [],  # Should be blocked (too short)
            "expected_names": [],
            "reason": "Short numeric ID should be blocked",
        },
        {
            "path": "/FILE20150803011837/report.pdf",
            "expected_ids": [],  # Should be blocked
            "expected_names": [],
            "reason": "FILE timestamp should be blocked",
        },
        {
            "path": "/cases/2018-4567_Smith_John/report.pdf",
            "expected_ids": ["2018-4567"],
            "expected_names": [],  # "Smith" and "John" are single words, should be blocked
            "reason": "Valid numeric case ID",
        },
        {
            "path": "/LOP170405000550_Investigation/file.pdf",
            "expected_ids": ["LOP170405000550"],
            "expected_names": [],
            "reason": "Long alphanumeric case ID",
        },
    ]

    failures = []
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['reason']}")
        print(f"  Path: {test['path']}")

        extracted_ids = extract_ids_from_metadata(test["path"])
        extracted_names = extract_names_from_metadata(test["path"])

        print(f"  Expected IDs:   {test['expected_ids']}")
        print(f"  Extracted IDs:  {extracted_ids}")

        # Check IDs match
        if set(extracted_ids) == set(test["expected_ids"]):
            print("  ✓ Case IDs match")
        else:
            print("  ✗ Case IDs DO NOT match (FAIL)")
            failures.append(
                f"Test {i}: Case IDs mismatch - got {extracted_ids}, expected {test['expected_ids']}"
            )

        print(f"  Expected Names: {test['expected_names']}")
        print(f"  Extracted Names: {extracted_names}")

        # Check names match
        if set(extracted_names) == set(test["expected_names"]):
            print("  ✓ Names match")
        else:
            print("  ✗ Names DO NOT match (FAIL)")
            failures.append(
                f"Test {i}: Names mismatch - got {extracted_names}, expected {test['expected_names']}"
            )

    return failures


def test_extraction_from_filename():
    """Test extraction from realistic filename examples."""

    print("\n" + "=" * 80)
    print("TESTING EXTRACTION FROM FILENAMES")
    print("=" * 80)

    test_cases = [
        {
            "filename": "IA2018-0167_Investigation_Report.pdf",
            "expected_ids": ["IA2018-0167"],
            "reason": "IA case in filename",
        },
        {
            "filename": "V1_Video_Recording.mp4",
            "expected_ids": [],  # Should be blocked
            "reason": "V1 should be blocked (too generic)",
        },
        {
            "filename": "Report_TF1050A.pdf",
            "expected_ids": [],  # Should be blocked
            "reason": "TF code should be blocked",
        },
        {
            "filename": "Case_22-1460_Final.pdf",
            "expected_ids": ["22-1460"],
            "reason": "Valid year-case number format",
        },
    ]

    failures = []
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['reason']}")
        print(f"  Filename: {test['filename']}")

        extracted_ids = extract_ids_from_metadata(test["filename"])

        print(f"  Expected: {test['expected_ids']}")
        print(f"  Extracted: {extracted_ids}")

        if set(extracted_ids) == set(test["expected_ids"]):
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
            failures.append(
                f"Filename test {i}: got {extracted_ids}, expected {test['expected_ids']}"
            )

    return failures


def main():
    """Run all tests and report results."""

    print("\n" + "=" * 80)
    print("REGEX EXTRACTION TEST SUITE")
    print("=" * 80)

    all_failures = []

    # Run all test suites
    all_failures.extend(test_case_id_exclusions())
    all_failures.extend(test_name_exclusions())
    all_failures.extend(test_extraction_from_filepath())
    all_failures.extend(test_extraction_from_filename())

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if not all_failures:
        print("✓ ALL TESTS PASSED")
        print("\nThe regex extraction patterns are working correctly.")
        print("You can now run rerun_regex_extraction.py on the full dataset.")
        return 0
    else:
        print(f"✗ {len(all_failures)} TESTS FAILED")
        print("\nFailures:")
        for i, failure in enumerate(all_failures, 1):
            print(f"  {i}. {failure}")
        print("\nPlease fix the extraction patterns before running on the full dataset.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
