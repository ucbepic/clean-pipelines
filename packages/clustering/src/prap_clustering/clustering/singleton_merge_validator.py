"""
Validates whether singleton documents in the same directory should be merged.

Uses LLM to determine if files in a directory appear to be from the same case/incident
or if they're just random files dumped together.
"""

import logging
from pathlib import Path

from pydantic import BaseModel

from prap_clustering._llm import get_llm

logger = logging.getLogger(__name__)


class _MergeDecisionResponse(BaseModel):
    """Structured response for should_merge_singletons."""

    should_merge: bool


MERGE_VALIDATION_PROMPT = """You are validating whether documents in the same directory should be merged into a cluster.

DIRECTORY: {directory_path}

FILES TO MERGE ({num_files} files):
{file_list}

TASK: Determine if these files appear to be from the SAME case/incident, or if they're unrelated files that happen to be in the same directory.

MERGE if:
- Files share a common case identifier (e.g., "16115838_report.pdf", "16115838_photo.jpg")
- Files appear to be parts of the same incident (e.g., "report.pdf", "attachment.pdf", "evidence.pdf")
- Directory name itself suggests a specific case (e.g., "Case 2018-0167", "Investigation-Smith")

DON'T MERGE if:
- Files have different case identifiers (e.g., "16115838.pdf", "22107584.pdf", "18144508.pdf")
- Files appear to be from unrelated cases dumped in a generic directory
- No clear pattern suggesting files belong together

Return your decision."""


def should_merge_singletons(directory_path: str, filenames: list[str], max_files_to_show: int = 50, output_log_path: str = None) -> bool:
    """
    Determine if singleton files in a directory should be merged into a cluster.

    Args:
        directory_path: Path to the directory containing the files
        filenames: List of full file paths (gdrive_path/gdrive_name)
        max_files_to_show: Maximum number of files to show in prompt (default: 50)
        output_log_path: Optional path to log file for recording decisions

    Returns:
        bool: True if files should be merged, False otherwise
    """
    if not filenames or len(filenames) < 2:
        return False

    # Extract just the filenames (not full paths) for LLM prompt
    just_filenames = [Path(f).name for f in filenames]

    # Limit files shown to avoid context overflow
    files_to_show = just_filenames[:max_files_to_show]
    if len(just_filenames) > max_files_to_show:
        remaining = len(just_filenames) - max_files_to_show
        file_list = "\n".join(f"- {fn}" for fn in files_to_show)
        file_list += f"\n... and {remaining} more files"
    else:
        file_list = "\n".join(f"- {fn}" for fn in files_to_show)

    # Build full path list for logging (human review)
    full_paths_to_show = filenames[:max_files_to_show]
    if len(filenames) > max_files_to_show:
        remaining = len(filenames) - max_files_to_show
        full_path_list = "\n".join(f"- {fp}" for fp in full_paths_to_show)
        full_path_list += f"\n... and {remaining} more files"
    else:
        full_path_list = "\n".join(f"- {fp}" for fp in full_paths_to_show)

    prompt = MERGE_VALIDATION_PROMPT.format(
        directory_path=directory_path,
        num_files=len(just_filenames),
        file_list=file_list
    )

    try:
        # Structured output via response_format=<PydanticModel>.
        response = get_llm().complete(prompt, response_format=_MergeDecisionResponse)
        should_merge = bool(response.should_merge)

        # Log the decision
        decision_str = 'MERGE' if should_merge else 'SKIP'
        logger.info(f"  {directory_path}: {decision_str} ({len(just_filenames)} files)")

        # Write to log file if provided (use full paths for human review)
        if output_log_path:
            with open(output_log_path, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"DIRECTORY: {directory_path}\n")
                f.write(f"FILES: {len(filenames)} total\n")
                f.write(f"{full_path_list}\n")
                f.write(f"\nDECISION: {decision_str}\n")

        return bool(should_merge)

    except Exception as e:
        logger.error(f"  Error validating {directory_path}: {e}")
        # Conservative default: don't merge if uncertain
        logger.info(f"  {directory_path}: SKIP (error, defaulting to no merge)")

        # Log error to file if provided (use full paths for human review)
        if output_log_path:
            with open(output_log_path, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"DIRECTORY: {directory_path}\n")
                f.write(f"FILES: {len(filenames)} total\n")
                f.write(f"{full_path_list}\n")
                f.write(f"\nERROR: {e}\n")
                f.write("DECISION: SKIP (error)\n")

        return False
