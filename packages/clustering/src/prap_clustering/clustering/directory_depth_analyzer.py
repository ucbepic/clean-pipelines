import logging
import random
from pathlib import Path

from pydantic import BaseModel, Field

from prap_clustering._llm import get_llm

logger = logging.getLogger(__name__)


class _MinDepthResponse(BaseModel):
    """Structured response for analyze_directory_depth."""

    min_depth: int = Field(ge=0, le=3)


DIRECTORY_DEPTH_PROMPT = """Analyze these sample filepaths from a police document repository.

SAMPLE PATHS:
{sample_paths}

Determine the minimum shared directory depth (min_depth) for safely merging singleton documents:

- **min_depth = 0** (DEFAULT): Merge singletons sharing same parent directory
  Use when paths are case-specific: "Agency/Case 2018-0167/file.pdf"

- **min_depth = 1**: Require one more level of nesting
  Use when you see generic folders: "Agency/Data/file.pdf", "Agency/Documents/file.pdf"

- **min_depth = 2+**: Require even more nesting (rare, very flat structure)

Be conservative - only use values > 0 if you clearly see generic directory names like "Data", "Documents", "Files", "Export", "Batch"."""


def analyze_directory_depth(
    singleton_filepaths: list[str], sample_size: int = 20, output_dir: str = "../../data/output"
) -> int:
    """
    Determine minimum directory depth for merging singletons using LLM analysis.

    Args:
        singleton_filepaths: List of filepaths for singleton documents
        sample_size: Number of paths to sample for analysis
        output_dir: Directory to save sampled paths text file

    Returns:
        int: Minimum shared directory depth (0 = merge immediately, 1+ = require more nesting)
    """
    if not singleton_filepaths:
        return 0

    sample_count = min(sample_size, len(singleton_filepaths))
    sampled_paths = random.sample(singleton_filepaths, sample_count)

    # Save sampled paths to text file for manual review
    from pathlib import Path

    output_path = Path(output_dir) / "llm_directory_depth_analysis_sample.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"Sampled {sample_count} paths from {len(singleton_filepaths)} singletons\n")
        f.write("=" * 80 + "\n\n")
        for path in sampled_paths:
            f.write(f"{path}\n")

    logger.info(f"Saved sampled paths to: {output_path}")

    formatted_paths = "\n".join(f"- {path}" for path in sampled_paths)
    prompt = DIRECTORY_DEPTH_PROMPT.format(sample_paths=formatted_paths)

    try:
        # Structured output via response_format=<PydanticModel>.
        response = get_llm().complete(prompt, response_format=_MinDepthResponse)
        min_depth = int(response.min_depth)
        min_depth = max(0, min(3, min_depth))

        # Append result to the same file
        with open(output_path, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"LLM RESULT: min_depth = {min_depth}\n")

        logger.info(f"LLM determined min_depth = {min_depth}")
        return min_depth
    except Exception as e:
        logger.error(f"Error in LLM analysis: {e}, defaulting to 0")

        # Log error to file
        with open(output_path, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ERROR: {e}\n")
            f.write("Defaulting to min_depth = 0\n")

        return 0


def get_grouping_directory(filepath: str, min_depth: int) -> str:
    """
    Get directory path for grouping based on min_depth.

    Args:
        filepath: Full file path
        min_depth: How many levels up from parent (0 = parent, 1 = grandparent, etc.)

    Returns:
        Directory path for grouping
    """
    if not filepath:
        return ""

    path = Path(filepath)
    parent = path.parent

    for _ in range(min_depth):
        parent = parent.parent
        if parent == Path("."):
            break

    return str(parent)
