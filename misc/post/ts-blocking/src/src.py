import argparse
import os
from datetime import datetime

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create blocking keys for officer record linkage")
    parser.add_argument(
        "--input", type=str, default="../data/train-data/data/output/labeled_data.csv"
    )
    parser.add_argument("--output", type=str, default="data/output/labeled_data_w_blocks.csv")
    parser.add_argument("--prefix-len", type=int, default=1)
    parser.add_argument("--suffix-len", type=int, default=2)
    return parser.parse_args()


def parse_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def check_date_overlap(incident_date, start_date, end_date):
    incident = parse_date(incident_date)
    start = parse_date(start_date)
    end = parse_date(end_date)

    if incident is None:
        return False

    # If no start date, only check end date
    if start is None and end is not None:
        return incident <= end

    # If no end date, only check start date
    if end is None and start is not None:
        return incident >= start

    # If both dates present, check overlap
    if start is not None and end is not None:
        return start <= incident <= end

    # If no dates to compare against
    return False


def safe_str(value):
    """Convert value to string, handling NaN and None values."""
    if pd.isna(value):
        return ""
    return str(value)


def create_blocking_keys(row, prefix_len=1, suffix_len=2):
    # Name fields from incident
    incident_first = safe_str(row.get("incident_first_name", ""))
    incident_middle = safe_str(row.get("incident_middle_name", ""))
    incident_last = safe_str(row.get("incident_last_name", ""))
    incident_name = safe_str(row.get("incident_name", ""))
    incident_suffix = safe_str(row.get("incident_suffix", ""))

    # Name fields from post
    post_first = safe_str(row.get("post_first_name", ""))
    post_middle = safe_str(row.get("post_middle_name", ""))
    post_last = safe_str(row.get("post_last_name", ""))
    post_name = safe_str(row.get("post_name", ""))
    post_suffix = safe_str(row.get("post_suffix", ""))

    # Date fields
    incident_date = row.get("incident_date")
    start_date = row.get("post_start_date")
    end_date = row.get("post_end_date")

    keys = []

    # Only proceed with name-based blocking if both records have names
    has_incident_names = bool(incident_first and incident_last)
    has_post_names = bool(post_first and post_last)

    if not (has_incident_names and has_post_names):
        return ["missing_required_names"]

    # Date-based blocking
    if check_date_overlap(incident_date, start_date, end_date):
        keys.append("date_valid")

    # Incident name blocking
    if len(incident_first) >= prefix_len:
        keys.append(f"i_first_pre_{incident_first[:prefix_len].lower()}")
    if len(incident_first) >= suffix_len:
        keys.append(f"i_first_suf_{incident_first[-suffix_len:].lower()}")

    if len(incident_last) >= prefix_len:
        keys.append(f"i_last_pre_{incident_last[:prefix_len].lower()}")
    if len(incident_last) >= suffix_len:
        keys.append(f"i_last_suf_{incident_last[-suffix_len:].lower()}")

    # Post name blocking
    if len(post_first) >= prefix_len:
        keys.append(f"p_first_pre_{post_first[:prefix_len].lower()}")
    if len(post_first) >= suffix_len:
        keys.append(f"p_first_suf_{post_first[-suffix_len:].lower()}")

    if len(post_last) >= prefix_len:
        keys.append(f"p_last_pre_{post_last[:prefix_len].lower()}")
    if len(post_last) >= suffix_len:
        keys.append(f"p_last_suf_{post_last[-suffix_len:].lower()}")

    # Middle name blocking (if both present)
    if incident_middle and post_middle:
        keys.append(f"middle_{incident_middle[0].lower()}_{post_middle[0].lower()}")

    # Suffix blocking (if both present)
    if incident_suffix and post_suffix:
        keys.append(f"suffix_{incident_suffix[0].lower()}_{post_suffix[0].lower()}")

    # Full name blocking
    if incident_name:
        if len(incident_name) >= prefix_len:
            keys.append(f"i_name_pre_{incident_name[:prefix_len].lower()}")
        if len(incident_name) >= suffix_len:
            keys.append(f"i_name_suf_{incident_name[-suffix_len:].lower()}")

    if post_name:
        if len(post_name) >= prefix_len:
            keys.append(f"p_name_pre_{post_name[:prefix_len].lower()}")
        if len(post_name) >= suffix_len:
            keys.append(f"p_name_suf_{post_name[-suffix_len:].lower()}")

    return keys or ["missing"]


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    df["blocking_keys"] = df.apply(
        lambda row: "|".join(
            create_blocking_keys(row, prefix_len=args.prefix_len, suffix_len=args.suffix_len)
        ),
        axis=1,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}")


if __name__ == "__main__":
    main()
