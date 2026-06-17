"""Ground-truth date helpers."""

import logging

import pandas as pd

logger = logging.getLogger("prap.incident_date.helpers")


def format_date_from_components(year, month, day, source_name) -> str | None:
    """Format `YYYY-MM-DD` from year/month/day; None if any component is empty/NaN."""
    if pd.isna(year) or pd.isna(month) or pd.isna(day) or year == "" or month == "" or day == "":
        return None
    try:
        y = str(int(float(year)))
        m = str(int(float(month))).zfill(2)
        d = str(int(float(day))).zfill(2)
        return f"{y}-{m}-{d}"
    except (ValueError, TypeError) as e:
        logger.warning(f"Error formatting date from {source_name} for current row: {e}")
        logger.warning(f"Values: year={year}, month={month}, day={day}")
        return None


def parse_misconduct_date_ranges(misconduct_ranges: str) -> list[str]:
    """Parse comma-separated YYYYMMDD or YYYYMMDD-YYYYMMDD into a list of YYYY-MM-DD."""
    dates: list[str] = []
    if pd.isna(misconduct_ranges) or misconduct_ranges == "":
        return dates
    try:
        s = str(misconduct_ranges)
        for date_range in s.split(","):
            date_range = date_range.strip()
            if not date_range:
                continue
            if "-" in date_range:
                start, end = date_range.split("-")
                if len(start) == 8 and len(end) == 8:
                    dates.append(f"{start[:4]}-{start[4:6]}-{start[6:8]}")
                    dates.append(f"{end[:4]}-{end[4:6]}-{end[6:8]}")
            elif len(date_range) == 8:
                dates.append(f"{date_range[:4]}-{date_range[4:6]}-{date_range[6:8]}")
    except Exception as e:
        logger.warning(f"Error parsing Misconduct_date_ranges: {e}, value: {misconduct_ranges}")
    return dates


def get_ground_truth_dates(row: pd.Series) -> list[str]:
    """Extract every ground-truth date for a row: Start, End, and any Misconduct_date_ranges."""
    dates: list[str] = []
    if (
        not pd.isna(row["Start_year"])
        and not pd.isna(row["Start_month"])
        and not pd.isna(row["Start_day"])
        and row["Start_year"] != ""
        and row["Start_month"] != ""
        and row["Start_day"] != ""
    ):
        d = format_date_from_components(
            row["Start_year"], row["Start_month"], row["Start_day"], "Start_year/month/day"
        )
        if d:
            dates.append(d)
    if (
        not pd.isna(row["End_year"])
        and not pd.isna(row["End_month"])
        and not pd.isna(row["End_day"])
        and row["End_year"] != ""
        and row["End_month"] != ""
        and row["End_day"] != ""
    ):
        d = format_date_from_components(
            row["End_year"], row["End_month"], row["End_day"], "End_year/month/day"
        )
        if d:
            dates.append(d)
    if not pd.isna(row["Misconduct_date_ranges"]) and row["Misconduct_date_ranges"] != "":
        dates.extend(parse_misconduct_date_ranges(row["Misconduct_date_ranges"]))
    return dates
