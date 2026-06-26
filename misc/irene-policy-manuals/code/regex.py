import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# === PATHS ===
BASE_PATH = Path(__file__).resolve().parent.parent / "processed_data"
INPUT_CSV = BASE_PATH / "sample_processed.csv"
OUTPUT_CSV = BASE_PATH / "document_predictions_with_confidence.csv"


# === Extract distinctive patterns from known policy manuals ===
# This function extracts keywords, headers, and section patterns from the first few pages
# of known policy manuals. It uses TF-IDF to identify the most common terms and regex
# patterns to identify headers and sections.
# The function returns sets of keywords, headers, and section patterns for later use.
def extract_policy_features(df, max_pages=3, top_n=30):
    manual_pages = df[df["label"] == "policy_manual"]
    first_pages = (
        manual_pages.sort_values("page_num").groupby("filename", group_keys=False).head(max_pages)
    )

    texts = first_pages["text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    vectorizer.fit_transform(texts)
    keywords = set(vectorizer.get_feature_names_out())

    header_counter = Counter()
    section_counter = Counter()
    for text in texts:
        for line in text.splitlines():
            line = line.strip()
            # Detect headers: ALL CAPS, allows digits, 5–51 characters
            if re.match(r"^[A-Z0-9][A-Z0-9\s]{4,50}$", line):
                header_counter[line.lower()] += 1
            # Detect section headers like 1.1, 2.3.5
            section_match = re.match(r"^(\d+(?:\.\d+){1,})", line)
            if section_match:
                section_counter[section_match.group(1)] += 1

    headers = set([h for h, _ in header_counter.most_common(top_n)])
    section_patterns = set(section_counter.keys())
    return keywords, headers, section_patterns


# === Scan page text for matches to features ===
def scan_page(text, keyword_set, header_set, section_set):
    flags = defaultdict(int)
    if not isinstance(text, str):
        return flags

    text_lower = text.lower()
    for kw in keyword_set:
        if re.search(rf"\b{re.escape(kw)}\b", text_lower):
            flags["keyword"] += 1

    for line in text.splitlines():
        line = line.strip()
        if any(h in line.lower() for h in header_set):
            flags["header"] += 1
        if re.match(r"^\d+(?:\.\d+){1,}\b", line) or any(pat in line for pat in section_set):
            flags["section"] += 1
    return flags


# === Build one row per file from aggregated page features ===
def build_feature_matrix(df, keyword_set, header_set, section_set):
    features = []
    for filename, group in df.groupby("filename"):
        early_flags = defaultdict(int)
        late_flags = defaultdict(int)

        for _, row in group.iterrows():
            page_num = row["page_num"]
            flags = scan_page(row.get("text", ""), keyword_set, header_set, section_set)
            target = early_flags if page_num <= 2 else late_flags
            for k, v in flags.items():
                target[k] += v

        features.append(
            {
                "filename": filename,
                "early_keyword": early_flags["keyword"],
                "late_keyword": late_flags["keyword"],
                "early_header": early_flags["header"],
                "late_header": late_flags["header"],
                "early_section": early_flags["section"],
                "late_section": late_flags["section"],
                "page_count": group["page_num"].nunique(),
                "text_length": group["text"].astype(str).str.len().sum(),
                "true_label": group["label"].iloc[0],
            }
        )
    return pd.DataFrame(features)


# === Main pipeline ===
def main():
    print(f"Reading input from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    print("Extracting distinctive features from known policy_manuals...")
    keywords, headers, section_patterns = extract_policy_features(df)

    print("Building document-level feature matrix...")
    feature_df = build_feature_matrix(df, keywords, headers, section_patterns)

    # Save filenames separately before dropping
    filenames = feature_df["filename"].values

    # Prepare features and labels
    X = feature_df.drop(columns=["filename", "true_label"]).copy()
    y = feature_df["true_label"]

    # Train/test split
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, stratify=y, random_state=42
    )

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and compute confidence
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    confidence = y_proba.max(axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Save predictions + confidence
    results_df = pd.DataFrame(
        {
            "filename": filenames_test,
            "true_label": y_test.values,
            "predicted_label": y_pred,
            "confidence_score": confidence,
        }
    )

    # Add match column
    results_df["correct_prediction"] = results_df["true_label"] == results_df["predicted_label"]

    # Save to CSV
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions with confidence to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
