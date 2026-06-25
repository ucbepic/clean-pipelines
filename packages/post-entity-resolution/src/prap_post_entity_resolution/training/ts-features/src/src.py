import argparse
import os
import pickle

import jellyfish
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Generate features for entity resolution")
    parser.add_argument(
        "--input",
    )
    parser.add_argument(
        "--output",
    )
    return parser.parse_args()


def calculate_string_similarity(str1, str2):
    """Calculate various string similarity metrics."""
    if pd.isna(str1) or pd.isna(str2):
        return {
            "jaro_winkler": 0,
            "levenshtein_norm": 1,  # Normalized to [0,1], 1 means maximum distance
            "length_ratio": 0,
        }

    str1 = str(str1).lower()
    str2 = str(str2).lower()

    max_len = max(len(str1), len(str2))
    min_len = min(len(str1), len(str2))

    if max_len == 0:
        return {"jaro_winkler": 0, "levenshtein_norm": 1, "length_ratio": 1}

    return {
        "jaro_winkler": jellyfish.jaro_winkler_similarity(str1, str2),
        "levenshtein_norm": 1 - (jellyfish.levenshtein_distance(str1, str2) / max_len),
        "length_ratio": min_len / max_len if max_len > 0 else 1,
    }


def calculate_embedding_similarity(text1, text2, model):
    """Calculate cosine similarity between embeddings of two texts."""
    if pd.isna(text1) or pd.isna(text2):
        return 0

    text1 = str(text1).lower()
    text2 = str(text2).lower()

    embedding1 = model.encode([text1], convert_to_tensor=True)
    embedding2 = model.encode([text2], convert_to_tensor=True)

    embedding1_np = embedding1.cpu().numpy()
    embedding2_np = embedding2.cpu().numpy()

    similarity = cosine_similarity(embedding1_np, embedding2_np)[0][0]

    return similarity


def engineer_name_features(df):
    """Generate both string-based and embedding-based features for names."""
    print("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    features = {}

    # Define column mappings
    name_columns = {
        "first": ("incident_first_name", "post_first_name"),
        "middle": ("incident_middle_name", "post_middle_name"),
        "last": ("incident_last_name", "post_last_name"),
        "suffix": ("incident_suffix", "post_suffix"),  # Changed from suffix_name to suffix
    }

    for name_part, (incident_col, post_col) in name_columns.items():
        print(f"Processing {name_part} names...")

        # String-based similarities
        string_similarities = df.apply(
            lambda x: calculate_string_similarity(x[incident_col], x[post_col]), axis=1
        )

        features.update(
            {
                f"{name_part}_name_jaro": [x["jaro_winkler"] for x in string_similarities],
                f"{name_part}_name_levenshtein": [
                    x["levenshtein_norm"] for x in string_similarities
                ],
                f"{name_part}_name_length_ratio": [x["length_ratio"] for x in string_similarities],
            }
        )

        # Embedding-based similarity
        features[f"{name_part}_name_embedding"] = df.apply(
            lambda x: calculate_embedding_similarity(x[incident_col], x[post_col], model), axis=1
        )

    # Full name processing
    print("Processing full names...")
    df["incident_full_name"] = df.apply(
        lambda x: " ".join(
            filter(
                None,
                [
                    str(x["incident_first_name"]) if not pd.isna(x["incident_first_name"]) else "",
                    str(x["incident_middle_name"])
                    if not pd.isna(x["incident_middle_name"])
                    else "",
                    str(x["incident_last_name"]) if not pd.isna(x["incident_last_name"]) else "",
                    str(x["incident_suffix"]) if not pd.isna(x["incident_suffix"]) else "",
                ],
            )
        ),
        axis=1,
    )

    df["post_full_name"] = df.apply(
        lambda x: " ".join(
            filter(
                None,
                [
                    str(x["post_first_name"]) if not pd.isna(x["post_first_name"]) else "",
                    str(x["post_middle_name"]) if not pd.isna(x["post_middle_name"]) else "",
                    str(x["post_last_name"]) if not pd.isna(x["post_last_name"]) else "",
                    str(x["post_suffix"]) if not pd.isna(x["post_suffix"]) else "",
                ],
            )
        ),
        axis=1,
    )

    # Calculate full name similarities
    full_name_strings = df.apply(
        lambda x: calculate_string_similarity(x["incident_full_name"], x["post_full_name"]), axis=1
    )

    features.update(
        {
            "full_name_jaro": [x["jaro_winkler"] for x in full_name_strings],
            "full_name_levenshtein": [x["levenshtein_norm"] for x in full_name_strings],
            "full_name_length_ratio": [x["length_ratio"] for x in full_name_strings],
            "full_name_embedding": df.apply(
                lambda x: calculate_embedding_similarity(
                    x["incident_full_name"], x["post_full_name"], model
                ),
                axis=1,
            ),
        }
    )

    # Add computed features to dataframe
    for feature_name, feature_values in features.items():
        df[feature_name] = feature_values

    return df


def normalize_features(df):
    """Normalize numerical features to [0,1] range."""
    scaler = MinMaxScaler()

    features_to_scale = [
        col
        for col in df.columns
        if any(x in col for x in ["jaro", "levenshtein", "length_ratio", "embedding"])
    ]

    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df, scaler


def main():
    args = parse_args()

    print(f"Reading data from {args.input}")
    df = pd.read_csv(args.input)

    print("Generating features...")
    df = engineer_name_features(df)

    print("Normalizing features...")
    df, scaler = normalize_features(df)  # Get the scaler

    # Save the scaler
    scaler_path = args.output.replace(".csv", "_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Saving features to {args.output}")
    df.to_csv(args.output, index=False)
    print("Feature generation complete!")


if __name__ == "__main__":
    main()
