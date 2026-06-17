import pandas as pd


def read_tbl():
    # dfa = pd.read_csv("../../data/output/autofolio_1.1.0_output--Fresno Police Department--2025-01-08_07-50-39 - autofolio_1.1.0_output--Fresno Police Department--2025-01-08_07-50-39-1.csv")
    # dfa = pd.read_csv("../../data/output/autofolio_1.2.0_output--Los Angeles Police Department--2025-03-16_22-28-30 - autofolio_1.2.0_output--Los Angeles Police Department--2025-03-16_22-28-30.csv")
    # dfa = pd.read_csv("../../data/output/autofolio_1.2.0_output--Los Angeles County Sheriff--2025-03-14_16-16-56 - autofolio_1.2.0_output--Los Angeles County Sheriff--2025-03-14_16-16-56.csv")
    # dfa = pd.read_csv("../../data/output/autofolio_1.1.0_output--San Bernardino County Sheriff--2025-01-08_00-09-41 - autofolio_1.1.0_output--San Bernardino County Sheriff--2025-01-08_00-09-41.csv")
    dfa = pd.read_csv("../../data/output/autofolio_1.1.0_output--Tulare County Sheriff--2024-12-21_01-12-52 - autofolio_1.1.0_output--Tulare County Sheriff--2024-12-21_01-12-52_with_extracted_features.csv")



    dfa = dfa.drop(columns=["ocr_text_per_page"])

    # dfb = pd.read_csv("../../data/input/autofolio_1.1.0_output--Fresno Police Department--2025-01-08_07-50-39 - autofolio_1.1.0_output--Fresno Police Department--2025-01-08_07-50-39-1.csv")
    # dfb = pd.read_csv("../../data/input/autofolio_1.2.0_output--Los Angeles Police Department--2025-03-16_22-28-30 - autofolio_1.2.0_output--Los Angeles Police Department--2025-03-16_22-28-30.csv")
    # dfb = pd.read_csv("../../data/input/autofolio_1.2.0_output--Los Angeles County Sheriff--2025-03-14_16-16-56 - autofolio_1.2.0_output--Los Angeles County Sheriff--2025-03-14_16-16-56.csv")
    # dfb = pd.read_csv("../../data/input/autofolio_1.1.0_output--San Bernardino County Sheriff--2025-01-08_00-09-41 - autofolio_1.1.0_output--San Bernardino County Sheriff--2025-01-08_00-09-41.csv")
    dfb = pd.read_csv("../../data/input/autofolio_1.1.0_output--Tulare County Sheriff--2024-12-21_01-12-52 - autofolio_1.1.0_output--Tulare County Sheriff--2024-12-21_01-12-52.csv")


    dfb = dfb[(dfb.ocr_text_per_page.fillna("") == "")]
    dfb = dfb.drop(columns=["ocr_text_per_page"])

    df = pd.concat([dfa, dfb])

    print(df.shape)
    print(df.head(10))
    return df


if __name__ == "__main__":
    df = read_tbl()
    df.to_csv("../../data/output/autofolio_1.1.0_output--Tulare County Sheriff--2024-12-21_01-12-52 - autofolio_1.1.0_output--Tulare County Sheriff--2024-12-21_01-12-52_ocr_col_dropped.csv", index=False)
