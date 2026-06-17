# Doc classification 

This system classifies documents into positive and negative classes using a two-step process:
1. Preprocessing/labeling of PDF documents
2. Training a CNN model on the labeled data

## Directory Structure

NOTE: We need to create a labeled_df_train.csv and labeled_df_test.csv, which can be used to evaluate the model on a separate test dataset. The preprocessing script needs to be run twice: once for train data and once for test data

```
project/
├── label/
│   ├── data/
│   │   ├── input/
│   │   │   ├── pos-train/  # Place positive class PDFs here for training
│   │   │   ├── neg-train/  # Place negative class PDFs here for training
│   │   │   ├── pos-test/   # Place positive class PDFs here for testing
│   │   │   └── neg-test/   # Place negative class PDFs here for testing
│   │   └── output/
│   │       ├── labeled_df_train.csv  # Generated labels and image paths for training
│   │       ├── labeled_df_test.csv   # Generated labels and image paths for testing
│   │       └── [unique_hashes]/      # Contains preprocessed page images
│   └── src/
│       └── src.py  # PDF preprocessing script
└── train/
    └── src/
        └── train.py  # CNN model training script
```

## Workflow

### 1. Document Preprocessing (Labeling)

The labeling process converts PDFs to preprocessed images and generates a CSV with labels.

#### Input
- Place positive class PDFs in `label/data/input/pos-train/`
- Place negative class PDFs in `label/data/input/neg-train/`

#### Run the Preprocessing
```
cd label
run the command make
```

#### Output
- `labeled_df_train.csv`: Contains metadata and labels for each page
- `labeled_df_test.csv`: Contains metadata and labels for each page
- Preprocessed page images stored in subdirectories named with hash prefixes (e.g., `ff086523/`)

### 2. Model Training

The training process uses the labeled data to train a CNN classifier.

#### Input
- `labeled_df_train.csv` from the previous step
- `labeled_df_test.csv` from the previous step
- Preprocessed images in their respective hash directories

#### Run the Training
cd train
run the command make
```

#### Output
- Best model saved to `models/best_model.pth`
- Final model saved to `models/final_model.pth`
- Training statistics printed to console
- Test results saved to `data/output/test_results.csv`
