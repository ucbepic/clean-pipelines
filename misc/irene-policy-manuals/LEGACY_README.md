## PDF Text Extraction and Document Classification

This project provides tools to extract and classify content from law enforcement-related PDF documents. The primary goal is to determine which documents are policy manuals, which are case files that contain embedded policy manuals, and which are case files without policy content.

The project includes two main scripts:

**label.py:** Extracts text from PDFs using both direct parsing and OCR fallback. It outputs structured, per-page data for further analysis and training.

**regex.py:** Attempts to classify entire documents using a combination of heuristic features and a Random Forest classifier. This approach is documented but has proven insufficient for accurate classification and should be revisited in future work.

### Project Goal
The core task is to classify each PDF as one of the following:

**- policy_manual:** A standalone policy manual

**- embedded_policy_manuals:** A case file that contains embedded policy language

**- no_policy_manuals:** A case file with no policy content

The long-term aim is to develop a reliable model or pipeline that can automatically detect policy manuals, even when they appear in unstructured or embedded formats.

### 1. label.py: PDF Text Extraction

This script processes PDFs stored in labeled folders. It attempts to extract text directly from each page and falls back on OCR if the text is missing or too short.

**Input Folder Structure**

files_sample/
├── policy_manuals/
│   └── *.pdf
├── cases_with_embedded_policy_manuals/
│   └── *.pdf
├── cases_without_policy_manuals/
│   └── *.pdf

**Output**

**File:** processed_data/sample_processed.csv

**Columns:**

**- filename:** PDF file name
**- page_num:** Page number (0-indexed)
**- label:** Assigned class (from input folder)
**- text:** Extracted text
**- extraction_method:** direct (parsed) or OCR (image-to-text)

**Run the Script**
python **label.py** - This will process all PDFs and output a page-level dataset to processed_data/sample_processed.csv.

### 2. regex.py: Heuristic and ML-Based Classification

This script attempts to classify entire documents using features such as:

- Frequent keywords extracted via TF-IDF
- Capitalized headers
- Structured section numbers (e.g., 1.2.3)
- Distribution of these features across early vs. late pages
- It builds a feature matrix and trains a Random Forest model to classify documents

**Limitations**
We know this approach is not strong enough for reliable classification. Current accuracy is around 50%, which is insufficient for production use. This script is included to document what has been tried.

**Future efforts should explore better methods, such as:**

- Large Language Models (LLMs) for zero-shot classification
- Full-document embeddings and similarity analysis
- Fine-tuned transformers or hybrid rule-based + ML systems

**Output**

**File:** processed_data/document_predictions_with_confidence.csv

**Columns:**

**-filename:** The name of the original PDF file (e.g., example_policy.pdf)
**-true_label:** The actual class assigned based on the file's folder:  
    - policy_manual
    - embedded_policy_manuals
    - no_policy_manuals
**-predicted_label:** The label predicted by the classifier
**-confidence_score:** A float between 0 and 1 representing the model’s confidence in its prediction
**-correct_prediction:** True if the predicted label matches the true label, otherwise False

**Run the Script**
python regex.py

**Dependencies**
Install required Python packages:

pip install pandas pytesseract Pillow pymupdf scikit-learn

Also make sure you have Tesseract OCR installed.

### Project Structure

irene-policy-manuals/
├── code/
│   ├── label.py                            # PDF text extraction script
│   └── regex.py                            # Document-level classifier 
├── files_sample/                           # Input PDFs organized by class
│   ├── policy_manuals/
│   ├── cases_with_embedded_policy_manuals/
│   └── cases_without_policy_manuals/
├── processed_data/
│   ├── sample_processed.csv                # Output from label.py
│   └── document_predictions_with_confidence.csv  # Output from regex.py

### Future Directions
Some ideas for the next phase of work:

- Use an LLM to classify documents based on a document-level prompt (e.g., summarization + classification)
- Refine training data and labels for better model supervision

### A Note to Future Developers
This repo captures early efforts to build a document classification pipeline using heuristic and ML-based approaches. While the current models need improvement, the structure is in place to support further experimentation and refinement.