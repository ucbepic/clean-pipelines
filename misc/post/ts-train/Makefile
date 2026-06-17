INPUT_FILE := ../ts-features/data/output/features.csv
OUTPUT_DIR := data/output
TEST_SIZE := 0.2

.PHONY: all clean train-xgb train-rf train-lr

all: train-xgb train-rf train-lr

train-xgb:
	python3 src/src.py \
		--input $(INPUT_FILE) \
		--output-dir $(OUTPUT_DIR) \
		--model-type xgboost \
		--test-size $(TEST_SIZE)

train-rf:
	python3 src/src.py \
		--input $(INPUT_FILE) \
		--output-dir $(OUTPUT_DIR) \
		--model-type random_forest \
		--test-size $(TEST_SIZE)

train-lr:
	python3 src/src.py \
		--input $(INPUT_FILE) \
		--output-dir $(OUTPUT_DIR) \
		--model-type logistic \
		--test-size $(TEST_SIZE)

clean:
	rm -rf $(OUTPUT_DIR)