INPUT_FILE := ../train-data/data/output/labeled_data_10_1_2025.csv
OUTPUT_FILE := data/output/labeled_data_w_blocks.csv
PREFIX_LEN := 1
SUFFIX_LEN := 2

.PHONY: all clean

all:
	python3 src/src.py \
		--input $(INPUT_FILE) \
		--output $(OUTPUT_FILE) \
		--prefix-len $(PREFIX_LEN) \
		--suffix-len $(SUFFIX_LEN)

clean:
	rm -f $(OUTPUT_FILE)