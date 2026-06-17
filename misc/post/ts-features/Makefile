INPUT_FILE := ../ts-blocking/data/output/labeled_data_w_blocks.csv
OUTPUT_FILE := data/output/features.csv

.PHONY: all clean

all:
	python3 src/src.py \
		--input $(INPUT_FILE) \
		--output $(OUTPUT_FILE)

clean:
	rm -f $(OUTPUT_FILE)