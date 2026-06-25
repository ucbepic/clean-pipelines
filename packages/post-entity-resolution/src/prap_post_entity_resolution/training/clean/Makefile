# Directory configurations
INPUT_DIR := data/input
OUTPUT_DIR := data/output

# Input files
CSV_INPUT := $(INPUT_DIR)/california-processed.csv
JSON_INPUT := $(INPUT_DIR)/fresno-county-extracted-officer-data.json

# Output files
POST_OUTPUT := $(OUTPUT_DIR)/df_post.csv
CPDP_OUTPUT := $(OUTPUT_DIR)/df_cpdp.csv

# Python script
SCRIPT := src/src.py

# Default target
all: $(POST_OUTPUT) $(CPDP_OUTPUT)

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Rule to process both CSV and JSON data
$(POST_OUTPUT) $(CPDP_OUTPUT): $(CSV_INPUT) $(JSON_INPUT) $(SCRIPT) | $(OUTPUT_DIR)
	python3 $(SCRIPT) \
		--csv-input $(CSV_INPUT) \
		--json-input $(JSON_INPUT) \
		--output-dir $(OUTPUT_DIR)

# Clean target to remove generated files
clean:
	rm -rf $(OUTPUT_DIR)

# Phony targets
.PHONY: all clean

# Print help information
help:
	@echo "Available targets:"
	@echo "  all    - Process both CSV and JSON data (default)"
	@echo "  clean  - Remove all generated files"
	@echo "  help   - Show this help message"

.PHONY: help