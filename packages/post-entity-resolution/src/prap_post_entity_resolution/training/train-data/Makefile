INPUT_DIR := ../clean/data/output
OUTPUT_DIR := data/output

INCIDENT_DATA := $(INPUT_DIR)/df_cpdp.csv
POST_DATA := $(INPUT_DIR)/df_post.csv

MATCHES_OUTPUT := $(OUTPUT_DIR)/entity_matches.xlsx
TRAINING_OUTPUT := $(OUTPUT_DIR)/training_data.csv

SCRIPT := src/src.py


all: $(MATCHES_OUTPUT) $(TRAINING_OUTPUT)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(MATCHES_OUTPUT) $(TRAINING_OUTPUT): $(INCIDENT_DATA) $(POST_DATA) $(SCRIPT) | $(OUTPUT_DIR)
	python3 $(SCRIPT) \
		--incident-data $(INCIDENT_DATA) \
		--post-data $(POST_DATA) \
		--output $(MATCHES_OUTPUT) \
		--training-output $(TRAINING_OUTPUT)

# Clean target to remove generated files
clean:
	rm -rf $(OUTPUT_DIR)

# Clean intermediate files but keep final output
clean-interim:
	find $(OUTPUT_DIR) -type f ! -name 'entity_matches.xlsx' ! -name 'training_data.csv' -delete

.PHONY: all clean clean-interim help

