# Directory structure
SRC_DIR := src
DATA_DIR := data
INPUT_DIR := $(DATA_DIR)/input
OUTPUT_DIR := $(DATA_DIR)/output
POS_DIR := $(INPUT_DIR)/pos-train
NEG_DIR := $(INPUT_DIR)/neg-train
# Python settings
PYTHON := python3

# Script settings (can be overridden via command line)
IMAGE_WIDTH := 224
IMAGE_HEIGHT := 224
VERBOSITY := INFO

# Script arguments
SCRIPT_ARGS := --pos-dir $(POS_DIR) \
               --neg-dir $(NEG_DIR) \
               --output-dir $(OUTPUT_DIR) \
               --image-size $(IMAGE_WIDTH) $(IMAGE_HEIGHT) \
               --verbosity $(VERBOSITY)

# Default target
.PHONY: all
all: init process

# Create project structure
.PHONY: init
init:
	@echo "Creating project structure..."
	@mkdir -p $(SRC_DIR)
	@mkdir -p $(POS_DIR)
	@mkdir -p $(NEG_DIR)
	@mkdir -p $(OUTPUT_DIR)
	@touch $(SRC_DIR)/__init__.py
	@if [ ! -f $(SRC_DIR)/label.py ]; then \
		mv label.py $(SRC_DIR)/label.py 2>/dev/null || true; \
	fi
	@echo "Project structure created."
	@echo "Place positive PDFs in: $(POS_DIR)"
	@echo "Place negative PDFs in: $(NEG_DIR)"

# Process PDFs and generate labeled dataset
.PHONY: process
process:
	@echo "Processing PDFs..."
	@$(PYTHON) $(SRC_DIR)/label.py $(SCRIPT_ARGS)
	@echo "Processing complete. Check $(OUTPUT_DIR)/labeled_df.csv for results."

# Clean output files but preserve input PDFs
.PHONY: clean
clean:
	@echo "Cleaning output files..."
	@rm -rf $(OUTPUT_DIR)/*
	@mkdir -p $(OUTPUT_DIR)
	@echo "Clean complete."

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make init     - Create project structure"
	@echo "  make process  - Process PDFs and generate labeled dataset"
	@echo "  make clean    - Remove output files"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  IMAGE_WIDTH  - Width of output thumbnails (default: $(IMAGE_WIDTH))"
	@echo "  IMAGE_HEIGHT - Height of output thumbnails (default: $(IMAGE_HEIGHT))"
	@echo "  VERBOSITY   - Logging level (DEBUG|INFO|WARNING|ERROR) (default: $(VERBOSITY))"
	@echo ""
	@echo "Directory structure:"
	@echo "  $(POS_DIR) - Place positive PDF examples here"
	@echo "  $(NEG_DIR) - Place negative PDF examples here"
	@echo "  $(OUTPUT_DIR) - Generated files will be placed here"