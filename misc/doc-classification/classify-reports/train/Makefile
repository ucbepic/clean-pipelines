# Directory structure
SRC_DIR := src
DATA_CSV := ../label/data/output/labeled_df_train.csv
TEST_CSV := ../label/data/output/labeled_df_test.csv
MODEL_DIR := models
LOG_DIR := logs

# Python settings
PYTHON := python3

# Training settings (can be overridden via command line)
BATCH_SIZE := 32
NUM_EPOCHS := 10
LEARNING_RATE := 0.001
TRAIN_SPLIT := 0.8
SEED := 42
DEVICE := cuda

# Ensure model directory is included in arguments
TRAIN_ARGS := --data-csv $(DATA_CSV) \
              --batch-size $(BATCH_SIZE) \
              --epochs $(NUM_EPOCHS) \
              --train-split $(TRAIN_SPLIT) \
              --seed $(SEED) \
              --device $(DEVICE) \
              --model-dir $(MODEL_DIR) \
			  --test-csv $(TEST_CSV) 

# Default target
.PHONY: all
all: init train

# Create project structure
.PHONY: init
init:
	@echo "Creating project structure..."
	@mkdir -p $(SRC_DIR)
	@mkdir -p $(MODEL_DIR)
	@mkdir -p $(LOG_DIR)
	@touch $(SRC_DIR)/__init__.py
	@echo "Project structure created:"
	@echo "  Models will be saved to: $(MODEL_DIR)"
	@echo "  Logs will be saved to: $(LOG_DIR)"

# Train the model
.PHONY: train
train: init
	@echo "Training CNN model..."
	@mkdir -p $(MODEL_DIR)
	@PYTHONPATH=. $(PYTHON) $(SRC_DIR)/src.py $(TRAIN_ARGS) 2>&1 | tee $(LOG_DIR)/training_$(shell date +%Y%m%d_%H%M%S).log
	@echo "Training complete. Check $(LOG_DIR) for training log and $(MODEL_DIR) for saved models."

# Validate latest model
.PHONY: validate
validate:
	@echo "Validating model..."
	@PYTHONPATH=. $(PYTHON) $(SRC_DIR)/src.py \
		--data-csv $(DATA_CSV) \
		--model-path $(MODEL_DIR)/best_model.pth \
		--mode validate \
		--device $(DEVICE) 2>&1 | tee $(LOG_DIR)/validation_$(shell date +%Y%m%d_%H%M%S).log

# Clean generated files but keep directories
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	@rm -f $(LOG_DIR)/*
	@rm -f $(MODEL_DIR)/*
	@echo "Clean complete."

# Remove all generated directories and files
.PHONY: clean-all
clean-all:
	@echo "Removing all generated directories and files..."
	@rm -rf $(MODEL_DIR)
	@rm -rf $(LOG_DIR)
	@echo "Clean complete."

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make init      - Create project structure"
	@echo "  make train     - Train the CNN model"
	@echo "  make validate  - Run validation on latest model"
	@echo "  make clean     - Remove generated files but keep directories"
	@echo "  make clean-all - Remove all generated directories and files"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Training Configuration (can be overridden):"
	@echo "  BATCH_SIZE    - Batch size for training (default: $(BATCH_SIZE))"
	@echo "  NUM_EPOCHS    - Number of training epochs (default: $(NUM_EPOCHS))"
	@echo "  TRAIN_SPLIT   - Train/validation split ratio (default: $(TRAIN_SPLIT))"
	@echo "  SEED         - Random seed (default: $(SEED))"
	@echo "  DEVICE       - Training device: cuda/cpu (default: $(DEVICE))"
	@echo ""
	@echo "Directory Structure:"
	@echo "  $(SRC_DIR)     - Source code directory"
	@echo "  $(MODEL_DIR)   - Saved models directory"
	@echo "  $(LOG_DIR)     - Training and validation logs directory"
	@echo ""
	@echo "Example usage:"
	@echo "  make train BATCH_SIZE=64 DEVICE=cuda"
	@echo "  make validate DEVICE=cpu"i