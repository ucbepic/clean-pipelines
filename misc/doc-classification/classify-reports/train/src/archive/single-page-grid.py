import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
import argparse
import logging
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
from datetime import datetime

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentDataset(Dataset):
    """Dataset for single-page document classification"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(row['label'], dtype=torch.long)
        
        try:
            img_path = f"../label/{row['img_filepath']}"
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = Image.merge('RGB', (image, image, image))
                image = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            image = torch.zeros((3, 224, 224))
        
        return image, label

def get_data_loaders(train_df, valid_df, batch_size=32):
    """Create data loaders with augmentation"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = DocumentDataset(train_df, transform=train_transform)
    valid_dataset = DocumentDataset(valid_df, transform=valid_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, valid_loader

class ModelFactory:
    """Factory class for creating different ResNet models"""
    @staticmethod
    def get_model(model_name, num_classes=2):
        model_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
            
        base_model = model_map[model_name](pretrained=True)
        
        # Modify first conv layer for grayscale
        new_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight = nn.Parameter(base_model.conv1.weight.mean(dim=1, keepdim=True).repeat(1,3,1,1))
        base_model.conv1 = new_conv1
        
        # Get feature dimension
        if model_name in ['resnet18', 'resnet34']:
            feature_dim = 512
        else:
            feature_dim = 2048
            
        # Remove last layer and add custom classifier
        modules = list(base_model.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        
        model = nn.Sequential(
            feature_extractor,
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        return model

class ExperimentManager:
    def __init__(self, train_df, valid_df, test_df, device, base_output_dir, args):
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.device = device
        self.base_output_dir = base_output_dir
        self.args = args
        self.results = []
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(base_output_dir, f'experiment_{self.timestamp}')
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.experiment_dir, 'experiment.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Track best model based on test performance
        self.best_model_info = {
            'test_f1': float('-inf'),
            'params': None,
            'model_path': None
        }

    def get_parameter_grid(self):
        return {
            'model_name': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [self.args.batch_size] if self.args.batch_size else [16, 32, 64],
            'optimizer': ['adam', 'sgd'],
            'scheduler': ['step', 'cosine', None],
            'epochs': [self.args.epochs] if self.args.epochs else [5, 10, 15]
        }

    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()

        return running_loss / len(train_loader), 100. * total_correct / total_samples

    def evaluate_model(self, model, data_loader, phase='validation'):
        """Evaluate model on validation or test set"""
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = {
            'accuracy': 100 * accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='weighted'),
            'recall': recall_score(all_labels, all_predictions, average='weighted'),
            'f1': f1_score(all_labels, all_predictions, average='weighted')
        }
        
        return metrics, all_predictions, all_probabilities


    def evaluate_on_test(self, model, exp_dir):
        """Evaluate model on the holdout test set"""
        # Verify we're using the correct test set
        logger.info(f"Evaluating on test set with {len(self.test_df)} examples")
        
        # Create test data loader
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        test_dataset = DocumentDataset(self.test_df, transform=test_transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=min(32, len(self.test_df)),  # Ensure batch size doesn't exceed dataset size
            shuffle=False, 
            num_workers=4
        )
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                # Add logging to verify batches
                logger.debug(f"Processing test batch {batch_idx + 1}, batch size: {len(labels)}")
                
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
        # Verify we got the expected number of predictions
        assert len(all_predictions) == len(self.test_df), \
            f"Got {len(all_predictions)} predictions but expected {len(self.test_df)}"
    
        # Calculate metrics
        metrics = {
            'accuracy': 100 * accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='weighted'),
            'recall': recall_score(all_labels, all_predictions, average='weighted'),
            'f1': f1_score(all_labels, all_predictions, average='weighted')
        }
    
        # Create results DataFrame using the test_df
        results_df = pd.DataFrame({
            'img_filepath': self.test_df['img_filepath'].values,
            'true_label': all_labels,
            'predicted_label': all_predictions,
            'probability_class_0': [prob[0] for prob in all_probabilities],
            'probability_class_1': [prob[1] for prob in all_probabilities]
        })
        
        if 'document_id' in self.test_df.columns:
            # Merge with original test_df to ensure correct document_ids
            results_df = pd.merge(
                results_df,
                self.test_df[['img_filepath', 'document_id']],
                on='img_filepath',
                how='inner'  # Use inner join to ensure we only have test set examples
            )
        
        # Verify the results DataFrame has the correct number of rows
        assert len(results_df) == len(self.test_df), \
            f"Results DataFrame has {len(results_df)} rows but expected {len(self.test_df)}"
        
        # Save test predictions for this experiment
        test_results_path = os.path.join(exp_dir, 'test_predictions.csv')
        results_df.to_csv(test_results_path, index=False)
        
        # Log test results
        logger.info("\nTest Set Results:")
        logger.info(f"Number of test examples: {len(self.test_df)}")
        logger.info(f"Accuracy: {metrics['accuracy']:.2f}%")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        
        logger.info("\nPrediction Statistics:")
        logger.info(f"Total predictions made: {len(results_df)}")
        logger.info(f"Predictions by class:")
        for label in sorted(results_df['predicted_label'].unique()):
            count = (results_df['predicted_label'] == label).sum()
            percentage = count/len(results_df)*100
            logger.info(f"Class {label}: {count} predictions ({percentage:.1f}%)")
        
        return metrics, test_results_path

    def run_single_experiment(self, params, exp_dir):
        # Create data loaders
        train_loader, valid_loader = get_data_loaders(
            self.train_df, 
            self.valid_df, 
            batch_size=params['batch_size']
        )
        
        # Create model and training components
        model = ModelFactory.get_model(params['model_name']).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = (optim.Adam(model.parameters(), lr=params['learning_rate']) 
                   if params['optimizer'] == 'adam' 
                   else optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9))
        
        scheduler = None
        if params['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        elif params['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        # Training loop with validation
        for epoch in range(params['epochs']):
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            if scheduler:
                scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{params['epochs']}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Save trained model
        model_path = os.path.join(exp_dir, 'model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'params': params
        }, model_path)
        
        # Evaluate on test set
        test_metrics, test_predictions_path = self.evaluate_on_test(model, exp_dir)
        
        return test_metrics, model_path, test_predictions_path
    
    def run_experiments(self):
        param_grid = self.get_parameter_grid()
        experiments = [dict(zip(param_grid.keys(), v)) 
                      for v in itertools.product(*param_grid.values())]
        
        logger.info(f"Starting {len(experiments)} experiments")
        
        for i, params in enumerate(experiments, 1):
            logger.info(f"\nExperiment {i}/{len(experiments)}")
            logger.info(f"Parameters: {params}")
            
            exp_dir = os.path.join(self.experiment_dir, f'exp_{i}')
            os.makedirs(exp_dir, exist_ok=True)
            
            try:
                test_metrics, model_path, test_predictions_path = self.run_single_experiment(params, exp_dir)
                
                # Update best model based on test F1 score
                if test_metrics['f1'] > self.best_model_info['test_f1']:
                    self.best_model_info = {
                        'test_f1': test_metrics['f1'],
                        'params': params,
                        'model_path': model_path,
                        'test_predictions_path': test_predictions_path
                    }
                
                # Store complete results including test metrics
                self.results.append({
                    **params,
                    **test_metrics
                })
                
                # Save updated results after each experiment
                results_df = pd.DataFrame(self.results)
                results_df.to_csv(os.path.join(self.experiment_dir, 'all_experiments_results.csv'), index=False)
                
            except Exception as e:
                logger.error(f"Error in experiment {i}: {str(e)}")
                continue
        
        # Log best model details at the end
        logger.info("\nBest Model (based on test F1 score):")
        logger.info(f"Parameters: {self.best_model_info['params']}")
        logger.info(f"Test F1: {self.best_model_info['test_f1']:.4f}")
        logger.info(f"Model path: {self.best_model_info['model_path']}")
        logger.info(f"Test predictions: {self.best_model_info['test_predictions_path']}")

    def save_results(self):
        # Save all experiments results
        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(self.experiment_dir, 'experiments_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # Save best model summary
        with open(os.path.join(self.experiment_dir, 'best_model_summary.txt'), 'w') as f:
            f.write("Best Model Details (based on test set performance):\n\n")
            f.write("Parameters:\n")
            for k, v in self.best_model_info['params'].items():
                f.write(f"{k}: {v}\n")
            f.write("\nTest Metrics:\n")
            if len(self.results) > 0:
                best_result = max(self.results, key=lambda x: x['f1'])
                for k, v in best_result.items():
                    if k not in self.best_model_info['params']:
                        f.write(f"{k}: {v:.4f}\n")

def main(args):
    # Load data
    train_df = pd.read_csv(args.data_csv)
    test_df = pd.read_csv(args.test_csv)
    
    # Split train into train/valid
    doc_ids = train_df['document_id'].unique()
    doc_labels = train_df.groupby('document_id')['label'].first()
    
    train_ids, valid_ids = train_test_split(
        doc_ids,
        train_size=args.train_split,
        random_state=args.seed,
        stratify=doc_labels
    )
    
    train_df_split = train_df[train_df['document_id'].isin(train_ids)]
    valid_df = train_df[train_df['document_id'].isin(valid_ids)]

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    experiment_manager = ExperimentManager(
        train_df=train_df_split,
        valid_df=valid_df,
        test_df=test_df,
        device=device,
        base_output_dir=args.output_dir,
        args=args
    )
    
    experiment_manager.run_experiments()
    
    logger.info("All experiments completed. Check experiments_results.csv for metrics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document classification model training')
    parser.add_argument('--data-csv', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--test-csv', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--output-dir', type=str, default='experiments', help='Base output directory')
    parser.add_argument('--batch-size', type=int, help='Override batch size for all experiments')
    parser.add_argument('--epochs', type=int, help='Override number of epochs for all experiments')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    main(args)