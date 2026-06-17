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
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            # Convert single channel to RGB by repeating channel
            if self.transform:
                image = Image.merge('RGB', (image, image, image))
                image = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            image = torch.zeros((3, 224, 224))
        
        return image, label

def get_data_loaders(train_df, valid_df, batch_size=32):  # Increased batch size
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

class DocumentClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Use ResNet50 instead of ResNet18
        resnet = models.resnet152(pretrained=True)
        
        # Modify first conv layer
        new_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            new_conv1.weight = nn.Parameter(resnet.conv1.weight.mean(dim=1, keepdim=True).repeat(1,3,1,1))
        
        resnet.conv1 = new_conv1
        
        # Unfreeze all layers for better adaptation
        for param in resnet.parameters():
            param.requires_grad = True
                
        # Remove last layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Update classifier to handle ResNet50's larger feature dimension (2048 vs 512)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),  # First reduce dimensionality
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(valid_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return val_loss, val_acc, precision, recall, f1


def train_model(model, train_loader, valid_loader, epochs, device, model_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    best_val_loss = float('inf')
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
          
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        
        # Save if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'val_acc': best_val_loss
            }, best_model_path)
            logger.info(f"Saved new best model at epoch {epoch+1}")
    
    return best_model_path

def balance_dataset_by_pages(df):
    """
    Balance a dataset by downsampling the majority class at the page level
    while maintaining document integrity.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'filename' and 'label' columns
        
    Returns:
        pd.DataFrame: Balanced dataframe
    """
    # Get current class distribution at page level
    page_level_dist = df['label'].value_counts()
    
    # Identify majority and minority classes
    minority_class = page_level_dist.index[page_level_dist.argmin()]
    majority_class = page_level_dist.index[page_level_dist.argmax()]
    target_count = page_level_dist[minority_class]
    
    # Get documents containing majority class pages
    majority_docs = df[df['label'] == majority_class]['filename'].unique()
    np.random.shuffle(majority_docs)
    
    # Select documents until we reach target count
    selected_docs = []
    current_pages = 0
    
    for doc in majority_docs:
        doc_pages = len(df[(df['filename'] == doc) & (df['label'] == majority_class)])
        if current_pages + doc_pages <= target_count:
            selected_docs.append(doc)
            current_pages += doc_pages
    
    # Create balanced dataset
    majority_docs_mask = df['filename'].isin(selected_docs)
    minority_docs_mask = df['label'] == minority_class
    balanced_df = df[majority_docs_mask | minority_docs_mask].copy()
    
    return balanced_df
    
def main(args):
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load and split data
    df = pd.read_csv(args.data_csv)
    df = df.pipe(balance_dataset_by_pages)
    
    test_df = pd.read_csv(args.test_csv)
    
    # Log initial class distribution by document_id
    doc_level_dist = df.groupby('filename')['label'].first().value_counts()
    logger.info("\nInitial class distribution (document-level):")
    for label, count in doc_level_dist.items():
        logger.info(f"Class {label}: {count} documents ({count/len(doc_level_dist)*100:.1f}%)")

    # Log initial class distribution by pages
    page_level_dist = df['label'].value_counts()
    logger.info("\nInitial class distribution (page-level):")
    for label, count in page_level_dist.items():
        logger.info(f"Class {label}: {count} pages ({count/len(df)*100:.1f}%)")

    # Get unique document IDs for splitting
    doc_ids = df['filename'].unique()
    doc_labels = df.groupby('filename')['label'].first()
    
    # Split on document level with stratification
    train_ids, temp_ids = train_test_split(
        doc_ids,
        train_size=args.train_split,
        random_state=args.seed,
        stratify=doc_labels
    )
    valid_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=args.seed,
        stratify=doc_labels[temp_ids]
    )

    # Log the counts
    logger.info(f"Total document IDs: {len(doc_ids)}")
    logger.info(f"Training document IDs: {len(train_ids)}")
    logger.info(f"Validation document IDs: {len(valid_ids)}")
    logger.info(f"Test document IDs: {len(test_ids)}")
    
    # Create dataframes for each split
    train_df = df[df['filename'].isin(train_ids)]
    valid_df = df[df['filename'].isin(valid_ids)]
    
    # Log train split distributions
    train_doc_dist = train_df.groupby('filename')['label'].first().value_counts()
    logger.info("\nTraining set distribution (document-level):")
    for label, count in train_doc_dist.items():
        logger.info(f"Class {label}: {count} documents ({count/len(train_doc_dist)*100:.1f}%)")
    
    train_page_dist = train_df['label'].value_counts()
    logger.info("\nTraining set distribution (page-level):")
    for label, count in train_page_dist.items():
        logger.info(f"Class {label}: {count} pages ({count/len(train_df)*100:.1f}%)")

    # Log validation split distributions
    valid_doc_dist = valid_df.groupby('filename')['label'].first().value_counts()
    logger.info("\nValidation set distribution (document-level):")
    for label, count in valid_doc_dist.items():
        logger.info(f"Class {label}: {count} documents ({count/len(valid_doc_dist)*100:.1f}%)")
    
    valid_page_dist = valid_df['label'].value_counts()
    logger.info("\nValidation set distribution (page-level):")
    for label, count in valid_page_dist.items():
        logger.info(f"Class {label}: {count} pages ({count/len(valid_df)*100:.1f}%)")

    logger.info(f"Training pages: {len(train_df)}")
    logger.info(f"Validation pages: {len(valid_df)}")

    # Create data loaders
    train_loader, valid_loader = get_data_loaders(train_df, valid_df, args.batch_size)

    # Log batch distribution from train_loader
    batch_labels = []
    for _, labels in train_loader:
        batch_labels.extend(labels.numpy())
    batch_dist = pd.Series(batch_labels).value_counts()
    logger.info("\nBatch distribution:")
    for label, count in batch_dist.items():
        logger.info(f"Class {label}: {count} samples ({count/len(batch_labels)*100:.1f}%)")

    # Setup and train model
    model = DocumentClassifier(num_classes=2).to(device)
    best_model_path = train_model(
        model, 
        train_loader, 
        valid_loader, 
        args.epochs,
        device,
        args.model_dir
    )

    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'val_loss': float('inf'),
        'val_acc': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    # Load the best model for evaluation
    logger.info(f"Loading best model from {best_model_path} for evaluation...")
    best_model = DocumentClassifier(num_classes=2).to(device)
    checkpoint = torch.load(best_model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']} "
                f"with validation loss: {checkpoint['val_loss']:.4f} "
                f"and accuracy: {checkpoint['val_acc']:.4f}")
    
    # Log test set class distribution before evaluation
    test_doc_dist = test_df.groupby('filename')['label'].first().value_counts()
    logger.info("\nTest set distribution (document-level):")
    for label, count in test_doc_dist.items():
        logger.info(f"Class {label}: {count} documents ({count/len(test_doc_dist)*100:.1f}%)")
    
    test_page_dist = test_df['label'].value_counts()
    logger.info("\nTest set distribution (page-level):")
    for label, count in test_page_dist.items():
        logger.info(f"Class {label}: {count} pages ({count/len(test_df)*100:.1f}%)")
    
    logger.info(f"\nEvaluating on test set with {test_df.filename.nunique()} documents...")
    
    test_loader, _ = get_data_loaders(test_df, test_df, args.batch_size) 
    
    best_model.eval()
    predictions = []
    probabilities = []
    actuals = []
    image_paths = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = best_model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
            
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + len(inputs)
            image_paths.extend(test_df.iloc[start_idx:end_idx]['img_filepath'].tolist())

    results_df = pd.DataFrame({
        'img_filepath': image_paths,
        'true_label': actuals,
        'predicted_label': predictions,
        'probability_class_0': [prob[0] for prob in probabilities],
        'probability_class_1': [prob[1] for prob in probabilities]
    })

    if 'filename' in test_df.columns:
        results_df = pd.merge(
            results_df,
            test_df[['img_filepath', 'filename']],
            on='img_filepath',
            how='left'
        )

    output_path = "data/output/test_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved prediction results to {output_path}")

    accuracy = 100 * (results_df['true_label'] == results_df['predicted_label']).mean()
    precision = precision_score(results_df['true_label'], results_df['predicted_label'], average='weighted')
    recall = recall_score(results_df['true_label'], results_df['predicted_label'], average='weighted')
    f1 = f1_score(results_df['true_label'], results_df['predicted_label'], average='weighted')

    logger.info("\nTest Set Results (using best validation model):")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Log detailed prediction statistics
    logger.info("\nPrediction Statistics:")
    logger.info(f"Total predictions made: {len(results_df)}")
    logger.info(f"Predictions by class:")
    for label in sorted(results_df['predicted_label'].unique()):
        count = (results_df['predicted_label'] == label).sum()
        percentage = count/len(results_df)*100
        logger.info(f"Class {label}: {count} predictions ({percentage:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN for multi-page document classification')
    parser.add_argument('--data-csv', type=str, required=True, help='Path to labeled_df.csv')
    parser.add_argument('--test-csv', type=str, required=True, help='Path to test_df.csv')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--train-split', type=float, default=0.6, help='Proportion of data to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use for training')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
