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
    """Dataset for multi-page document classification"""
    def __init__(self, df, transform=None):
        self.doc_groups = df.groupby('document_id')
        self.doc_ids = list(self.doc_groups.groups.keys())
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        doc_id = self.doc_ids[idx]
        doc_pages = self.doc_groups.get_group(doc_id).sort_values('page_index')
        
        label = torch.tensor(doc_pages.iloc[0]['label'], dtype=torch.long)
        
        pages = []
        for _, row in doc_pages.iterrows():
            try:
                img_path = f"../label/{row['img_filepath']}"
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                
                if self.transform:
                    image = Image.merge('RGB', (image, image, image))
                    image = self.transform(image)
                pages.append(image)
            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {e}")
                pages.append(torch.zeros((3, 224, 224)))
        
        pages_tensor = torch.stack(pages)
        return pages_tensor, label


def get_data_loaders(train_df, valid_df, batch_size=32):
    # Simplified transforms for binary classification
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),  # Reduced rotation
        transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),  # Tighter crop
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
        collate_fn=collate_documents
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_documents
    )

    return train_loader, valid_loader

class DocumentClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Switch to ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # Modify first conv layer for grayscale input
        new_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize with average of pretrained weights
        with torch.no_grad():
            new_conv1.weight = nn.Parameter(resnet.conv1.weight.mean(dim=1, keepdim=True).repeat(1,3,1,1))
        
        resnet.conv1 = new_conv1
        
        # Freeze early layers
        for param in list(resnet.children())[:-3]:  # Freeze fewer layers for ResNet18
            for p in param.parameters():
                p.requires_grad = False
                
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Update feature_dim for ResNet18
        feature_dim = 512  # Changed from 2048 (ResNet50) to 512 (ResNet18)
        
        # Simplified attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout
            nn.Linear(128, 1)
        )
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduced dropout
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, num_pages, channels, height, width = x.size()
        x = x.view(batch_size * num_pages, channels, height, width)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_pages, -1)
        attention_weights = self.attention(features)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_features = torch.sum(features * attention_weights, dim=1)
        output = self.classifier(weighted_features)
        return output


def collate_documents(batch):
    """
    Custom collate function to handle documents with different numbers of pages
    Args:
        batch: List of tuples (pages_tensor, label)
    """
    # Get max number of pages in this batch
    max_pages = max(pages.size(0) for pages, _ in batch)
    
    # Get other dimensions from first item
    _, c, h, w = batch[0][0].size()
    
    # Initialize tensors for batched data
    batch_size = len(batch)
    padded_pages = torch.zeros(batch_size, max_pages, c, h, w)
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill in the tensors
    for i, (pages, label) in enumerate(batch):
        num_pages = pages.size(0)
        padded_pages[i, :num_pages] = pages
        labels[i] = label
    
    return padded_pages, labels


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for pages, labels in train_loader:
        pages, labels = pages.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(pages)
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
        for pages, labels in valid_loader:
            pages, labels = pages.to(device), labels.to(device)
            outputs = model(pages)
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
                'loss': best_val_loss
            }, best_model_path)
            logger.info(f"Saved new best model at epoch {epoch+1}")
    
    return best_model_path

def main(args):
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load and split data
    df = pd.read_csv(args.data_csv)

    logger.info(f"Loaded {df.document_id.nunique()} samples from {args.data_csv}")
    
    # Get unique document IDs for splitting
    doc_ids = df['document_id'].unique()
    
    # Split on document level
    train_ids, temp_ids = train_test_split(
        doc_ids,
        train_size=args.train_split,
        random_state=args.seed
    )
    valid_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=args.seed
    )

    # Log the counts
    logger.info(f"Total document IDs: {len(doc_ids)}")
    logger.info(f"Training document IDs: {len(train_ids)}")
    logger.info(f"Validation document IDs: {len(valid_ids)}")
    logger.info(f"Test document IDs: {len(test_ids)}")
    
    # Create dataframes for each split
    train_df = df[df['document_id'].isin(train_ids)]
    valid_df = df[df['document_id'].isin(valid_ids)]

    logger.info(f"Training pages: {len(train_df)}")
    logger.info(f"Validation pages: {len(valid_df)}")

    # Create data loaders
    train_loader, valid_loader = get_data_loaders(train_df, valid_df, args.batch_size)

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
    
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Load the best model for evaluation
    logger.info(f"Loading best model from {best_model_path} for evaluation...")
    best_model = DocumentClassifier(num_classes=2).to(device)
    checkpoint = torch.load(best_model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load and evaluate on test set
    test_df = pd.read_csv(args.test_csv)
    logger.info(f"\nEvaluating on test set with {test_df.document_id.nunique()} documents...")
    
    test_loader, _ = get_data_loaders(test_df, test_df, args.batch_size) 
    
    best_model.eval()
    correct = 0
    total = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(actuals, predictions, average='weighted')
    recall = recall_score(actuals, predictions, average='weighted')
    f1 = f1_score(actuals, predictions, average='weighted')

    logger.info("\nTest Set Results (using best validation model):")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN for multi-page document classification')
    parser.add_argument('--data-csv', type=str, required=True, help='Path to labeled_df.csv')
    parser.add_argument('--test-csv', type=str, required=True, help='Path to test_df.csv')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--train-split', type=float, default=0.6, help='Proportion of data to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
