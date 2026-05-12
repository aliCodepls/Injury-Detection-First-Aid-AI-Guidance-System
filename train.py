import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

# =========================================================
# CONFIGURATION
# =========================================================

DATA_ROOT = "dataset/train"  # Change to your path
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping
CLASS_MAPPING = {
    "Burns": 0,
    "Cuts_lacerations": 1,
    "Abrasions": 2,
    "Insect_bites": 3,
    "Bruises": 4
}

print(f"🎯 Device: {DEVICE}")
print(f"📁 Data root: {DATA_ROOT}")

# =========================================================
# MULTI-OUTPUT DATASET (Loads image + ALL targets from JSON)
# =========================================================

class MultiOutputDataset(Dataset):
    def __init__(self, root_dir, class_mapping, transform=None):
        self.root_dir = root_dir
        self.class_mapping = class_mapping
        self.transform = transform
        
        self.images = []
        self.labels = []          # injury type
        self.severities = []      # severity score (1-10)
        self.bleedings = []       # has_bleeding (0/1)
        self.swellings = []       # has_swelling (0/1)
        self.emergencies = []     # emergency_needed (0/1)
        
        # Scan folders
        for class_name, class_idx in class_mapping.items():
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ Warning: {class_path} not found")
                continue
            
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    json_path = os.path.join(class_path, Path(img_file).stem + '.json')
                    
                    if not os.path.exists(json_path):
                        continue
                    
                    # Load JSON to get targets
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                    self.severities.append(data.get('severity_score', 3))
                    self.bleedings.append(1.0 if data.get('has_bleeding', False) else 0.0)
                    self.swellings.append(1.0 if data.get('has_swelling', False) else 0.0)
                    self.emergencies.append(1.0 if data.get('emergency_needed', False) else 0.0)
        
        print(f"📸 Loaded {len(self.images)} images with all targets")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Return image + ALL targets
        return (
            image,
            self.labels[idx],           # injury type (0-4)
            self.severities[idx],       # severity (1-10)
            self.bleedings[idx],        # bleeding (0/1)
            self.swellings[idx],        # swelling (0/1)
            self.emergencies[idx]       # emergency (0/1)
        )

# =========================================================
# MULTI-OUTPUT MODEL (Single EfficientNet → 5 outputs)
# =========================================================

class MultiOutputEfficientNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Backbone: EfficientNetB0
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        backbone_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Shared features
        self.shared = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output heads
        # 1. Injury type classifier (5 classes)
        self.classifier_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # 2. Severity regressor (1-10)
        self.severity_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output 0-1, then scale to 1-10
        )
        
        # 3. Bleeding classifier (binary)
        self.bleeding_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 4. Swelling classifier (binary)
        self.swelling_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 5. Emergency classifier (binary)
        self.emergency_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Shared backbone
        features = self.backbone(x)
        shared = self.shared(features)
        
        # Different heads
        injury_type = self.classifier_head(shared)
        severity = self.severity_head(shared) * 9 + 1  # Scale 0-1 → 1-10
        bleeding = self.bleeding_head(shared)
        swelling = self.swelling_head(shared)
        emergency = self.emergency_head(shared)
        
        return injury_type, severity, bleeding, swelling, emergency

# =========================================================
# DATA AUGMENTATION
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================================================
# LOAD DATASET
# =========================================================

print("\n📂 Loading dataset...")
full_dataset = MultiOutputDataset(DATA_ROOT, CLASS_MAPPING, transform=train_transform)

# Split
train_idx, temp_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.3,
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=42
)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

# Fix transforms
full_dataset.transform = val_transform
val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform
full_dataset.transform = train_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Train: {len(train_dataset)} images")
print(f"✅ Validation: {len(val_dataset)} images")
print(f"✅ Test: {len(test_dataset)} images")

# =========================================================
# TRAINING SETUP
# =========================================================

model = MultiOutputEfficientNet(num_classes=5).to(DEVICE)

# Different loss functions for different outputs
criterion_class = nn.CrossEntropyLoss()
criterion_severity = nn.L1Loss()  # Mean Absolute Error
criterion_binary = nn.BCELoss()   # Binary classification

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

print(f"\n🧠 Model: Multi-Output EfficientNetB0")
print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# =========================================================
# TRAINING LOOP
# =========================================================

best_val_loss = float('inf')

print("\n🚀 Starting training...")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_class_loss = 0
    train_sev_loss = 0
    train_bleed_loss = 0
    train_swell_loss = 0
    train_emerg_loss = 0
    
    for batch in train_loader:
        images = batch[0].to(DEVICE)
        labels = batch[1].to(DEVICE)
        severities = batch[2].float().to(DEVICE)
        bleedings = batch[3].float().to(DEVICE)
        swellings = batch[4].float().to(DEVICE)
        emergencies = batch[5].float().to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_class, pred_sev, pred_bleed, pred_swell, pred_emerg = model(images)
        
        # Calculate losses
        loss_class = criterion_class(pred_class, labels)
        loss_sev = criterion_severity(pred_sev.squeeze(), severities)
        loss_bleed = criterion_binary(pred_bleed.squeeze(), bleedings)
        loss_swell = criterion_binary(pred_swell.squeeze(), swellings)
        loss_emerg = criterion_binary(pred_emerg.squeeze(), emergencies)
        
        # Total loss (weighted)
        total_loss = loss_class + 0.5 * loss_sev + loss_bleed + loss_swell + loss_emerg
        
        total_loss.backward()
        optimizer.step()
        
        train_class_loss += loss_class.item()
        train_sev_loss += loss_sev.item()
        train_bleed_loss += loss_bleed.item()
        train_swell_loss += loss_swell.item()
        train_emerg_loss += loss_emerg.item()
    
    # Validation
    model.eval()
    val_class_loss = 0
    val_sev_loss = 0
    val_bleed_loss = 0
    val_swell_loss = 0
    val_emerg_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            severities = batch[2].float().to(DEVICE)
            bleedings = batch[3].float().to(DEVICE)
            swellings = batch[4].float().to(DEVICE)
            emergencies = batch[5].float().to(DEVICE)
            
            pred_class, pred_sev, pred_bleed, pred_swell, pred_emerg = model(images)
            
            val_class_loss += criterion_class(pred_class, labels).item()
            val_sev_loss += criterion_severity(pred_sev.squeeze(), severities).item()
            val_bleed_loss += criterion_binary(pred_bleed.squeeze(), bleedings).item()
            val_swell_loss += criterion_binary(pred_swell.squeeze(), swellings).item()
            val_emerg_loss += criterion_binary(pred_emerg.squeeze(), emergencies).item()
    
    avg_val_loss = (val_class_loss + val_sev_loss + val_bleed_loss + val_swell_loss + val_emerg_loss) / 5
    
    # Print progress
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    print(f"  Train - Class: {train_class_loss/len(train_loader):.4f} | Sev: {train_sev_loss/len(train_loader):.4f} | Bleed: {train_bleed_loss/len(train_loader):.4f}")
    print(f"  Val   - Class: {val_class_loss/len(val_loader):.4f} | Sev: {val_sev_loss/len(val_loader):.4f} | Bleed: {val_bleed_loss/len(val_loader):.4f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "multi_output_wound_model.pth")
        print(f"  ✅ Saved best model")
    
    scheduler.step(avg_val_loss)

# =========================================================
# TEST EVALUATION
# =========================================================

print("\n" + "=" * 60)
print("📊 Evaluating on Test Set...")
print("=" * 60)

model.load_state_dict(torch.load("multi_output_wound_model.pth"))
model.eval()

test_class_correct = 0
test_sev_errors = []
test_bleed_correct = 0
test_swell_correct = 0
test_emerg_correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        images = batch[0].to(DEVICE)
        labels = batch[1].to(DEVICE)
        severities = batch[2].float().to(DEVICE)
        bleedings = batch[3].float().to(DEVICE)
        swellings = batch[4].float().to(DEVICE)
        emergencies = batch[5].float().to(DEVICE)
        
        pred_class, pred_sev, pred_bleed, pred_swell, pred_emerg = model(images)
        
        # Classification accuracy
        _, predicted = torch.max(pred_class, 1)
        test_class_correct += (predicted == labels).sum().item()
        
        # Severity MAE
        test_sev_errors.extend(torch.abs(pred_sev.squeeze() - severities).cpu().numpy())
        
        # Binary accuracies
        test_bleed_correct += ((pred_bleed.squeeze() > 0.5) == bleedings.bool()).sum().item()
        test_swell_correct += ((pred_swell.squeeze() > 0.5) == swellings.bool()).sum().item()
        test_emerg_correct += ((pred_emerg.squeeze() > 0.5) == emergencies.bool()).sum().item()
        
        total += len(labels)

print(f"\n🎯 Test Results:")
print(f"   Injury Type Accuracy: {test_class_correct/total:.2%}")
print(f"   Severity MAE: {np.mean(test_sev_errors):.2f} (avg error in points)")
print(f"   Bleeding Accuracy: {test_bleed_correct/total:.2%}")
print(f"   Swelling Accuracy: {test_swell_correct/total:.2%}")
print(f"   Emergency Accuracy: {test_emerg_correct/total:.2%}")

print("\n" + "=" * 60)
print("✅ Training Complete!")
print("📁 Model saved to: multi_output_wound_model.pth")
print("=" * 60)

# Save class mapping
with open("class_mapping.json", "w") as f:
    json.dump(CLASS_MAPPING, f, indent=2)