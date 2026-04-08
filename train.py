import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

DEVICE = "cuda"
DATA_DIR = "PlantVillage"
EPOCHS = 10
BATCH  = 32

print(f"GPU : {torch.cuda.get_device_name(0)}")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Deux datasets séparés avec leurs propres transforms
train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform_train)
val_dataset   = datasets.ImageFolder(DATA_DIR, transform=transform_val)
classes = train_dataset.classes
print(f"{len(classes)} classes trouvées")

# Split identique sur les deux
indices    = list(range(len(train_dataset)))
split      = int(0.8 * len(indices))
torch.manual_seed(42)
perm       = torch.randperm(len(indices)).tolist()
train_idx  = perm[:split]
val_idx    = perm[split:]

train_loader = DataLoader(Subset(train_dataset, train_idx),
                          batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(Subset(val_dataset, val_idx),
                          batch_size=BATCH, shuffle=False, num_workers=0)

# Modèle
model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = nn.Linear(1024, len(classes))
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_acc = 0.0
for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += labels.size(0)

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)

    train_acc = 100 * correct / total
    val_acc   = 100 * val_correct / val_total
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.3f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"model": model.state_dict(), "classes": classes}, "plant_model.pth")
        print(f"  -> Modèle sauvegardé (val acc: {val_acc:.1f}%)")

print(f"\nTerminé ! Meilleure précision : {best_acc:.1f}%")