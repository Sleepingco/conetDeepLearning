import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

# 설정
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 데이터 경로
train_dir = './cats_and_dogs_filtered/cats_and_dogs_filtered/train'
test_dir = './cats_and_dogs_filtered/cats_and_dogs_filtered/validation'

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 설정
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# 손실함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# 저장 설정
save_path = 'cats_and_dogs_filtered_vgg16_pytorch.pth'
best_loss = float('inf')
patience = 5
wait = 0

# 시각화를 위한 기록 리스트
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []


# 학습 루프
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0.0, 0

    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    train_loss /= len(train_loader.dataset)
    train_acc = train_correct.double() / len(train_loader.dataset)

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    val_loss /= len(test_loader.dataset)
    val_acc = val_correct.double() / len(test_loader.dataset)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    # 기록 저장
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc.item())
    val_accuracies.append(val_acc.item())

    # EarlyStopping & Checkpoint
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), save_path)
        wait = 0
        print("✔️ Best model saved.")
    else:
        wait += 1
        if wait >= patience:
            print("🛑 Early stopping.")
            break

# 학습 그래프 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# miniconda3/envs/torchpip/python.exe c:/VisualStudio-WorkSpace/conetDeepLearning/20250528/transferlearning_pytorch.py
# C:\Users\main\miniconda3\envs\torchpip\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# C:\Users\main\miniconda3\envs\torchpip\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Epoch 1/30: 100%|██████████████████████████████████| 63/63 [00:22<00:00,  2.74it/s]
# Epoch 1: Train Loss=0.2734, Acc=0.8650 | Val Loss=0.0555, Acc=0.9800
# ✔️ Best model saved.
# Epoch 2/30: 100%|██████████████████████████████████| 63/63 [00:22<00:00,  2.80it/s]
# Epoch 2: Train Loss=0.1487, Acc=0.9300 | Val Loss=0.0338, Acc=0.9900
# ✔️ Best model saved.
# Epoch 3/30: 100%|██████████████████████████████████| 63/63 [00:22<00:00,  2.77it/s]
# Epoch 3: Train Loss=0.1053, Acc=0.9530 | Val Loss=0.0344, Acc=0.9880
# Epoch 4/30: 100%|██████████████████████████████████| 63/63 [00:22<00:00,  2.77it/s]
# Epoch 4: Train Loss=0.1302, Acc=0.9335 | Val Loss=0.0511, Acc=0.9780
# Epoch 5/30: 100%|██████████████████████████████████| 63/63 [00:22<00:00,  2.77it/s]
# Epoch 5: Train Loss=0.1164, Acc=0.9525 | Val Loss=0.0303, Acc=0.9880
# ✔️ Best model saved.
# Epoch 6/30: 100%|██████████████████████████████████| 63/63 [00:22<00:00,  2.75it/s]
# Epoch 6: Train Loss=0.1178, Acc=0.9455 | Val Loss=0.0255, Acc=0.9910
# ✔️ Best model saved.
# Epoch 7/30: 100%|██████████████████████████████████| 63/63 [00:22<00:00,  2.75it/s]
# Epoch 7: Train Loss=0.1009, Acc=0.9555 | Val Loss=0.0275, Acc=0.9870
# Epoch 8/30: 100%|██████████████████████████████████| 63/63 [00:22<00:00,  2.76it/s]
# Epoch 8: Train Loss=0.1029, Acc=0.9540 | Val Loss=0.0236, Acc=0.9900
# ✔️ Best model saved.
# Epoch 9/30: 100%|██████████████████████████████████| 63/63 [00:23<00:00,  2.69it/s]
# Epoch 9: Train Loss=0.1125, Acc=0.9485 | Val Loss=0.0366, Acc=0.9870
# Epoch 10/30: 100%|█████████████████████████████████| 63/63 [00:22<00:00,  2.75it/s]
# Epoch 10: Train Loss=0.0859, Acc=0.9610 | Val Loss=0.0258, Acc=0.9880
# Epoch 11/30: 100%|█████████████████████████████████| 63/63 [00:23<00:00,  2.72it/s]
# Epoch 11: Train Loss=0.0840, Acc=0.9665 | Val Loss=0.0416, Acc=0.9840
# Epoch 12/30: 100%|█████████████████████████████████| 63/63 [00:23<00:00,  2.71it/s]
# Epoch 12: Train Loss=0.0890, Acc=0.9600 | Val Loss=0.0559, Acc=0.9790
# Epoch 13/30: 100%|█████████████████████████████████| 63/63 [00:23<00:00,  2.70it/s]
# Epoch 13: Train Loss=0.0821, Acc=0.9640 | Val Loss=0.0331, Acc=0.9860
# 🛑 Early stopping