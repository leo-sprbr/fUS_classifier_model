import torch
import os
import torch.nn as nn
import torch.optim as optim
from transformers import VideoMAEForVideoClassification, VideoMAEConfig
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates  # For elastic morphing
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
# Re Load data file
# Set the directory where the .pt files were saved
directory = "/network/iss/cenir/analyse/irm/users/leo.sperber/"  # Adjust if needed

print("ONLY SECUNDO DATA WITHOUT HIGH PASS FILTER ACQ-WISE SPLIT, MASKED CBV, NO SHAVE OFF SIDES, SMALL VIDEO-MAE TEST WITH DROPOUT, WINDOW 16, STRIDE 1 BALANCED DATASET ZSCORE NORM HIGH PASSS")

# Dataset parameters
window_size = 16
stride = 1  # Overlapping windows for train/test since acquisitions are separated
image_size = 112
exclusion_distance = 50
batch_size = 128
num_workers = 0  # Adjust based on your system
num_channels = 1
num_labels = 2
num_epochs = 100


# Paths to the saved files
train_images_path = os.path.join(directory, r"train_images_112x112_clip99_5_split0.7_modeacq_hp.pt")
train_labels_path= os.path.join(directory, r"train_labels_112x112_clip99_5_split0.7_modeacq_hp.pt")
test_images_path = os.path.join(directory, r"test_images_112x112_clip99_5_split0.7_modeacq_hp.pt")
test_labels_path= os.path.join(directory, r"test_labels_112x112_clip99_5_split0.7_modeacq_hp.pt")
train_acqs_path = os.path.join(directory, r"train_acq_ids_112x112_clip99_5_split0.7_modeacq_hp.pt")
test_acqs_path = os.path.join(directory, r"test_acq_ids_112x112_clip99_5_split0.7_modeacq_hp.pt")


# Load the tensors
train_images = torch.load(train_images_path)
train_labels = torch.load(train_labels_path)
test_images = torch.load(test_images_path)
test_labels = torch.load(test_labels_path)
train_acqs = torch.load(train_acqs_path)
test_acqs = torch.load(test_acqs_path)

#train_images = train_images.unsqueeze(dim=1)
#test_images = test_images.unsqueeze(dim=1)

# Print shapes to confirm
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Train acquisition indices shape: {train_acqs.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")
print(f"Test acquisition indices shape: {test_acqs.shape}")


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using device: {device}")

# Dataset class with train/test mode
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import math

def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel."""
    x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)  # Ensure float32
    y = x
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel
    
class GaussianNoise:
    """Add Gaussian noise to the tensor."""
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

class FilteredFUSWindowDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor, window_size=8, stride=1, image_size=112, mode='train'):
        """
        Custom Dataset for sliding windows of FUS frames with cropping and normalization.
        For 'train' mode: Balances windows between labels 0 and 1 by downsampling majority class, and applies data augmentation.
        For 'test' mode: Includes all windows, no balancing, no augmentation.
        
        Args:
            data_tensor (torch.Tensor): Shape [N, 1, H, W] (e.g., [6000, 1, 112, 112]).
            labels_tensor (torch.Tensor): Shape [N] with labels (0, 1). Assumes -1 already excluded.
            window_size (int): Number of frames per window (e.g., 8).
            stride (int): Step size for sliding window (e.g., 9 for non-overlapping with small gap).
            image_size (int): Target square size for frames (e.g., 112).
            mode (str): 'train' or 'test' to control balancing and augmentation.
        """
        assert data_tensor.shape[0] == labels_tensor.shape[0], "Data and labels length mismatch"
        assert window_size > 0, "Window size must be positive"
        assert mode in ['train', 'test'], "Mode must be 'train' or 'test'"
        
        self.data = data_tensor
        self.labels = labels_tensor
        self.window_size = window_size
        self.stride = stride
        self.image_size = image_size
        self.mode = mode
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size))  # Ensure square (redundant if already sized)
        ])

        # Build valid indices, skipping excluded zones
        self.valid_indices = []
        for i in range(window_size - 1, len(self.labels), stride):
            if self.labels[i] != -1:  # Assume -1 already filtered out in preprocessing
                self.valid_indices.append(i)

        if len(self.valid_indices) == 0:
            print("Warning: No valid windows found. Check labels or window_size.")
        
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        try:
            end_idx = self.valid_indices[idx]
            start_idx = end_idx - self.window_size + 1
            if start_idx < 0 or end_idx >= len(self.data):
                raise IndexError(f"Invalid window range: [{start_idx}:{end_idx + 1}]")
            
            window = self.data[start_idx:end_idx + 1]  # [T, 1, H, W]
            H, W = window.shape[-2:]
            window = torch.stack([self.transform(frame) for frame in window])  # [T, 1, 112, 112]
            
            # Verify window shape
            if window.shape[0] != self.window_size:
                raise ValueError(f"Expected {self.window_size} frames, got {window.shape[0]}")
                
            # === DATA AUGMENTATION (ONLY IN TRAIN MODE) ===
            if self.mode == 'train':
                # 1. Random Affine (same transformation for all frames)
                affine = T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))
                angle, translations, scale, shear = affine.get_params(
                    affine.degrees, affine.translate, affine.scale, affine.shear,
                    (self.image_size, self.image_size)
                )
                for t in range(window.size(0)):
                    window[t] = TF.affine(
                        window[t], angle=angle, translate=translations,
                        scale=scale, shear=shear,
                        interpolation=TF.InterpolationMode.BILINEAR, fill=0
                    )
                '''
                # 2. Elastic deformation (same deformation for all frames)
                alpha = 50.0
                sigma = 5.0
                filter_size = max(5, math.ceil(3 * sigma))
                kernel = gaussian_kernel(filter_size, sigma).unsqueeze(0).unsqueeze(0).to(window.device)
                padding = filter_size // 2

                dx = (torch.rand(1, 1, H, W, device=window.device, dtype=torch.float32) * 2 - 1)
                dy = (torch.rand(1, 1, H, W, device=window.device, dtype=torch.float32) * 2 - 1)

                dx = F.conv2d(dx, kernel, padding=padding) * alpha
                dy = F.conv2d(dy, kernel, padding=padding) * alpha

                dx = dx.squeeze()
                dy = dy.squeeze()

                norm_dx = dx * 2 / W
                norm_dy = dy * 2 / H

                x_coords = torch.linspace(-1, 1, W, device=window.device, dtype=torch.float32)
                y_coords = torch.linspace(-1, 1, H, device=window.device, dtype=torch.float32)
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

                grid = torch.stack([xx + norm_dx, yy + norm_dy], dim=-1).unsqueeze(0)  # [1, H, W, 2]
                  
                for t in range(window.size(0)):
                    frame = window[t].unsqueeze(0)  # [1, 1, H, W]
                    # print('frame.shape', frame.shape, 'grid.shape', grid.shape)
                    warped = F.grid_sample(
                        frame.float(), grid.float(), mode='bilinear',
                        padding_mode='zeros', align_corners=True
                    )
                    window[t] = warped.squeeze(0)
                '''
                # 3. Gaussian noise (independent per frame)
                noise = GaussianNoise(std=0.01)
                window = noise(window)
            
            
            label = self.labels[end_idx]
            return window, label
        except Exception as e:
            print(f"Error in __getitem__ at idx {idx}, end_idx {end_idx}: {e}")
            raise

# Create datasets
train_dataset = FilteredFUSWindowDataset(
    data_tensor=train_images,
    labels_tensor=train_labels,
    window_size=window_size,
    stride=stride,
    image_size=image_size,
    mode='train'
)

test_dataset = FilteredFUSWindowDataset(
    data_tensor=test_images,
    labels_tensor=test_labels,
    window_size=window_size,
    stride=stride,
    image_size=image_size,
    mode='test'
)

print(f"Train dataset size: {len(train_dataset)} windows")
print(f"Test dataset size: {len(test_dataset)} windows")


def compute_label_distribution(dataset):
    """
    Compute label counts from the dataset's valid windows.
    """
    if len(dataset.valid_indices) == 0:
        return {label: 0 for label in range(num_labels)}
    
    valid_labels = dataset.labels[dataset.valid_indices].numpy()  # Convert to numpy for unique
    unique, counts = np.unique(valid_labels, return_counts=True)
    dist = dict(zip(unique, counts))
    
    # Fill missing labels with 0
    for label in range(num_labels):
        if label not in dist:
            dist[label] = 0
    
    return dist

# Compute distributions
train_dist = compute_label_distribution(train_dataset)
test_dist = compute_label_distribution(test_dataset)

# Print distributions
print("Train Dataset Label Distribution:")
for label, count in train_dist.items():
    print(f"Label {label}: {count} ({count / len(train_dataset) * 100:.2f}%)" if len(train_dataset) > 0 else f"Label {label}: 0 (0.00%)")

print("\nTest Dataset Label Distribution:")
for label, count in test_dist.items():
    print(f"Label {label}: {count} ({count / len(test_dataset) * 100:.2f}%)" if len(test_dataset) > 0 else f"Label {label}: 0 (0.00%)")

# Compute class weights based on train distribution (inverse frequency)
total_train_samples = len(train_dataset)
class_weights = []
for label in range(num_labels):
    count = train_dist.get(label, 0)
    if count > 0:
        weight = total_train_samples / (num_labels * count)
    else:
        weight = 0.0  # Or handle as needed (e.g., 1.0 if no samples)
    class_weights.append(weight)

weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print("\nClass Weights Tensor for CrossEntropyLoss (based on train dist):")
print(weights_tensor)


# Create DataLoaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True  # Avoid partial batch issues
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # No shuffling for test
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False  # Keep all test samples
)

print(f"Train DataLoader: {len(train_dataloader)} batches of size {batch_size}")
print(f"Test DataLoader: {len(test_dataloader)} batches of size {batch_size}")

print("Trying to import videoMAE from HuggingFace")
# Load pre-trained model name
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

model_name = "MCG-NJU/videomae-small-finetuned-kinetics"

# Custom config for your FUS data
config = VideoMAEConfig.from_pretrained(model_name)
config.num_channels = num_channels  # Grayscale FUS data
config.image_size = image_size  # Your frame size
config.num_frames = window_size   # Your window size
config.num_labels = num_labels   # Work (0) vs Pause (1)
config.attention_dropout = 0.1  # Add dropout to attention weights
config.hidden_dropout = 0.1     # Add dropout in transformer blocks
config.classifier_dropout = 0.5 # Add dropout before classifier

# Initialize model with custom config
model = VideoMAEForVideoClassification(config)

# Load pre-trained weights and adapt for 1 channel
pretrained_model = VideoMAEForVideoClassification.from_pretrained(model_name)
pretrained_dict = pretrained_model.state_dict()

# Adapt patch embedding weights: Average over RGB channels
pretrained_embed_weight = pretrained_dict['videomae.embeddings.patch_embeddings.projection.weight']  # Smaller: [768, 3, 2, 16, 16]
new_embed_weight = pretrained_embed_weight.mean(dim=1, keepdim=True)  # [768, 1, 2, 16, 16]
pretrained_dict['videomae.embeddings.patch_embeddings.projection.weight'] = new_embed_weight

# Remove classifier weights (reinitialized for 2 classes)
del pretrained_dict['classifier.weight']
del pretrained_dict['classifier.bias']

# Load adapted weights
model.load_state_dict(pretrained_dict, strict=False)
model.to(device)
print("Base VideoMAE loaded with adapted weights")


# BEST PRACTICE SETTINGS FOR VideoMAE FINE-TUNING
base_lr            = 3e-6        # Backbone: very low
classifier_lr      = 1e-5        # Head can learn faster
weight_decay       = 0.05
warmup_ratio       = 0.1         # 10% of total steps as warmup
label_smoothing    = 0.1
effective_batch_size = 128       # Target effective batch size
actual_batch_size  = train_dataloader.batch_size
accumulation_steps = max(1, effective_batch_size // actual_batch_size)

# Model path
best_model_path = f'fus_videomae_small_{window_size}_hp_weighted_zscore_masked_STABLE.pth'

# ============================= LOSS & OPTIMIZER =============================
criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device), label_smoothing=label_smoothing)

# Layer-wise learning rates (critical!)
backbone_params = []
classifier_params = []

for name, param in model.named_parameters():
    if "classifier" in name:
        classifier_params.append(param)
    else:
        backbone_params.append(param)

optimizer = optim.AdamW([
    {"params": backbone_params,     "lr": base_lr},
    {"params": classifier_params,   "lr": classifier_lr}
], weight_decay=weight_decay)

# Total training steps
total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(warmup_ratio * total_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ============================= TRACKING =============================
train_accs = []
test_accs = []
train_losses = []
test_losses = []
best_test_loss = float('inf')
patience = 30
patience_counter = 0

print(f"Training with:")
print(f"  Backbone LR: {base_lr}, Classifier LR: {classifier_lr}")
print(f"  Effective batch size: {effective_batch_size} (accum={accumulation_steps})")
print(f"  Warmup: {warmup_steps} steps, Cosine decay")
print(f"  Label smoothing: {label_smoothing}")
print(f"  Early stopping patience: {patience}")

# ============================= TRAINING LOOP =============================
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0
    optimizer.zero_grad()

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for step, (batch_windows, batch_labels) in enumerate(progress_bar):
        inputs = batch_windows.to(device).float()
        labels = batch_labels.to(device, dtype=torch.long)

        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels) / accumulation_steps  # Scale loss

        loss.backward()

        # Gradient accumulation
        if (step + 1) % accumulation_steps == 0 or (step + 1 == len(train_dataloader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent explosions
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps  # Unscale for logging
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'loss': f'{train_loss/(step+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{current_lr:.2e}'
        })

    # Epoch stats
    train_acc = 100. * correct / total
    avg_train_loss = train_loss / len(train_dataloader)
    train_accs.append(train_acc)
    train_losses.append(avg_train_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # ============================= VALIDATION =============================
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_windows, batch_labels in test_dataloader:
            inputs = batch_windows.to(device).float()
            labels = batch_labels.to(device, dtype=torch.long)

            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_test_loss = test_loss / len(test_dataloader)
    test_acc = 100. * correct / total
    test_accs.append(test_acc)
    test_losses.append(avg_test_loss)

    print(f"? Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}% | LR: {current_lr:.2e}")

    # ============================= SAVE BEST + EARLY STOP =============================
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"NEW BEST MODEL SAVED! Test Loss: {best_test_loss:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print("Early stopping triggered!")
        break

# ============================= FINAL PLOTS =============================
print(f"Training complete. Best model: {best_model_path}")

# Accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(train_accs, label='Train Accuracy', linewidth=2)
plt.plot(test_accs, label='Test Accuracy', linewidth=2)
plt.title('Accuracy over Epochs (Stable Training)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'accuracy_plot_weighted_{window_size}_zscore_hp.png', dpi=300)
plt.close()

# Loss plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(test_losses, label='Test Loss', linewidth=2)
plt.title('Loss over Epochs (Stable Training)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'loss_plot_weighted_{window_size}_zscore_hp.png', dpi=300)
plt.close()

print("Plots saved. You're now training like a pro!")