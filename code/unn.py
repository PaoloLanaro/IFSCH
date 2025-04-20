import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms
import torch.optim as optim
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2

import time

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        """
        U-Net for climbing hold segmentation
        n_channels: number of input channels (3 for RGB images)
        n_classes: number of output classes (2 for binary segmentation - hold/not hold)
        """
        super(UNet, self).__init__()

        # 4 Encoder blocks to extract features
        self.enc1 = self.double_conv(n_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)

        # Drop some amount of information so that we don't overfit and regularize
        self.pool = nn.MaxPool2d(2)
        # tested optimal dropout rate
        self.dropout = nn.Dropout(0.3)
        self.bottleneck = self.double_conv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(1024, 512)  # 1024 due to concatenation

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(512, 256)   # 512 due to concatenation

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(256, 128)   # 256 due to concatenation

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(128, 64)    # 128 due to concatenation

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        b = self.dropout(b)

        # Decoder path
        d4 = self.upconv4(b)
        # Skip connection
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        # Skip connection
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        # Skip connection
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        # Skip connection
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        logits = self.outc(d1)

        return logits

    def double_conv(self, in_channels, out_channels):
        """Double convolution block with batch normalization"""
        # Padding allows the output image to be the same as the input size
        # Bath normalization allows for stabilizing the neural net during training
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ClimbingWallDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        """
        Custom dataset for climbing wall segmentation

        images_dir: directory containing the input images
        masks_dir: directory containing the VGG IA JSON files
        transform: transforms to apply to the images
        mask_transform: transforms to apply to the masks
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform

        # Get all image files
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load corresponding mask from JSON
        json_name = os.path.splitext(image_name)[0] + '.json'
        json_path = os.path.join(self.masks_dir, json_name)

        # Initialize an empty mask with the same dimensions as the image
        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        # Parse JSON and create mask
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)

                    # Handle VGG Image Annotator format
                    if '_via_img_metadata' in data:
                        # Get the first key in _via_img_metadata (there should be only one for single image annotation)
                        img_key = next(iter(data['_via_img_metadata']))
                        img_data = data['_via_img_metadata'][img_key]

                        # Process all regions
                        for region in img_data.get('regions', []):
                            shape_attrs = region.get('shape_attributes', {})

                            if shape_attrs.get('name') == 'polygon':
                                # Get polygon points
                                x_points = shape_attrs.get('all_points_x', [])
                                y_points = shape_attrs.get('all_points_y', [])

                                # Create points array for CV2
                                if len(x_points) == len(y_points) and len(x_points) > 2:
                                    points = np.array(list(zip(x_points, y_points)), dtype=np.int32)
                                    # Fill polygon
                                    cv2.fillPoly(mask, [points], 1)

                            elif shape_attrs.get('name') == 'rect':
                                # Get rectangle coordinates
                                x = shape_attrs.get('x', 0)
                                y = shape_attrs.get('y', 0)
                                width = shape_attrs.get('width', 0)
                                height = shape_attrs.get('height', 0)

                                # Fill rectangle
                                cv2.rectangle(mask, (x, y), (x + width, y + height), 1, -1)

                    # Handle simpler format
                    elif 'shapes' in data:
                        for shape in data['shapes']:
                            if shape.get('shape_type') == 'polygon':
                                points = np.array(shape['points'], dtype=np.int32)
                                cv2.fillPoly(mask, [points], 1)
                            elif shape.get('shape_type') == 'rectangle':
                                p1, p2 = shape['points']
                                cv2.rectangle(mask, tuple(p1), tuple(p2), 1, -1)

                except Exception as e:
                    print(f"Error parsing JSON for {json_name}: {e}")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert mask to tensor
        if self.mask_transform:
            mask = self.mask_transform(mask)

            mask = torch.tensor(np.array(mask), dtype=torch.long)
        else:
            mask = torch.from_numpy(mask).long()

        return image, mask

# handle class imbalance for background foreground
def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for segmentation
    """
    pred = F.softmax(pred, dim=1)
    pred = pred[:, 1, :, :]  # Get the probability for the hold class

    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def train_model(model, device, train_loader, val_loader, optimizer, num_epochs=25):
    """
    Training function for the segmentation model
    """
    best_val_loss = float('inf')
    loss_history = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            # CrossEntropyLoss for multi-class segmentation
            ce_loss = F.cross_entropy(outputs, masks)

            # Dice loss for better segmentation quality (
            d_loss = dice_loss(outputs, masks.float())

            # Combined loss
            loss = ce_loss + d_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                # Calculate validation loss
                ce_loss = F.cross_entropy(outputs, masks)
                d_loss = dice_loss(outputs, masks.float())
                loss = ce_loss + d_loss

                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {round(epoch_loss, 6)}, Val Loss: {round(val_loss, 6)}')

        loss_history.append(val_loss)
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return model, loss_history

def main():
    # Set training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # current optimal image size
    image_size = 192

    # Define transforms
    image_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)
        # transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = ClimbingWallDataset(
        images_dir='../data/processed/train/images',
        masks_dir='../data/processed/train/annotations',
        transform=image_transforms,
        mask_transform=mask_transforms
    )

    val_dataset = ClimbingWallDataset(
        images_dir='../data/processed/val/images',
        masks_dir='../data/processed/val/annotations',
        transform=image_transforms,
        mask_transform=mask_transforms
    )

    # Create data loaders
    # tested optimal batch size
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model
    model = UNet(n_channels=3, n_classes=2).to(device)

    # Define optimizer
    # tested optimal optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    trained_model, loss_history = train_model(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=250
    )

    print("Training complete")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', markersize=3)
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig('loss_history.png')
    plt.show()

    # Save the final model
    torch.save(trained_model.state_dict(), 'latest_segmentation_model.pth')

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    print(f'Finished training in {round(total_minutes, 4)} minutes')

