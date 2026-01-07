import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import Config


class KidneyCTDataset(Dataset):
    """
    PyTorch Dataset for Kidney CT images
    """

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # grayscale → 3-channel (CvT expects RGB-like input)
        image = np.stack([image, image, image], axis=0)
        image = torch.FloatTensor(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def load_images(data_dir, img_size=Config.IMG_SIZE, labels=Config.LABELS):
    """
    Load and preprocess kidney CT images

    Directory structure:
        data_dir/
            Normal/
            Cyst/
            Tumor/
            Stone/
    """
    data = []

    for label in labels:
        label_path = os.path.join(data_dir, label)
        class_idx = labels.index(label)

        if not os.path.exists(label_path):
            print(f"Warning: {label_path} not found")
            continue

        for fname in os.listdir(label_path):
            img_path = os.path.join(label_path, fname)

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, (img_size, img_size))
                data.append([img, class_idx])

            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    return np.array(data, dtype=object)


def prepare_data(dataset_path):
    """
    Prepare train/test splits for Kidney CT dataset

    Kaggle dataset structure:
        dataset_path/
            Normal/
            Cyst/
            Tumor/
            Stone/
    """
    print("Loading Kidney CT data...")

    data = load_images(dataset_path)

    images = np.array([x[0] for x in data], dtype=np.float32) / 255.0
    labels = np.array([x[1] for x in data], dtype=np.int64)

    # deterministic split
    rng = np.random.default_rng(Config.SEED)
    indices = rng.permutation(len(images))

    split_idx = int(len(images) * Config.TRAIN_SPLIT)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    x_train, y_train = images[train_idx], labels[train_idx]
    x_test, y_test = images[test_idx], labels[test_idx]

    print(f"Train samples: {len(x_train)}")
    print(f"Test samples : {len(x_test)}")

    for i, cls in enumerate(Config.LABELS):
        print(f"Train {cls}: {np.sum(y_train == i)}")
        print(f"Test  {cls}: {np.sum(y_test == i)}")

    return x_train, y_train, x_test, y_test


def create_dataloaders(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS
):
    """
    Create PyTorch DataLoaders
    """
    train_dataset = KidneyCTDataset(x_train, y_train)
    test_dataset = KidneyCTDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
