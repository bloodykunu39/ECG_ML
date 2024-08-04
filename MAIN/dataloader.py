"This module contains the functions to convert the dataset into dataloader"

import torch
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, images, labels):
        """
        Args:
            images (list of numpy arrays or tensors): List of image data.
            labels (list of int): List of labels corresponding to the images.
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        # Return the total number of samples
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve the image and label at index `idx`
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert image and label to tensors if they are not already
        image = torch.tensor(image, dtype=torch.float32)  # Adjust dtype as needed
        label = torch.tensor(label, dtype=torch.long)  # Long is typically used for classification labels
        
        return image, label
