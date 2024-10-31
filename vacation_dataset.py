"""
VacationDataset class for loading and transforming vacation images.
This was not used since the ImageFolder class was used instead.

Attributes
----------
annotations : pandas.DataFrame
    DataFrame containing image file names and corresponding labels.
root_dir : str
    Directory with all the images.
transform : callable, optional
    Optional transform to be applied on a sample.
target_transform : callable, optional
    Optional transform to be applied on the target.

Methods
-------
__len__()
    Returns the number of images in the dataset.
__getitem__(idx)
    Returns the image and label at the specified index.

Parameters
----------
annotations_file : str
    Path to the CSV file with annotations.
root_dir : str
    Directory with all the images.
transform : callable, optional
    Optional transform to be applied on a sample.
target_transform : callable, optional
    Optional transform to be applied on the target.

Initialize the VacationDataset.
Args:
    annotations_file (str): Path to the CSV file with annotations.
    root_dir (str): Directory with all the images.
    transform (callable, optional): Optional transform to be applied on a sample.
    target_transform (callable, optional): Optional transform to be applied on the target.

Return the number of images in the dataset.
Returns:
    int: Number of images in the dataset.

Get the image and label at the specified index.
Args:
    idx (int): Index of the image and label to retrieve.
Returns:
    tuple: (image, label) where image is the loaded image tensor and label is the corresponding label.
"""

from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class VacationDataset(Dataset):
    """
    Dataset class for vacation images.

    Parameters
    ----------
    annotations_file : str
        Path to the CSV file with annotations.
    root_dir : str
        Directory with all the images.
    transform : callable, optional
        Optional transform to be applied on a sample.
    target_transform : callable, optional
        Optional transform to be applied on the target.

    Attributes
    ----------
    annotations : pandas.DataFrame
        DataFrame containing image file names and labels.
    root_dir : str
        Directory with all the images.
    transform : callable
        Transform to be applied on a sample.
    target_transform : callable
        Transform to be applied on the target.

    Methods
    -------
    __len__()
        Returns the number of images in the dataset.
    __getitem__(idx)
        Loads and returns the image and label at the specified index.
    """

    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
        """
        Initialize the dataset with annotations file, root directory, and optional transforms.

        Parameters
        ----------
        annotations_file : str
            Path to the CSV file containing annotations.
        root_dir : str
            Directory with all the images.
        transform : callable, optional
            Optional transform to be applied on a sample.
        target_transform : callable, optional
            Optional transform to be applied on the target.
        """
        
        self.annotations = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns
        -------
        int
            The number of images in the dataset.
        """
        
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Load image and label at index idx.

        Parameters
        ----------
        idx : int
            Index of the image and label to be loaded.

        Returns
        -------
        tuple
            A tuple containing the transformed image and label.
        """
       
        # Read image
        img_name = Path(self.annotations.iloc[idx, 0])
        class_name = self.annotations.iloc[idx, 1]
        img_path = str(Path(self.root_dir, class_name, img_name))
        image = read_image(img_path)
        
        # Label
        label = self.annotations.iloc[idx, 2]
        
        # Transform image and label
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
