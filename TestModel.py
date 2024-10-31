"""
This module provides functions to display images from tensors and test a trained model on a test dataset. It was provided as a part of the course material and was not modified.

Functions
---------
imshow(inp, title=None)
test_model(test_loader, trained_model, class_names)
"""

import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

def imshow(inp, title=None):
    """
    Display an image from a Tensor.

    Parameters
    ----------
    inp : Tensor
        The input tensor to be displayed. The tensor is expected to have shape (C, H, W).
    title : str, optional
        The title of the image. Default is None.

    Notes
    -----
    The function assumes the input tensor is normalized with mean [0.485, 0.456, 0.406] and 
    standard deviation [0.229, 0.224, 0.225]. The tensor is denormalized before displaying.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2.0)

def test_model(test_loader, trained_model, class_names):
    """
    Test the trained model on the test dataset and display the results.

    Parameters
    ----------
    test_loader : DataLoader
        DataLoader for the test dataset.
    trained_model : torch.nn.Module
        The trained model to be tested.
    class_names : list of str
        List of class names corresponding to the model's output classes.

    Returns
    -------
    None
    """
    print('showing test results')
    with torch.no_grad():
        #Get correct device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trained_model = trained_model.to(device)
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inp = torchvision.utils.make_grid(inputs)
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(len(inputs)):
                inp = inputs.data[i]
                imshow(inp, class_names[preds[i]])