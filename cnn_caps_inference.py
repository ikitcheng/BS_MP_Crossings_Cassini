# Script: detect.py
# Author: I Kit Cheng
# general libraries
import time
import gc
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy
from tqdm import tqdm
from tqdm import trange
from glob import glob
import h5py
from IPython import display
import pandas as pd
import itertools


# ML metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc, classification_report

# tensorflow
import tensorflow as tf

# pytorch deep learning
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.models import resnet18
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #
def load_model(path, model_filename="best.pt"): 
    """ Load a pretrained model. """
    with open(os.path.join(path, model_filename), 'rb') as f:
        if torch.cuda.is_available():
            model = torch.load(f) 
        else:
            model = torch.load(f, map_location=torch.device('cpu'))
    return model.eval().to(device)


def save_pkl(result, filename):
    with open(filename, "wb") as f:
        pickle.dump(result, f)


def load_pkl(filename):
    """ Load pickle file. """
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result


def load_img(filename):
    """ Load .jpg or .png image as tensor. """
    img = tf.keras.preprocessing.image.load_img(filename, color_mode='rgb', target_size=(224,224))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = np.array([img_arr])
    img_arr /= 255
    img_tensor = torch.FloatTensor(img_arr).permute([0,3,1,2])
    return img_tensor

def plot_img_tensor(img_tensor, label=None):
    """ Assume shape of img_tensor is (1,3,224,224) """
    plt.imshow(img_tensor.squeeze().permute(1,2,0))
    if label is not None:
        print('label: ', label.numpy())
    plt.show()

def preprocess_capsplot(filename, transform=None, plot=False):
    """ Preprocess image for single capsplot prediction with model. """
    # load image
    img = load_img(filename)

    # normalise the image
    resize_norm_transform = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.Normalize(mean=[0.403,0.647,0.577], std=[0.301,0.183,0.312]),
                                  ])
    img_stdscaled = resize_norm_transform(img)

    if transform is not None:
        img_stdscaled = transform(img_stdscaled)
    
    if plot:
        plot_img_tensor(img)

    return img_stdscaled

def predict(filename, model, transform=None, plot=False):
    """ Predict single capsplot with cnn-caps model. """
    img = preprocess_capsplot(filename, transform=transform, plot=plot)
    result = model(torch.FloatTensor(img).to(device))[1].data.cpu().numpy()
    return result
# ---------------------------------------------------------------------------- #
#                                     Model                                    #
# ---------------------------------------------------------------------------- #
class Identity(nn.Module):
    # create layer that returns unchanged input

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class TransferredNet(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.pretrained_model.fc = Identity()

        self.head = nn.Sequential(
            nn.Linear(512, 16),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(16, N_CLASSES), # output logits
            #nn.Sigmoid()
        )
        
    def forward(self, input):
        logits = self.head(self.pretrained_model(input))
        probs = F.softmax(logits, dim=1)
        return logits, probs


if __name__ == '__main__':
    # USER INPUTS (CONSTANTS)
    N_CLASSES = 3
    CLASS_LABELS = ['0_notCrossing', '1_MP', '2_BS']
    PATH = "./models/"
    MODEL_NAME = "resnet18_custom_softmaxCrossEntropy_then_EDL_train_full.pt" #"resnet18_custom_softmaxCrossEntropy_train_full.pt" 
    RANDOM_SEED = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    MODEL_NAME = "resnet18_custom_softmaxCrossEntropy_train_full_multiclass.pt"
    model = load_model(path=PATH, model_filename=MODEL_NAME)
    
    # predict on image
    filename='./dataset/images/2007-03-16T130300_2007-03-16T150300.png'
    result = predict(filename, model, plot=True)
    print('Prediction', result)