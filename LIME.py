# -*- coding: utf-8 -*-
"""
GrandCAM implementation

original paper: D. Garreau et al. “What does LIME really see in images?” In: ICML. 2021
"""
import torch
from torch import nn
import numpy as np
from skimage import segmentation
from sklearn.linear_model import Ridge
from numpy.linalg import norm
from torch.nn.functional import softmax 

class Lime():
    ""

    def __init__(self, img, n_class, n_superpixels = 50, bandwidth_parameter=0.25):
        """
        Inputs
            img:                    256 x 256 x 3 tensor (or array?)
            n_class:                Which class to evaluate
            n_superpixels:          Number of superpixels, assumed 50(?)
            bandwidth_parameter:    float, as specified on the paper

        
        """
        self.img = img
        self.img_dimensions = img.shape
        self.n_class = n_class
        self.n_superpixels = n_superpixels
        # self.activated_sup_pixels = np.ones(self.n_superpixels)
        self.bandwidth_parameter = bandwidth_parameter
        
    def superpixel_seg(self):
        """
        Segments the image into superpixels and stores their averages.

        Outputs:
            self.mapping:           dict {superpixel_id: [list of pixel indices]}
            self.superpixels_avg:   list of mean RGB values per superpixel
            self.labels:            segmentation map (H x W)
        """

        self.labels = skimage.segmentation(self.img, n_segments=self.n_superpixels, compactness=10, start_label=0)
        n_sp = len(np.unique(self.labels))

        self.mapping = {}
        self.superpixels_avg = []

        for sp_id in range(n_sp):
            mask = (self.labels == sp_id)
            coords = np.argwhere(mask)
            self.mapping[sp_id] = coords
            self.superpixels_avg.append(self.img[mask].mean(axis=0))




    def _cal_weight(self, activation_vector):
        """
        Returns the similarity between the original image and 
        the sampled one as a coefficient. The calculation is
        based only on the number of inactivated superpixels.

        Inputs
            activation_vector : np.array of shape n_superpixels

        Outputs
            weight : float
        """
        ones = np.ones_like(activation_vector)
        cos_sim = np.dot(activation_vector, ones) / (np.linalg.norm(activation_vector) * np.linalg.norm(ones))
        d_cos = 1 - cos_sim
        weight = np.exp(- (d_cos ** 2) / (2 * self.bandwidth_parameter ** 2))
        return weight
    
    def generate_img(self):
        """
        Generates a new image based on the superpixels that
        are activated

        inputs

        outputs
            out_img :   array of the same size as the image
        """
        out_img = np.zeros(self.img_dimensions)
        random_activation = np.random.radint(2, self.n_superpixels)

        for i, pixels in self.mapping.items():
            for (y, x) in pixels:
                if random_activation[i]:
                    out_img[y, x, :] = self.img[y, x, :]
                else:
                    out_img[y, x, :] = self.superpixels_avg[i]
        return out_img, random_activation
    




    def sample_images(self, n_samples):
        """
        Returns the sampled images, alongside each image's 
        weight and random activation

        inputs
            n_samples:     int, number of sampled images

        outputs
            imgs:          list of images as np.arrays
            weights:       list with the weight of each image
            activations:   list with the vectors of activations of each image
        """
        imgs = []
        activations = []
        weights = []
        for _ in range(n_samples):
            new_img, act = self.generate_img()
            imgs.append(new_img)
            activations.append(act)
            weights.append(self._cal_weight(act))
        return imgs, weights, activations

    def fit_surrogate_model(self, alexnet28, n_samples=1000, alpha=1):
        """
        Fits the black-box model to a linear model with a Ridge regularization.

        Inputs
            alexnet28:  Trained alexnet28 neural network object
            n_samples:  int
            alpha:      ridge regularization coefficient (assumed 1)

        Outputs:
            beta:   Array with calculated weights of the linear model
        """
        imgs, weights, activations = self.sample_images(n_samples)
        preds = []

        for im in imgs:
            im_t = torch.tensor(im.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                y = alexnet28(im_t)[0, self.n_class].item()
            preds.append(y)
        
        activations_arr = np.stack(activations, axis=0)
        preds_arr = np.array(preds)
        
        reg = Ridge(alpha=1.0)
        reg.fit(activations_arr, preds_arr, sample_weight=weights)
        self.beta = reg.coef_
        
        return self.beta

    def explain(self, n_superpixels=5):
        """
        Displays the image generated by the most relevant superpixels.

        Inputs
            n_superpixels:  int

        Outputs
            explanation:    an array with the image generated by the
                            most relevant superpixels

        """
        idx = np.argsort(self.beta)[-n_superpixels:]
        mask = np.isin(self.labels, idx)
        explanation = np.zeros_like(self.img)
        explanation[mask] = self.img[mask]
        return explanation