#!/usr/bin/env python

# These are the python implementations of the denoising algorithms based on the Marcenko-Pasteur theorem. They are taken from the book
# Machine Learning for Asset Managers by Marcos M Lopez de Prado

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity

def mpPDF(var, q, pts):
    """
    This implements Marcenko-Pasteur probability density function, the parameter q= T/N
    """
    eMin, eMax = var * (1 - np.sqrt(1 / q)) ** 2, var * (1 + np.sqrt(1 / q)) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * np.sqrt((eMax - eVal) * (eVal - eMin))
    pdf = pd.Series(pdf, index=eVal)

    return pdf


def getPCA(matrix):

    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]

    eVal = np.diagflat(eVal)

    return eVal, eVec


def fitKDE(obs, bWidth=0.25, kernel="gaussian", x=None):
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)

    if x is None:
        x = np.unique(obs).reshape(-1, 1)

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    logProb = kde.score_samples(x)

    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf


x = np.random.normal(size=(10000, 1000))

eVal0, eVec0 = getPCA(np.corrcoef(x, rowvar=0))

pdf0 = mpPDF(1.0, q=x.shape[0] / x.shape[1], pts=1000)
pdf1 = fitKDE(np.diag(eVal0), bWidth=0.01)


#fig = plt.figure(figsize=(16, 8))
#ax1 = fig.add_subplot(111)

#pdf0.plot(ax=ax1)
#pdf1.plot(ax=ax1)

#plt.show(block=True)
