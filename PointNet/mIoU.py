import numpy as np


def comfusion(prediction, annotation, n):
    # Inputs:
    #     prediction - The prediction from model with shape (n, )
    #     annotation - The annotation of data with shape (n, )
    #     n - The class number
    # Outputs:
    #     cMatrix - The mIoU standard for the pair of data
    cMatrix = np.bincount(n * annotation + prediction, minlength=n*n).reshape(n, n)
    return cMatrix

def computeMIOU(cMatrix):
    # Inputs:
    #     cMatrix - The confusion matrix with shape (n, n)
    # Outputs:
    #     mIOU - The mIoU standard of current confusion matrix
    iouPerClass = np.diag(cMatrix) / (cMatrix.sum(1) + cMatrix.sum(0) - np.diag(cMatrix))
    mIOU = np.mean(iouPerClass)
    return mIOU
    
    