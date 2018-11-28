"""
Utility functions for tedana.selection
"""
import logging

import numpy as np
from sklearn import svm

LGR = logging.getLogger(__name__)


def do_svm(X_train, y_train, X_test, svmtype=0):
    """
    Implements Support Vector Classification on provided data

    Parameters
    ----------
    X_train : (N1 x F) array_like
        Training vectors, where n_samples is the number of samples in the
        training dataset and n_features is the number of features.
    y_train : (N1,) array_like
        Target values (class labels in classification, real numbers in
        regression)
    X_test : (N2 x F) array_like
        Test vectors, where n_samples is the number of samples in the test
        dataset and n_features is the number of features.
    svmtype : :obj:`int`, optional
        Desired support vector machine type. Must be in [0, 1, 2]. Default: 0

    Returns
    -------
    y_pred : (N2,) :obj:`numpy.ndarray`
        Predicted class labels for samples in `X_test`
    clf : {:obj:`sklearn.svm.SVC`, :obj:`sklearn.svm.LinearSVC`}
        Trained sklearn model instance
    """

    if svmtype == 0:
        clf = svm.SVC(kernel='linear')
    elif svmtype == 1:
        clf = svm.LinearSVC(loss='squared_hinge', penalty='l1', dual=False)
    elif svmtype == 2:
        clf = svm.SVC(kernel='linear', probability=True)
    else:
        raise ValueError('Input svmtype not in [0, 1, 2]: {}'.format(svmtype))

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred, clf


def getelbow_cons(arr, return_val=False):
    """
    Elbow using mean/variance method - conservative

    Parameters
    ----------
    arr : (C,) array_like
        Metric (e.g., Kappa or Rho) values.
    return_val : :obj:`bool`, optional
        Return the value of the elbow instead of the index. Default: False

    Returns
    -------
    :obj:`int` or :obj:`float`
        Either the elbow index (if return_val is True) or the values at the
        elbow index (if return_val is False)
    """
    if arr.ndim != 1:
        raise ValueError('Parameter arr should be 1d, not {0}d'.format(arr.ndim))
    arr = np.sort(arr)[::-1]
    nk = len(arr)
    temp1 = [(arr[nk - 5 - ii - 1] > arr[nk - 5 - ii:nk].mean() + 2 * arr[nk - 5 - ii:nk].std())
             for ii in range(nk - 5)]
    ds = np.array(temp1[::-1], dtype=np.int)
    dsum = []
    c_ = 0
    for d_ in ds:
        c_ = (c_ + d_) * d_
        dsum.append(c_)
    e2 = np.argmax(np.array(dsum))
    elind = np.max([getelbow(arr), e2])

    if return_val:
        return arr[elind]
    else:
        return elind


def getelbow(arr, return_val=False):
    """
    Elbow using linear projection method - moderate

    Parameters
    ----------
    arr : (C,) array_like
        Metric (e.g., Kappa or Rho) values.
    return_val : :obj:`bool`, optional
        Return the value of the elbow instead of the index. Default: False

    Returns
    -------
    :obj:`int` or :obj:`float`
        Either the elbow index (if return_val is True) or the values at the
        elbow index (if return_val is False)
    """
    if arr.ndim != 1:
        raise ValueError('Parameter arr should be 1d, not {0}d'.format(arr.ndim))
    arr = np.sort(arr)[::-1]
    n_components = arr.shape[0]
    coords = np.array([np.arange(n_components), arr])
    p = coords - coords[:, 0].reshape(2, 1)
    b = p[:, -1]
    b_hat = np.reshape(b / np.sqrt((b ** 2).sum()), (2, 1))
    proj_p_b = p - np.dot(b_hat.T, p) * np.tile(b_hat, (1, n_components))
    d = np.sqrt((proj_p_b ** 2).sum(axis=0))
    k_min_ind = d.argmax()

    if return_val:
        return arr[k_min_ind]
    else:
        return k_min_ind
