import numpy as np


def ctabsel(ctabfile):
    """
    Loads a pre-existing component table file

    Parameters
    ----------
    ctabfile : str
        Filepath to existing component table

    Returns
    -------
    ctab : (4,) tuple-of-arrays
        Tuple containing arrays of (1) accepted, (2) rejected, (3) mid, and (4)
        ignored components
    """

    with open(ctabfile, 'r') as src:
        ctlines = src.readlines()
    class_tags = ['#ACC', '#REJ', '#MID', '#IGN']
    class_dict = {}
    for ii, ll in enumerate(ctlines):
        for kk in class_tags:
            if ll[:4] is kk and ll[4:].strip() is not '':
                class_dict[kk] = ll[4:].split('#')[0].split(',')
    return tuple([np.array(class_dict[kk], dtype=int) for kk in class_tags])
