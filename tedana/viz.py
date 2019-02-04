"""
Functions to creating figures to inspect tedana output
"""
import logging
import os
import os.path as op

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from nibabel.filename_parser import splitext_addext

from tedana import model, utils

LGR = logging.getLogger(__name__)

def writecompfigs(ts, mask, comptable, mmix, n_vols,
                 acc, rej, midk, empty, ref_img):
    """
    Creates static figures that highlight certain aspects of tedana processing
    This includes figure for each component showing the component time course,
    the spatial weight map and a fast Fourier transform of the time course

    Parameters
    ----------
    ts : (S x T) array_like
        Time series from which to derive ICA betas
    mask : (S,) array_like
        Boolean mask array
    comptable : (N x 5) array_like
        Array with columns denoting (1) index of component, (2) Kappa score of
        component, (3) Rho score of component, (4) variance explained by
        component, and (5) normalized variance explained by component
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    n_vols : :obj:`int`
        Number of volumes in original time series
    acc : :obj:`list`
        Indices of accepted (BOLD) components in `mmix`
    rej : :obj:`list`
        Indices of rejected (non-BOLD) components in `mmix`
    midk : :obj:`list`
        Indices of mid-K (questionable) components in `mmix`
    empty : :obj:`list`
        Indices of ignored components in `mmix`
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk

    """

    # regenerate the beta images
    ts_B = model.get_coeffs(ts, mmix, mask)
    ts_B = ts_B.reshape(ref_img.shape[:3] + ts_B.shape[1:])
    # Mask out zeros
    ts_B = np.ma.masked_where(ts_B == 0, ts_B)

    # Get repitition time from ref_img
    tr = ref_img.header.get_zooms()[-1]

    # Start making plots
    if not os.path.exists('figures'):
        os.mkdir('figures')

    # This precalculates the Hz for the fft plots
    Fs = 1.0/tr
    # resampled frequency vector
    f = Fs * np.arange(0, n_vols // 2 + 1) / n_vols

    # Create indices for 6 cuts, based on dimensions
    cuts =[ts_B.shape[dim] // 6 for dim in range(3)]

    for compnum in range(0, mmix.shape[1], 1):

        allplot = plt.figure(figsize=(10, 9))
        ax_ts = plt.subplot2grid((5, 6), (0, 0), rowspan=1, colspan=6,
                                 fig=allplot)
        if compnum in acc:
            line_color = 'g'
        elif compnum in rej:
            line_color = 'r'
        elif compnum in midk:
            line_color = 'm'
        else:
            line_color = 'k'

        ax_ts.plot(mmix[:, compnum], color=line_color)

        # Title will include variance from comptable
        comp_var = "{0:.2f}".format(comptable.iloc[compnum][3])
        plt_title = 'Comp. {}: {}% variance'.format(compnum, comp_var)
        title = ax_ts.set_title(plt_title)
        title.set_y(1.5)
        ax_ts.set_xlabel('TRs')
        ax_ts.set_xbound(0, n_vols)
        # Make a second axis with units of time (s)
        max_xticks = 10
        xloc = plt.MaxNLocator(max_xticks)
        ax_ts.xaxis.set_major_locator(xloc)

        ax_ts2 = ax_ts.twiny()
        ax1Xs = ax_ts.get_xticks()

        ax2Xs = []
        for X in ax1Xs:
            ax2Xs.append(X * tr)
        ax_ts2.set_xticks(ax1Xs)
        ax_ts2.set_xbound(ax_ts.get_xbound())
        ax_ts2.set_xticklabels(ax2Xs)
        ax_ts2.set_xlabel('seconds')

        # Set range to ~1/10th of max beta
        imgmax = ts_B[:, :, :, compnum].max() * .1
        imgmin = ts_B[:, :, :, compnum].min() * .1

        for imgslice in range(1, 6, 1):
            # First row
            ax = plt.subplot2grid((5, 6), (1, imgslice - 1), rowspan=1, colspan=1)
            ax.imshow(np.rot90(ts_B[imgslice * cuts[0], :, :, compnum], k=1),
                        vmin=imgmin, vmax=imgmax, aspect='equal',
                        cmap='coolwarm')

            # Second row
            ax = plt.subplot2grid((5, 6), (2, imgslice - 1), rowspan=1, colspan=1)
            ax.imshow(np.rot90(ts_B[:, imgslice * cuts[1], :, compnum], k=1),
                        vmin=imgmin, vmax=imgmax, aspect='equal',
                        cmap='coolwarm')

            # Third Row
            ax = plt.subplot2grid((5, 6), (3, imgslice - 1), rowspan=1, colspan=1)
            ax_im = ax_z.imshow(ts_B[:, :, imgslice * cuts[2], compnum],
                                  vmin=imgmin, vmax=imgmax, aspect='equal',
                                  cmap='coolwarm')
            ax.axis('off')

        # Add a color bar to the plot.
        ax_cbar = allplot.add_axes([0.8, 0.3, 0.03, 0.37])
        cbar = allplot.colorbar(ax_im, ax_cbar)
        cbar.set_label('Component Beta', rotation=90)
        cbar.ax.yaxis.set_label_position('left')
        # Get fft for this subject, change to one sided amplitude
        # adapted from
        # https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
        y = mmix[:, compnum]
        Y = scipy.fftpack.fft(y)
        P2 = np.abs(Y/n_vols)
        P1 = P2[0:n_vols // 2 + 1]
        P1[1:-2] = 2 * P1[1:-2]

        # Plot it
        ax_fft = plt.subplot2grid((5, 6), (4, 0), rowspan=1, colspan=6)
        ax_fft.plot(f, P1)
        ax_fft.set_title('One Sided fft')
        ax_fft.set_xlabel('Hz')
        ax_fft.set_xbound(f[0], f[-1])

        # Fix spacing so TR label isn't overlapped
        allplot.subplots_adjust(hspace=0.4)
        plot_name = 'comp_{}.png'.format(str(compnum).zfill(3))
        compplot_name = os.path.join('figures', plot_name)
        plt.savefig(fname)
        plt.close()




def writekappascatter(ts, mask, comptable, mmix, n_vols,
                 acc, rej, midk, empty, ref_img):
    """
    Creates a scatter plot of Kappa vs Rho values. The shape and size of the
    points is based on classification and variance explained, respectively.

    Parameters
    ----------
    comptable : (N x 5) array_like
        Array with columns denoting (1) index of component, (2) Kappa score of
        component, (3) Rho score of component, (4) variance explained by
        component, and (5) normalized variance explained by component

    """

    # Creating Kappa Vs Rho plot
    ax_scatter = plt.gca()

    # Prebuild legend so that the marker sizes are uniform
    plt.scatter([], [], s=1, marker='*', c='g', label='accepted', alpha=0.5)
    plt.scatter([], [], s=1, marker='v', c='r', label='rejected', alpha=0.5)
    plt.scatter([], [], s=1, marker='d', c='k', label='ignored', alpha=0.5)
    plt.scatter([], [], s=1, marker='^', c='m', label='midk', alpha=0.5)
    ax_scatter.legend(markerscale=10)

    mkr_dict = {'accepted': '*', 'rejected': 'v', 'ignored': 'd', 'midk': '^'}
    col_dict = {'accepted': 'g', 'rejected': 'r', 'ignored': 'k', 'midk': 'm'}
    for kind in mkr_dict:
        d = comptable[comptable.classification == kind]
        plt.scatter(d.kappa, d.rho,
                    s=150 * d['variance explained'], marker=mkr_dict[kind],
                    c=col_dict[kind], alpha=0.5)

    ax_scatter.set_xlabel('kappa')
    ax_scatter.set_ylabel('rho')
    ax_scatter.set_title('Kappa vs Rho')
    ax_scatter.xaxis.label.set_fontsize(20)
    ax_scatter.yaxis.label.set_fontsize(20)
    ax_scatter.title.set_fontsize(25)
    scatter_title = os.path.join('figures', 'Kappa_vs_Rho_Scatter.png')
    plt.savefig(scatter_title)

    plt.close()


def writesummaryfig(comptable):
    """
    Creates a bar graph showing total variance explained by each component
    as well as the number of components identified for each category.

    Parameters
    ----------
    comptable : (N x 5) array_like
        Array with columns denoting (1) index of component, (2) Kappa score of
        component, (3) Rho score of component, (4) variance explained by
        component, and (5) normalized variance explained by component
    """

    var_acc = np.sum(comptable[comptable.classification == 'accepted']['variance explained'])
    var_rej = np.sum(comptable[comptable.classification == 'rejected']['variance explained'])
    var_ign = np.sum(comptable[comptable.classification == 'ignored']['variance explained'])
    count_acc = np.count_nonzero(comptable[comptable.classification == 'accepted']['classification'])
    count_rej = np.count_nonzero(comptable[comptable.classification == 'rejected']['classification'])
    count_ign = np.count_nonzero(comptable[comptable.classification == 'ignored']['classification'])

    fig, ax = plt.subplots(figsize=(10,7))
    acc_label = str(count_acc) + ' Accepted'
    rej_label = str(count_rej) + ' Rejected'
    ign_label = str(count_ign) + ' Ignored'
    print(acc_label)
    plt.bar([1,2,3], [var_acc, var_rej, var_ign], color=['g', 'r', 'k'])
    plt.xticks([1, 2, 3], (acc_label, rej_label, ign_label))
    plt.ylabel('Variance Explained')
    plt.title('Component Overview')
    sumfig_title = os.path.join('figures', 'Component_Overview.png')
    plt.savefig(sumfig_title)
