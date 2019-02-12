"""
Functions to creating figures to inspect tedana output
"""
import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from tedana import model

LGR = logging.getLogger(__name__)


def write_comp_figs(ts, mask, comptable, mmix, n_vols,
                  acc, rej, midk, empty, ref_img):
    """
    Creates static figures that highlight certain aspects of tedana processing
    This includes a figure for each component showing the component time course,
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

    # Get repetition time from ref_img
    tr = ref_img.header.get_zooms()[-1]

    # Start making plots
    if not os.path.isdir('figures'):
        os.mkdir('figures')

    # Create indices for 6 cuts, based on dimensions
    cuts = [ts_B.shape[dim] // 6 for dim in range(3)]

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
            # Limit to 2 decimal places
            seconds_val = round(X * tr, 2)
            ax2Xs.append(seconds_val)
        ax_ts2.set_xticks(ax1Xs)
        ax_ts2.set_xbound(ax_ts.get_xbound())
        ax_ts2.set_xticklabels(ax2Xs)
        ax_ts2.set_xlabel('seconds')

        # Set range to ~1/10th of max beta
        imgmax = ts_B[:, :, :, compnum].max() * .1
        imgmin = ts_B[:, :, :, compnum].min() * .1

        for idx, cut in enumerate(cuts):
            for imgslice in range(1, 6):
                ax = plt.subplot2grid((5, 6), (1, imgslice - 1), rowspan=1, colspan=1)
                ax.axis('off')

                to_plot = ts_B[imgslice * cuts[0], :, :, compnum]
                if idx in [0, 1]: # only for first 2 dimensions
                    to_plot = np.rot90(to_plot) # rotate the plotted slices by 90-deg

                ax.imshow(to_plot, vmin=imgmin, vmax=imgmax, aspect='equal',
                          cmap='coolwarm')

        # Add a color bar to the plot.
        ax_cbar = allplot.add_axes([0.8, 0.3, 0.03, 0.37])
        cbar = allplot.colorbar(ax_im, ax_cbar)
        cbar.set_label('Component Beta', rotation=90)
        cbar.ax.yaxis.set_label_position('left')

        # Get fft and freqs for this subject
        # adapted from @dangom
        spectrum, freqs = get_spectrum(mmix[:, compnum], tr)

        # Plot it
        ax_fft = plt.subplot2grid((5, 6), (4, 0), rowspan=1, colspan=6)
        ax_fft.plot(freqs, spectrum)
        ax_fft.set_title('One Sided fft')
        ax_fft.set_xlabel('Hz')
        ax_fft.set_xbound(freqs[0], freqs[-1])

        # Fix spacing so TR label does overlap with other plots
        allplot.subplots_adjust(hspace=0.4)
        plot_name = 'comp_{}.png'.format(str(compnum).zfill(3))
        compplot_name = os.path.join('figures', plot_name)
        plt.savefig(compplot_name)
        plt.close()


def write_kappa_scatter(comptable):
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

    # Set up for varying marker shape and color
    mkr_dict = {'accepted': ['*', 'g'], 'rejected': ['v', 'r'], 'ignored': ['d', 'k'], 'midk': ['^', 'm']}

    # Prebuild legend so that the marker sizes are uniform
    for kind in mkr_dict:
            plt.scatter([], [], s=1, markermkr_dict[kind][0],
                        c=mkr_dict[kind][1], label='accepted', alpha=0.5)
    # Create legend
    ax_scatter.legend(markerscale=10)

    # Plot actual values
    for kind in mkr_dict:
        d = comptable[comptable.classification == kind]
        plt.scatter(d.kappa, d.rho,
                    s=150 * d['variance explained'], marker=mkr_dict[kind][0],
                    c=mkr_dict[kind][1], alpha=0.5)

    # Finish labeling the plot.
    ax_scatter.set_xlabel('kappa')
    ax_scatter.set_ylabel('rho')
    ax_scatter.set_title('Kappa vs Rho')
    ax_scatter.xaxis.label.set_fontsize(20)
    ax_scatter.yaxis.label.set_fontsize(20)
    ax_scatter.title.set_fontsize(25)
    scatter_title = os.path.join('figures', 'Kappa_vs_Rho_Scatter.png')
    plt.savefig(scatter_title)

    plt.close()


def write_summary_fig(comptable):
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

    # Get the variance and count of each classification
    var_expl = []
    counts = {}
    for clf in ['accepted', 'rejected', 'ignored']:
        var_expl,append(np.sum(comptable[comptable.classification == clf]['variance explained']))
        counts[clf] =  comptable[comptable.classification == clf].count()[0] + ' ' + clf

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.bar([1, 2, 3], var_expl, color=['g', 'r', 'k'])
    plt.xticks([1, 2, 3], counts.values(), fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylabel('Variance Explained', fontsize=20)
    plt.title('Component Overview', fontsize=25)
    sumfig_title = os.path.join('figures', 'Component_Overview.png')
    plt.savefig(sumfig_title)


def get_spectrum(data: np.array, tr: float = 1):
    """
    Returns the power spectrum and corresponding frequencies when provided
    with a component time course and repitition time.

    Parameters
    ----------
    data : (S, ) array_like
            A timeseries S, on which you would like to perform an fft.
    tr : :obj:`float`
            Reptition time (TR) of the data
    """

    # adapted from @dangom
    power_spectrum = np.abs(np.fft.rfft(data)) ** 2
    freqs = np.fft.rfftfreq(power_spectrum.size * 2 - 1, tr)
    idx = np.argsort(freqs)
    return power_spectrum[idx], freqs[idx]
