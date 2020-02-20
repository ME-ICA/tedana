"""
Functions to creating figures to inspect tedana output
"""
import logging
import os

import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from tedana import stats
from tedana.utils import get_spectrum

LGR = logging.getLogger(__name__)
MPL_LGR = logging.getLogger('matplotlib')
MPL_LGR.setLevel(logging.WARNING)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def trim_edge_zeros(arr):
    """
    Trims away the zero-filled slices that surround many 3/4D arrays

    Parameters
    ----------
    ndarray: (S x T) array_like
        an array with signal, surrounded by slices that contain only zeros
        that should be removed.

    Returns
    ---------
    ndarray: (S x T) array_like
        an array with reduced dimensions, such that the array contains only
        non_zero values from edge to edge.
    """

    mask = arr != 0
    bounding_box = tuple(
                         slice(np.min(indexes), np.max(indexes) + 1)
                         for indexes in np.where(mask))
    return arr[bounding_box]


def write_comp_figs(ts, mask, comptable, mmix, ref_img, out_dir,
                    png_cmap):
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
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    out_dir : :obj:`str`
        Figures folder within output directory
    png_cmap : :obj:`str`
        The name of a matplotlib colormap to use when making figures. Optional.
        Default colormap is 'coolwarm'

    """
    # Get the lenght of the timeseries
    n_vols = len(mmix)

    # Check that colormap provided exists
    if png_cmap not in plt.colormaps():
        LGR.warning('Provided colormap is not recognized, proceeding with default')
        png_cmap = 'coolwarm'
    # regenerate the beta images
    ts_B = stats.get_coeffs(ts, mmix, mask)
    ts_B = ts_B.reshape(ref_img.shape[:3] + ts_B.shape[1:])
    # trim edges from ts_B array
    ts_B = trim_edge_zeros(ts_B)

    # Mask out remaining zeros
    ts_B = np.ma.masked_where(ts_B == 0, ts_B)

    # Get repetition time from ref_img
    tr = ref_img.header.get_zooms()[-1]

    # Create indices for 6 cuts, based on dimensions
    cuts = [ts_B.shape[dim] // 6 for dim in range(3)]
    expl_text = ''

    # Remove trailing ';' from rationale column
    comptable['rationale'] = comptable['rationale'].str.rstrip(';')
    for compnum in comptable.index.values:
        if comptable.loc[compnum, "classification"] == 'accepted':
            line_color = 'g'
            expl_text = 'accepted'
        elif comptable.loc[compnum, "classification"] == 'rejected':
            line_color = 'r'
            expl_text = 'rejection reason(s): ' + comptable.loc[compnum, "rationale"]
        elif comptable.loc[compnum, "classification"] == 'ignored':
            line_color = 'k'
            expl_text = 'ignored reason(s): ' + comptable.loc[compnum, "rationale"]
        else:
            # Classification not added
            # If new, this will keep code running
            line_color = '0.75'
            expl_text = 'other classification'

        allplot = plt.figure(figsize=(10, 9))
        ax_ts = plt.subplot2grid((5, 6), (0, 0),
                                 rowspan=1, colspan=6,
                                 fig=allplot)

        ax_ts.set_xlabel('TRs')
        ax_ts.set_xlim(0, n_vols)
        plt.yticks([])
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
        ax_ts2.set_xlim(ax_ts.get_xbound())
        ax_ts2.set_xticklabels(ax2Xs)
        ax_ts2.set_xlabel('seconds')

        ax_ts.plot(mmix[:, compnum], color=line_color)

        # Title will include variance from comptable
        comp_var = "{0:.2f}".format(comptable.loc[compnum, "variance explained"])
        comp_kappa = "{0:.2f}".format(comptable.loc[compnum, "kappa"])
        comp_rho = "{0:.2f}".format(comptable.loc[compnum, "rho"])
        plt_title = ('Comp. {}: variance: {}%, kappa: {}, rho: {}, '
                     '{}'.format(compnum, comp_var, comp_kappa, comp_rho,
                                 expl_text))
        title = ax_ts.set_title(plt_title)
        title.set_y(1.5)

        # Set range to ~1/10th of max positive or negative beta
        imgmax = 0.1 * np.abs(ts_B[:, :, :, compnum]).max()
        imgmin = imgmax * -1

        for idx, cut in enumerate(cuts):
            for imgslice in range(1, 6):
                ax = plt.subplot2grid((5, 6), (idx + 1, imgslice - 1), rowspan=1, colspan=1)
                ax.axis('off')

                if idx == 0:
                    to_plot = np.rot90(ts_B[imgslice * cuts[idx], :, :, compnum])
                if idx == 1:
                    to_plot = np.rot90(ts_B[:, imgslice * cuts[idx], :, compnum])
                if idx == 2:
                    to_plot = ts_B[:, :, imgslice * cuts[idx], compnum]

                ax_im = ax.imshow(to_plot, vmin=imgmin, vmax=imgmax, aspect='equal',
                                  cmap=png_cmap)

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
        ax_fft.set_xlim(freqs[0], freqs[-1])
        plt.yticks([])

        # Fix spacing so TR label does overlap with other plots
        allplot.subplots_adjust(hspace=0.4)
        plot_name = 'comp_{}.png'.format(str(compnum).zfill(3))
        compplot_name = os.path.join(out_dir, plot_name)
        plt.savefig(compplot_name)
        plt.close()


def write_kappa_scatter(comptable, out_dir):
    """
    Creates a scatter plot of Kappa vs Rho values. The shape and size of the
    points is based on classification and variance explained, respectively.

    Parameters
    ----------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. Requires at least four columns: "classification",
        "kappa", "rho", and "variance explained".
    out_dir : :obj:`str`
        Figures folder within output directory

    """

    # Creating Kappa Vs Rho plot
    ax_scatter = plt.gca()

    # Set up for varying marker shape and color
    mkr_dict = {'accepted': ['*', 'g'], 'rejected': ['v', 'r'],
                'ignored': ['d', 'k']}

    # Prebuild legend so that the marker sizes are uniform
    for kind in mkr_dict:
        plt.scatter([], [], s=1, marker=mkr_dict[kind][0],
                    c=mkr_dict[kind][1], label=kind, alpha=0.5)
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
    scatter_title = os.path.join(out_dir, 'Kappa_vs_Rho_Scatter.png')
    plt.savefig(scatter_title)

    plt.close()


def write_kappa_scree(comptable, out_dir):
    """
    Creates a scree plot sorted by kappa, showing the values of the kappa and
    rho metrics as well as the variance explained.

    Parameters
    ----------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. Requires at least four columns: "classification",
        "kappa", "rho", and "variance explained".
    out_dir : :obj:`str`
        Figures folder within output directory

    """

    fig, ax1 = plt.subplots(figsize=(10, 9))

    ax1.plot(comptable.index, comptable['variance explained'],
             'k-', alpha=0.5, linewidth=2, label='Variance')
    ax1.set_ylabel('Variance Explained', fontsize=15)
    ax2 = ax1.twinx()

    ax2.plot(comptable.index, comptable.kappa,
             'b-', linewidth=2, label='Kappa')
    ax2.plot(comptable.index, comptable.rho,
             'r-', linewidth=2, label='Rho')
    ax2.set_title('Kappa/Rho Metrics', fontsize=28)
    ax1.set_xlabel('Component Number', fontsize=15)
    ax2.set_ylabel('Metric Value', fontsize=15)
    fig.legend(loc='upper right', bbox_to_anchor=(0.82, 0.78))
    screefig_title = os.path.join(out_dir, 'Kappa_Rho_Scree_plot.png')
    fig.savefig(screefig_title)


def write_summary_fig(comptable, out_dir):
    """
    Creates a pie chart showing 1) The total variance explained by each
    component in the outer ring, 2) the variance explained by each
    individual component in the inner ring, 3) counts of each classification
    and 4) the amount of unexplained variance.

    Parameters
    ----------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. Requires at least two columns: "variance explained" and
        "classification".
    out_dir : :obj:`str`
        Figures folder within output directory
    """

    var_expl = []
    ind_var_expl = {}
    counts = {}
    # Get overall variance explained, each components variance and counts of comps
    for clf in ['accepted', 'rejected', 'ignored']:
        var_expl.append(np.sum(comptable[comptable.classification == clf]['variance explained']))
        ind_var_expl[clf] = comptable[comptable.classification == clf]['variance explained'].values
        counts[clf] = '{0} {1}'.format(comptable[comptable.classification == clf].count()[0], clf)

    # Generate Colormaps for individual components
    acc_colors = plt.cm.Greens(np.linspace(0.2, .6, len(ind_var_expl['accepted'].tolist())))
    rej_colors = plt.cm.Reds(np.linspace(0.2, .6, len(ind_var_expl['rejected'].tolist())))
    ign_colors = plt.cm.Greys(np.linspace(0.2, .8, len(ind_var_expl['ignored'].tolist())))
    unxp_colors = np.atleast_2d(np.array(plt.cm.Greys(0)))

    # Shuffle the colors so that neighboring wedges are (perhaps) visually seperable
    np.random.shuffle(rej_colors)
    np.random.shuffle(acc_colors)
    np.random.shuffle(ign_colors)

    # Decision on whether to include the unexplained variance in figure
    unexpl_var = [100 - np.sum(var_expl)]
    all_var_expl = []
    if unexpl_var >= [0.001]:
        var_expl += unexpl_var
        counts['unexplained'] = 'unexplained variance'
        # Combine individual variances from giant list
        for value in ind_var_expl.values():
            all_var_expl += value.tolist()
        # Add in unexplained variance
        all_var_expl += unexpl_var
        outer_colors = np.stack((plt.cm.Greens(0.7), plt.cm.Reds(0.7),
                                 plt.cm.Greys(0.7), plt.cm.Greys(0)))
        inner_colors = np.concatenate((acc_colors, rej_colors, ign_colors, unxp_colors), axis=0)
    else:
        for value in ind_var_expl.values():
            all_var_expl += value.tolist()
        outer_colors = np.stack((plt.cm.Greens(0.7), plt.cm.Reds(0.7), plt.cm.Greys(0.7)))
        inner_colors = np.concatenate((acc_colors, rej_colors, ign_colors), axis=0)

    labels = counts.values()

    fig, ax = plt.subplots(figsize=(16, 10))
    size = 0.3
    # Build outer, overall pie chart, and then inner individual comp pie
    ax.pie(var_expl, radius=1, colors=outer_colors, labels=labels,
           autopct='%1.1f%%', pctdistance=0.85, textprops={'fontsize': 20},
           wedgeprops=dict(width=size, edgecolor='w'))

    ax.pie(all_var_expl, radius=1 - size, colors=inner_colors,
           wedgeprops=dict(width=size))

    ax.set(aspect="equal")
    ax.set_title('Variance Explained By Classification', fontdict={'fontsize': 28})
    if unexpl_var < [0.001]:
        plt.text(1, -1, '*Unexplained Variance less than 0.001', fontdict={'fontsize': 12})
    sumfig_title = os.path.join(out_dir, 'Component_Overview.png')
    plt.savefig(sumfig_title)
