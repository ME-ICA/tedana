"""
Functions to creating figures to inspect tedana output
"""
import logging
import os

import matplotlib
import numpy as np

matplotlib.use("AGG")
import matplotlib.pyplot as plt
from nilearn import plotting

from tedana import io, stats, utils

LGR = logging.getLogger("GENERAL")
MPL_LGR = logging.getLogger("matplotlib")
MPL_LGR.setLevel(logging.WARNING)
RepLGR = logging.getLogger("REPORT")


def _trim_edge_zeros(arr):
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
    bounding_box = tuple(slice(np.min(indexes), np.max(indexes) + 1) for indexes in np.where(mask))
    return arr[bounding_box]


def carpet_plot(optcom_ts, denoised_ts, hikts, lowkts, mask, io_generator, gscontrol=None):
    """Generate a set of carpet plots for the combined and denoised data.

    Parameters
    ----------
    optcom_ts, denoised_ts, hikts, lowkts : (S x T) array_like
        Different types of data to plot.
    mask : (S,) array-like
        Binary mask used to apply to the data.
    io_generator : :obj:`tedana.io.OutputGenerator`
        The output generator for this workflow
    gscontrol : {None, 'mir', 'gsr'} or :obj:`list`, optional
        Additional denoising steps applied in the workflow.
        If any gscontrol methods were applied, then additional carpet plots will be generated for
        pertinent outputs from those steps.
        Default is None.
    """
    mask_img = io.new_nii_like(io_generator.reference_img, mask.astype(int))
    optcom_img = io.new_nii_like(io_generator.reference_img, optcom_ts)
    dn_img = io.new_nii_like(io_generator.reference_img, denoised_ts)
    hik_img = io.new_nii_like(io_generator.reference_img, hikts)
    lowk_img = io.new_nii_like(io_generator.reference_img, lowkts)

    # Carpet plots
    fig, ax = plt.subplots(figsize=(14, 7))
    plotting.plot_carpet(
        optcom_img,
        mask_img,
        figure=fig,
        axes=ax,
        title="Optimally Combined Data",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(io_generator.out_dir, "figures", "carpet_optcom.svg"))

    fig, ax = plt.subplots(figsize=(14, 7))
    plotting.plot_carpet(
        dn_img,
        mask_img,
        figure=fig,
        axes=ax,
        title="Denoised Data",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(io_generator.out_dir, "figures", "carpet_denoised.svg"))

    fig, ax = plt.subplots(figsize=(14, 7))
    plotting.plot_carpet(
        hik_img,
        mask_img,
        figure=fig,
        axes=ax,
        title="High-Kappa Data",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(io_generator.out_dir, "figures", "carpet_accepted.svg"))

    fig, ax = plt.subplots(figsize=(14, 7))
    plotting.plot_carpet(
        lowk_img,
        mask_img,
        figure=fig,
        axes=ax,
        title="Low-Kappa Data",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(io_generator.out_dir, "figures", "carpet_rejected.svg"))

    if (gscontrol is not None) and ("gsr" in gscontrol):
        optcom_with_gs_img = io_generator.get_name("has gs combined img")
        fig, ax = plt.subplots(figsize=(14, 7))
        plotting.plot_carpet(
            optcom_with_gs_img,
            mask_img,
            figure=fig,
            axes=ax,
            title="Optimally Combined Data (Pre-GSR)",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(io_generator.out_dir, "figures", "carpet_optcom_nogsr.svg"))

    if (gscontrol is not None) and ("mir" in gscontrol):
        mir_denoised_img = io_generator.get_name("mir denoised img")
        fig, ax = plt.subplots(figsize=(14, 7))
        plotting.plot_carpet(
            mir_denoised_img,
            mask_img,
            figure=fig,
            axes=ax,
            title="Denoised Data (Post-MIR)",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(io_generator.out_dir, "figures", "carpet_denoised_mir.svg"))

        mir_denoised_img = io_generator.get_name("ICA accepted mir denoised img")
        fig, ax = plt.subplots(figsize=(14, 7))
        plotting.plot_carpet(
            mir_denoised_img,
            mask_img,
            figure=fig,
            axes=ax,
            title="High-Kappa Data (Post-MIR)",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(io_generator.out_dir, "figures", "carpet_accepted_mir.svg"))


def comp_figures(ts, mask, comptable, mmix, io_generator, png_cmap):
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
    io_generator : :obj:`tedana.io.OutputGenerator`
        Output Generator object to use for this workflow
    """
    # Get the lenght of the timeseries
    n_vols = len(mmix)

    # Flip signs of mixing matrix as needed
    mmix = mmix * comptable["optimal sign"].values

    # regenerate the beta images
    ts_B = stats.get_coeffs(ts, mmix, mask)
    ts_B = ts_B.reshape(io_generator.reference_img.shape[:3] + ts_B.shape[1:])
    # trim edges from ts_B array
    ts_B = _trim_edge_zeros(ts_B)

    # Mask out remaining zeros
    ts_B = np.ma.masked_where(ts_B == 0, ts_B)

    # Get repetition time from reference image
    tr = io_generator.reference_img.header.get_zooms()[-1]

    # Create indices for 6 cuts, based on dimensions
    cuts = [ts_B.shape[dim] // 6 for dim in range(3)]
    expl_text = ""

    # Remove trailing ';' from rationale column
    comptable["rationale"] = comptable["rationale"].str.rstrip(";")
    for compnum in comptable.index.values:
        if comptable.loc[compnum, "classification"] == "accepted":
            line_color = "g"
            expl_text = "accepted"
        elif comptable.loc[compnum, "classification"] == "rejected":
            line_color = "r"
            expl_text = "rejection reason(s): " + comptable.loc[compnum, "rationale"]
        elif comptable.loc[compnum, "classification"] == "ignored":
            line_color = "k"
            expl_text = "ignored reason(s): " + comptable.loc[compnum, "rationale"]
        else:
            # Classification not added
            # If new, this will keep code running
            line_color = "0.75"
            expl_text = "other classification"

        allplot = plt.figure(figsize=(10, 9))
        ax_ts = plt.subplot2grid((5, 6), (0, 0), rowspan=1, colspan=6, fig=allplot)

        ax_ts.set_xlabel("TRs")
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
        ax_ts2.set_xlabel("seconds")

        ax_ts.plot(mmix[:, compnum], color=line_color)

        # Title will include variance from comptable
        comp_var = "{0:.2f}".format(comptable.loc[compnum, "variance explained"])
        comp_kappa = "{0:.2f}".format(comptable.loc[compnum, "kappa"])
        comp_rho = "{0:.2f}".format(comptable.loc[compnum, "rho"])
        plt_title = "Comp. {}: variance: {}%, kappa: {}, rho: {}, {}".format(
            compnum, comp_var, comp_kappa, comp_rho, expl_text
        )
        title = ax_ts.set_title(plt_title)
        title.set_y(1.5)

        # Set range to ~1/10th of max positive or negative beta
        imgmax = 0.1 * np.abs(ts_B[:, :, :, compnum]).max()
        imgmin = imgmax * -1

        for idx, _ in enumerate(cuts):
            for imgslice in range(1, 6):
                ax = plt.subplot2grid((5, 6), (idx + 1, imgslice - 1), rowspan=1, colspan=1)
                ax.axis("off")

                if idx == 0:
                    to_plot = np.rot90(ts_B[imgslice * cuts[idx], :, :, compnum])
                if idx == 1:
                    to_plot = np.rot90(ts_B[:, imgslice * cuts[idx], :, compnum])
                if idx == 2:
                    to_plot = ts_B[:, :, imgslice * cuts[idx], compnum]

                ax_im = ax.imshow(to_plot, vmin=imgmin, vmax=imgmax, aspect="equal", cmap=png_cmap)

        # Add a color bar to the plot.
        ax_cbar = allplot.add_axes([0.8, 0.3, 0.03, 0.37])
        cbar = allplot.colorbar(ax_im, ax_cbar)
        cbar.set_label("Component Beta", rotation=90)
        cbar.ax.yaxis.set_label_position("left")

        # Get fft and freqs for this subject
        # adapted from @dangom
        spectrum, freqs = utils.get_spectrum(mmix[:, compnum], tr)

        # Plot it
        ax_fft = plt.subplot2grid((5, 6), (4, 0), rowspan=1, colspan=6)
        ax_fft.plot(freqs, spectrum)
        ax_fft.set_title("One Sided fft")
        ax_fft.set_xlabel("Hz")
        ax_fft.set_xlim(freqs[0], freqs[-1])
        plt.yticks([])

        # Fix spacing so TR label does overlap with other plots
        allplot.subplots_adjust(hspace=0.4)
        plot_name = "comp_{}.png".format(str(compnum).zfill(3))
        compplot_name = os.path.join(io_generator.out_dir, "figures", plot_name)
        plt.savefig(compplot_name)
        plt.close()


def pca_results(criteria, n_components, all_varex, io_generator):
    """
    Plot the PCA optimization curve for each criteria, and the variance explained curve.

    Parameters
    ----------
    criteria : array-like
        AIC, KIC, and MDL optimization values for increasing number of components.
    n_components : array-like
        Number of optimal components given by each criteria.
    io_generator : object
        An object containing all the information needed to generate the output.
    """

    # Plot the PCA optimization curve for each criteria
    plt.figure(figsize=(10, 9))
    plt.title("PCA Criteria")
    plt.xlabel("PCA components")
    plt.ylabel("Arbitrary Units")

    # AIC curve
    plt.plot(criteria[0, :], color="tab:blue", label="AIC")
    # KIC curve
    plt.plot(criteria[1, :], color="tab:orange", label="KIC")
    # MDL curve
    plt.plot(criteria[2, :], color="tab:green", label="MDL")

    # Vertical line depicting the optimal number of components given by AIC
    plt.vlines(
        n_components[0],
        ymin=np.min(criteria),
        ymax=np.max(criteria),
        color="tab:blue",
        linestyles="dashed",
    )
    # Vertical line depicting the optimal number of components given by KIC
    plt.vlines(
        n_components[1],
        ymin=np.min(criteria),
        ymax=np.max(criteria),
        color="tab:orange",
        linestyles="dashed",
    )
    # Vertical line depicting the optimal number of components given by MDL
    plt.vlines(
        n_components[2],
        ymin=np.min(criteria),
        ymax=np.max(criteria),
        color="tab:green",
        linestyles="dashed",
    )
    # Vertical line depicting the optimal number of components for 90% variance explained
    plt.vlines(
        n_components[3],
        ymin=np.min(criteria),
        ymax=np.max(criteria),
        color="tab:red",
        linestyles="dashed",
        label="90% varexp",
    )
    # Vertical line depicting the optimal number of components for 95% variance explained
    plt.vlines(
        n_components[4],
        ymin=np.min(criteria),
        ymax=np.max(criteria),
        color="tab:purple",
        linestyles="dashed",
        label="95% varexp",
    )

    plt.legend()

    #  Save the plot
    plot_name = "pca_criteria.png"
    pca_criteria_name = os.path.join(io_generator.out_dir, "figures", plot_name)
    plt.savefig(pca_criteria_name)
    plt.close()

    # Plot the variance explained curve
    plt.figure(figsize=(10, 9))
    plt.title("Variance Explained")
    plt.xlabel("PCA components")
    plt.ylabel("Variance Explained")

    plt.plot(all_varex, color="black", label="Variance Explained")

    # Vertical line depicting the optimal number of components given by AIC
    plt.vlines(
        n_components[0],
        ymin=0,
        ymax=1,
        color="tab:blue",
        linestyles="dashed",
        label="AIC",
    )
    # Vertical line depicting the optimal number of components given by KIC
    plt.vlines(
        n_components[1],
        ymin=0,
        ymax=1,
        color="tab:orange",
        linestyles="dashed",
        label="KIC",
    )
    # Vertical line depicting the optimal number of components given by MDL
    plt.vlines(
        n_components[2],
        ymin=0,
        ymax=1,
        color="tab:green",
        linestyles="dashed",
        label="MDL",
    )
    # Vertical line depicting the optimal number of components for 90% variance explained
    plt.vlines(
        n_components[3],
        ymin=0,
        ymax=1,
        color="tab:red",
        linestyles="dashed",
        label="90% varexp",
    )
    # Vertical line depicting the optimal number of components for 95% variance explained
    plt.vlines(
        n_components[4],
        ymin=0,
        ymax=1,
        color="tab:purple",
        linestyles="dashed",
        label="95% varexp",
    )

    plt.legend()

    #  Save the plot
    plot_name = "pca_variance_explained.png"
    pca_variance_explained_name = os.path.join(io_generator.out_dir, "figures", plot_name)
    plt.savefig(pca_variance_explained_name)
    plt.close()
