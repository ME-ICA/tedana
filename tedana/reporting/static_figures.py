"""Functions to creating figures to inspect tedana output."""

import logging
import os
from io import BytesIO

import matplotlib
import nibabel as nb
import numpy as np

matplotlib.use("AGG")
import matplotlib.pyplot as plt
from nilearn import masking, plotting

from tedana import io, stats, utils

LGR = logging.getLogger("GENERAL")
MPL_LGR = logging.getLogger("matplotlib")
MPL_LGR.setLevel(logging.WARNING)
RepLGR = logging.getLogger("REPORT")


def _trim_edge_zeros(arr):
    """
    Trims away the zero-filled slices that surround many 3/4D arrays.

    Parameters
    ----------
    ndarray : (S x T) array_like
        an array with signal, surrounded by slices that contain only zeros
        that should be removed.

    Returns
    -------
    ndarray : (S x T) array_like
        an array with reduced dimensions, such that the array contains only
        non_zero values from edge to edge.
    """
    mask = arr != 0
    bounding_box = tuple(slice(np.min(indexes), np.max(indexes) + 1) for indexes in np.where(mask))
    return arr[bounding_box]


def carpet_plot(
    optcom_ts,
    denoised_ts,
    hikts,
    lowkts,
    mask,
    io_generator,
    gscontrol=None,
):
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
    fig.savefig(
        os.path.join(io_generator.out_dir, "figures", f"{io_generator.prefix}carpet_optcom.svg")
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    plotting.plot_carpet(
        dn_img,
        mask_img,
        figure=fig,
        axes=ax,
        title="Denoised Data",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(io_generator.out_dir, "figures", f"{io_generator.prefix}carpet_denoised.svg")
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    plotting.plot_carpet(
        hik_img,
        mask_img,
        figure=fig,
        axes=ax,
        title="High-Kappa Data",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(io_generator.out_dir, "figures", f"{io_generator.prefix}carpet_accepted.svg")
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    plotting.plot_carpet(
        lowk_img,
        mask_img,
        figure=fig,
        axes=ax,
        title="Low-Kappa Data",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(io_generator.out_dir, "figures", f"{io_generator.prefix}carpet_rejected.svg")
    )

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
        fig.savefig(
            os.path.join(
                io_generator.out_dir,
                "figures",
                f"{io_generator.prefix}carpet_optcom_nogsr.svg",
            )
        )

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
        fig.savefig(
            os.path.join(
                io_generator.out_dir,
                "figures",
                f"{io_generator.prefix}carpet_denoised_mir.svg",
            )
        )

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
        fig.savefig(
            os.path.join(
                io_generator.out_dir,
                "figures",
                f"{io_generator.prefix}carpet_accepted_mir.svg",
            )
        )


def plot_component(
    *,
    stat_img,
    component_timeseries,
    power_spectrum,
    frequencies,
    tr,
    classification_color,
    png_cmap,
    title,
    out_file,
):
    """Create a figure with a component's spatial map, time series, and power spectrum.

    Parameters
    ----------
    stat_img : :obj:`nibabel.Nifti1Image`
        Image of the component's spatial map
    component_timeseries : (T,) array_like
        Time series of the component
    power_spectrum : (T,) array_like
        Power spectrum of the component's time series
    frequencies : (T,) array_like
        Frequencies for the power spectrum
    tr : float
        Repetition time of the time series
    classification_color : str
        Color to use for the time series and power spectrum
    png_cmap : str
        Colormap to use for the spatial map
    title : str
        Title for the figure
    out_file : str
        Path to save the figure
    """
    import matplotlib.image as mpimg
    from matplotlib import gridspec

    # Set range to ~1/10th of max positive or negative beta
    imgmax = 0.1 * np.max(np.abs(stat_img.get_fdata()))

    # Save the figure to an in-memory file object
    display = plotting.plot_stat_map(
        stat_img,
        bg_img=None,
        display_mode="mosaic",
        cut_coords=5,
        vmax=imgmax,
        cmap=png_cmap,
        symmetric_cbar=True,
        colorbar=False,
        draw_cross=False,
        annotate=False,
    )
    display.annotate(size=30)
    example_ax = list(display.axes.values())[0]
    nilearn_fig = example_ax.ax.figure

    with BytesIO() as buf:
        nilearn_fig.savefig(buf, format="png")
        buf.seek(0)

        # Read the image back into an image array
        img = mpimg.imread(buf)

    plt.close(nilearn_fig)

    # Make the width of the original image the width of the new figure,
    # but add top and bottom axes that each take up 10% of the height
    width = 10
    img_hw_ratio = img.shape[0] / img.shape[1]
    img_dims = (width, (width * img_hw_ratio * 1.6))

    # Create a new figure and gridspec
    fig = plt.figure(figsize=img_dims)
    fig.suptitle(title, fontsize=14)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 10, 2], hspace=0.2)

    # Create three subplots
    # First is the time series of the component
    ax_ts = fig.add_subplot(gs[0])
    ax_ts.plot(component_timeseries, color=classification_color)
    ax_ts.set_xlim(0, len(component_timeseries) - 1)
    ax_ts.set_yticks([])

    max_xticks = 10
    xloc = plt.MaxNLocator(max_xticks)
    ax_ts.xaxis.set_major_locator(xloc)

    ax_ts2 = ax_ts.twiny()
    ax1_xs = ax_ts.get_xticks()

    ax2_xs = []
    for x in ax1_xs:
        # Limit to 2 decimal places
        seconds_val = round(x * tr, 2)
        ax2_xs.append(seconds_val)

    ax_ts2.set_xticks(ax1_xs)
    ax_ts2.set_xlim(ax_ts.get_xbound())
    ax_ts2.set_xticklabels(ax2_xs)
    ax_ts2.set_xlabel("seconds")

    # Second is the cached image of the spatial map
    ax_map = fig.add_subplot(gs[1])
    ax_map.axis("off")
    ax_map.imshow(img)

    # Third is the power spectrum of the component's time series
    ax_fft = fig.add_subplot(gs[2])
    ax_fft.plot(frequencies, power_spectrum, color=classification_color)
    ax_fft.set_title("One-Sided FFT")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_xlim(0, frequencies.max())
    ax_fft.set_yticks([])

    # Get the current positions of the second and last subplots
    # pos_ts = ax_ts.get_position()
    # pos_freq = ax_fft.get_position()

    # Adjust the positions of the second and last subplots
    # ax_ts.set_position([pos_ts.x0, pos_ts.y0 - 0.1, pos_ts.width, pos_ts.height])
    # ax_fft.set_position([pos_freq.x0, pos_freq.y0 - 0.2, pos_freq.width, pos_freq.height])

    fig.savefig(out_file)
    plt.close(fig)


def comp_figures(ts, mask, comptable, mmix, io_generator, png_cmap):
    """Create static figures that highlight certain aspects of tedana processing.

    This includes a figure for each component showing the component time course,
    the spatial weight map and a fast Fourier transform of the time course.

    Parameters
    ----------
    ts : (S x T) array_like
        Time series from which to derive ICA betas
    mask : (S,) array_like
        Boolean mask array
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    io_generator : :obj:`tedana.io.OutputGenerator`
        Output Generator object to use for this workflow
    """
    # Flip signs of mixing matrix as needed
    mmix = mmix * comptable["optimal sign"].values

    # regenerate the beta images
    component_maps_arr = stats.get_coeffs(ts, mmix, mask)
    component_maps_arr = component_maps_arr.reshape(
        io_generator.reference_img.shape[:3] + component_maps_arr.shape[1:],
    )

    # Get repetition time from reference image
    tr = io_generator.reference_img.header.get_zooms()[-1]

    # Remove trailing ';' from rationale column
    # comptable["rationale"] = comptable["rationale"].str.rstrip(";")
    for compnum in comptable.index.values:
        if comptable.loc[compnum, "classification"] == "accepted":
            line_color = "g"
            expl_text = "accepted reason(s): " + str(comptable.loc[compnum, "classification_tags"])

        elif comptable.loc[compnum, "classification"] == "rejected":
            line_color = "r"
            expl_text = "rejected reason(s): " + str(comptable.loc[compnum, "classification_tags"])

        elif comptable.loc[compnum, "classification"] == "ignored":
            line_color = "k"
            expl_text = "ignored reason(s): " + str(comptable.loc[compnum, "classification_tags"])

        else:
            # Classification not added
            # If new, this will keep code running
            line_color = "0.75"
            expl_text = "other classification"

        # Title will include variance from comptable
        comp_var = f"{comptable.loc[compnum, 'variance explained']:.2f}"
        comp_kappa = f"{comptable.loc[compnum, 'kappa']:.2f}"
        comp_rho = f"{comptable.loc[compnum, 'rho']:.2f}"

        plt_title = (
            f"Comp. {compnum}: variance: {comp_var}%, kappa: {comp_kappa}, "
            f"rho: {comp_rho}, {expl_text}"
        )
        component_img = nb.Nifti1Image(
            component_maps_arr[:, :, :, compnum],
            affine=io_generator.reference_img.affine,
            header=io_generator.reference_img.header,
        )

        component_timeseries = mmix[:, compnum]

        # Get fft and freqs for this component
        # adapted from @dangom
        spectrum, freqs = utils.get_spectrum(component_timeseries, tr)

        plot_name = f"{io_generator.prefix}comp_{str(compnum).zfill(3)}.png"
        compplot_name = os.path.join(io_generator.out_dir, "figures", plot_name)

        plot_component(
            stat_img=component_img,
            component_timeseries=component_timeseries,
            power_spectrum=spectrum,
            frequencies=freqs,
            tr=tr,
            classification_color=line_color,
            png_cmap=png_cmap,
            title=plt_title,
            out_file=compplot_name,
        )


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
    plot_name = f"{io_generator.prefix}pca_criteria.png"
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
    plot_name = f"{io_generator.prefix}pca_variance_explained.png"
    pca_variance_explained_name = os.path.join(io_generator.out_dir, "figures", plot_name)
    plt.savefig(pca_variance_explained_name)
    plt.close()


def plot_t2star_and_s0(
    *,
    io_generator: io.OutputGenerator,
    mask: np.ndarray,
) -> None:
    """Create T2* and S0 maps and histograms.

    Parameters
    ----------
    io_generator : :obj:`~tedana.io.OutputGenerator`
        The output generator for this workflow
    mask : (S,) :obj:`numpy.ndarray`
        Binary mask used to apply to the data.
    """
    t2star_img = io_generator.get_name("t2star img")
    s0_img = io_generator.get_name("s0 img")
    mask_img = io.new_nii_like(io_generator.reference_img, mask.astype(int))
    assert os.path.isfile(t2star_img), f"File {t2star_img} does not exist"
    assert os.path.isfile(s0_img), f"File {s0_img} does not exist"

    # Plot histograms
    t2star_data = masking.apply_mask(t2star_img, mask_img)
    t2s_p02, t2s_p98 = np.percentile(t2star_data, [2, 98])
    t2star_histogram = f"{io_generator.prefix}t2star_histogram.svg"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(t2star_data[t2star_data <= t2s_p98], bins=100)
    ax.set_xlim(0, t2s_p98)
    ax.set_title("T2*", fontsize=20)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_xlabel("Seconds\n(limited to 98th percentile)", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(io_generator.out_dir, "figures", t2star_histogram))

    s0_data = masking.apply_mask(s0_img, mask_img)
    s0_p02, s0_p98 = np.percentile(s0_data, [2, 98])
    s0_histogram = f"{io_generator.prefix}s0_histogram.svg"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(s0_data[s0_data <= s0_p98], bins=100)
    ax.set_xlim(0, s0_p98)
    ax.set_title("S0", fontsize=20)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_xlabel("Arbitrary Units\n(limited to 98th percentile)", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(io_generator.out_dir, "figures", s0_histogram))

    # Plot T2* and S0 maps
    t2star_plot = f"{io_generator.prefix}t2star_brain.svg"
    plotting.plot_stat_map(
        t2star_img,
        bg_img=None,
        display_mode="mosaic",
        symmetric_cbar=False,
        black_bg=True,
        cmap="gray",
        vmin=t2s_p02,
        vmax=t2s_p98,
        annotate=False,
        output_file=os.path.join(io_generator.out_dir, "figures", t2star_plot),
    )

    s0_plot = f"{io_generator.prefix}s0_brain.svg"
    plotting.plot_stat_map(
        s0_img,
        bg_img=None,
        display_mode="mosaic",
        symmetric_cbar=False,
        black_bg=True,
        cmap="gray",
        vmin=s0_p02,
        vmax=s0_p98,
        annotate=False,
        output_file=os.path.join(io_generator.out_dir, "figures", s0_plot),
    )
