"""Functions to creating figures to inspect tedana output."""

import logging
import os
import warnings
from io import BytesIO

import matplotlib
import nibabel as nb
import numpy as np
import pandas as pd

matplotlib.use("AGG")
import matplotlib.pyplot as plt
from nilearn import image, masking, plotting

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
    ndarray : (Mb x T) array_like
        an array with signal, surrounded by slices that contain only zeros that should be removed,
        where `Mb` is samples in base mask, and `T` is time.

    Returns
    -------
    ndarray : (Mb x T) array_like
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
    optcom_ts, denoised_ts, hikts, lowkts : (Mb x T) array_like
        Different types of data to plot, where `Mb` is samples in base mask, and `T` is time.
    mask : img-like
        Binary mask image used to mask the data.
    io_generator : :obj:`tedana.io.OutputGenerator`
        The output generator for this workflow
    gscontrol : {None, 'mir', 'gsr'} or :obj:`list`, optional
        Additional denoising steps applied in the workflow.
        If any gscontrol methods were applied, then additional carpet plots will be generated for
        pertinent outputs from those steps.
        Default is None.
    """
    optcom_img = masking.unmask(optcom_ts.T, io_generator.mask)
    dn_img = masking.unmask(denoised_ts.T, io_generator.mask)
    hik_img = masking.unmask(hikts.T, io_generator.mask)
    lowk_img = masking.unmask(lowkts.T, io_generator.mask)

    # Carpet plots
    fig, ax = plt.subplots(figsize=(14, 7))
    plotting.plot_carpet(
        optcom_img,
        mask,
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
        mask,
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
        mask,
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
        mask,
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
            mask,
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
            mask,
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

        if io_generator.verbose:
            mir_denoised_img = io_generator.get_name("ICA accepted mir denoised img")
            fig, ax = plt.subplots(figsize=(14, 7))
            plotting.plot_carpet(
                mir_denoised_img,
                mask,
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

    # nilearn raises a warning when creating a figure from an image with a non-diagonal affine.
    # This is not relevant for how we use this function and it flood our screen
    # output with repeated warnings so suppressing this warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A non-diagonal affine.*", category=UserWarning)
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
            resampling_interpolation="nearest",
        )

    display.annotate(size=20)
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


def comp_figures(ts, component_table, mixing, io_generator, png_cmap):
    """Create static figures that highlight certain aspects of tedana processing.

    This includes a figure for each component showing the component time course,
    the spatial weight map and a fast Fourier transform of the time course.

    Parameters
    ----------
    ts : (Mb x T) array_like
        Time series from which to derive ICA betas, where `Mb` is samples in base mask,
        and `T` is time
    component_table : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
    mixing : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    io_generator : :obj:`tedana.io.OutputGenerator`
        Output Generator object to use for this workflow
    """
    # regenerate the beta images
    component_maps_arr = stats.get_coeffs(ts, mixing)
    component_maps_arr = masking.unmask(component_maps_arr.T, io_generator.mask)
    component_maps_arr = component_maps_arr.get_fdata()

    # Get repetition time from reference image
    tr = io_generator.reference_img.header.get_zooms()[-1]

    # Remove trailing ';' from rationale column
    # component_table["rationale"] = component_table["rationale"].str.rstrip(";")
    for compnum in component_table.index.values:
        if component_table.loc[compnum, "classification"] == "accepted":
            line_color = "g"
            expl_text = "accepted reason(s): " + str(
                component_table.loc[compnum, "classification_tags"]
            )

        elif component_table.loc[compnum, "classification"] == "rejected":
            line_color = "r"
            expl_text = "rejected reason(s): " + str(
                component_table.loc[compnum, "classification_tags"]
            )

        elif component_table.loc[compnum, "classification"] == "ignored":
            line_color = "k"
            expl_text = "ignored reason(s): " + str(
                component_table.loc[compnum, "classification_tags"]
            )

        else:
            # Classification not added
            # If new, this will keep code running
            line_color = "0.75"
            expl_text = "other classification"

        # Title will include variance from component_table
        comp_var = f"{component_table.loc[compnum, 'variance explained']:.2f}"
        comp_kappa = f"{component_table.loc[compnum, 'kappa']:.2f}"
        comp_rho = f"{component_table.loc[compnum, 'rho']:.2f}"

        plt_title = (
            f"Comp. {compnum}: variance: {comp_var}%, kappa: {comp_kappa}, "
            f"rho: {comp_rho}, {expl_text}"
        )
        component_img = nb.Nifti1Image(
            component_maps_arr[:, :, :, compnum],
            affine=io_generator.reference_img.affine,
            header=io_generator.reference_img.header,
        )

        component_timeseries = mixing[:, compnum]

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
    mask: nb.Nifti1Image,
) -> None:
    """Create T2* and S0 maps and histograms.

    Parameters
    ----------
    io_generator : :obj:`~tedana.io.OutputGenerator`
        The output generator for this workflow
    mask : img
        Binary mask image used to apply to the data.
    """
    t2star_img = io_generator.get_name("t2star img")
    s0_img = io_generator.get_name("s0 img")
    assert os.path.isfile(t2star_img), f"File {t2star_img} does not exist"

    # Check if S0 image exists, add message to log if not
    s0_exists = os.path.isfile(s0_img)
    if not s0_exists:
        LGR.info(
            "S0 maps and T2* fit metrics are not in report since a pre-existing "
            "T2* map was provided"
        )

    # Plot histograms
    t2star_data = masking.apply_mask(t2star_img, mask)
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

    # Only plot S0 data if the file exists
    if s0_exists:
        s0_data = masking.apply_mask(s0_img, mask)
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A non-diagonal affine.*", category=UserWarning)
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
            resampling_interpolation="nearest",
        )

    # Only plot S0 map if the file exists
    if s0_exists:
        s0_plot = f"{io_generator.prefix}s0_brain.svg"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="A non-diagonal affine.*", category=UserWarning
            )
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
                resampling_interpolation="nearest",
            )


def plot_rmse(
    *,
    io_generator: io.OutputGenerator,
):
    """Plot the residual mean squared error map and time series for the monoexponential model fit.

    Parameters
    ----------
    io_generator : :obj:`~tedana.io.OutputGenerator`
        The output generator for this workflow.
    """
    import pandas as pd

    rmse_img = io_generator.get_name("rmse img")
    confounds_file = io_generator.get_name("confounds tsv")
    mask_img = io_generator.get_name("adaptive mask img")
    # At least 2 good echoes
    mask_img = image.binarize_img(mask_img, threshold=1.5, two_sided=False, copy_header=True)

    rmse_data = masking.apply_mask(rmse_img, mask_img)
    rmse_p02, rmse_p98 = np.percentile(rmse_data, [2, 98])

    # Get repetition time from reference image
    tr = io_generator.reference_img.header.get_zooms()[-1]

    # Load the confounds file
    confounds_df = pd.read_table(confounds_file)

    fig, ax = plt.subplots(figsize=(10, 6))
    rmse_arr = confounds_df["rmse_median"].values
    p25_arr = confounds_df["rmse_percentile25"].values
    p75_arr = confounds_df["rmse_percentile75"].values
    p02_arr = confounds_df["rmse_percentile02"].values
    p98_arr = confounds_df["rmse_percentile98"].values
    time_arr = np.arange(confounds_df.shape[0]) * tr
    ax.plot(time_arr, rmse_arr, color="black")
    ax.fill_between(
        time_arr,
        p25_arr,
        p75_arr,
        color="blue",
        alpha=0.2,
    )
    ax.plot(time_arr, p02_arr, color="black", linestyle="dashed")
    ax.plot(time_arr, p98_arr, color="black", linestyle="dashed")
    ax.set_ylabel("RMSE", fontsize=16)
    ax.set_xlabel(
        "Time (s)",
        fontsize=16,
    )
    ax.legend(["Median", "25th-75th percentiles", "2nd and 98th percentiles"])
    ax.set_title("Root mean squared error of T2* and S0 fit across voxels", fontsize=20)
    rmse_ts_plot = os.path.join(
        io_generator.out_dir,
        "figures",
        f"{io_generator.prefix}rmse_timeseries.svg",
    )
    ax.set_xlim(0, time_arr[-2])
    fig.savefig(rmse_ts_plot)
    plt.close(fig)

    # Plot RMSE
    rmse_brain_plot = os.path.join(
        io_generator.out_dir,
        "figures",
        f"{io_generator.prefix}rmse_brain.svg",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A non-diagonal affine.*", category=UserWarning)
        plotting.plot_stat_map(
            rmse_img,
            bg_img=None,
            display_mode="mosaic",
            cut_coords=4,
            symmetric_cbar=False,
            black_bg=True,
            cmap="Reds",
            vmin=rmse_p02,
            vmax=rmse_p98,
            annotate=False,
            output_file=rmse_brain_plot,
            resampling_interpolation="nearest",
        )


def plot_adaptive_mask(
    *,
    optcom: np.ndarray,
    io_generator: io.OutputGenerator,
):
    """Create a figure to show the adaptive mask.

    This figure shows the base mask, the adaptive mask thresholded for denoising (threshold >= 1),
    and the adaptive mask thresholded for classification (threshold >= 3),
    overlaid on the mean optimal combination image.

    Parameters
    ----------
    optcom : (Mb x T) :obj:`numpy.ndarray`
        Optimal combination of components, where `Mb` is samples in base mask, and `T` is time.
        The mean image over time is used as the underlay for the figure.
    io_generator : :obj:`~tedana.io.OutputGenerator`
        The output generator for this workflow.
    """
    from matplotlib.lines import Line2D

    adaptive_mask_img = io_generator.get_name("adaptive mask img")
    mean_optcom_img = masking.unmask(np.mean(optcom, axis=1), io_generator.mask)

    # Concatenate the three masks used in tedana to treat as a probabilistic atlas
    # At least 1 good echo
    mask_denoise = image.binarize_img(
        adaptive_mask_img,
        threshold=0.5,
        two_sided=False,
        copy_header=True,
    )
    # At least 3 good echoes
    mask_clf = image.binarize_img(
        adaptive_mask_img,
        threshold=2.5,
        two_sided=False,
        copy_header=True,
    )

    color_dict = {
        "Initial mask only": "#DC267F",
        "Optimal combination & Initial": "#FFB000",
        "Classification, OC & Initial": "#648FFF",
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A non-diagonal affine.*", category=UserWarning)
        ob = plotting.plot_epi(
            epi_img=mean_optcom_img,
            annotate=False,
            draw_cross=False,
            colorbar=False,
            display_mode="mosaic",
            cut_coords=5,
        )
        ob.add_contours(
            mask_clf,
            threshold=0.2,
            levels=[0.5],
            colors=[color_dict["Classification, OC & Initial"]],
            linewidths=1.5,
        )
        ob.add_contours(
            mask_denoise,
            threshold=0.2,
            levels=[0.5],
            colors=[color_dict["Optimal combination & Initial"]],
            linewidths=1.5,
        )
        ob.add_contours(
            io_generator.mask,
            threshold=0.2,
            levels=[0.5],
            colors=[color_dict["Initial mask only"]],
            linewidths=1.5,
        )

    legend_elements = []
    for k, v in color_dict.items():
        line = Line2D([0], [0], color=v, label=k, markersize=10)
        legend_elements.append(line)

    fig = ob.frame_axes.get_figure()
    width = fig.get_size_inches()[0]

    ob.frame_axes.set_zorder(100)
    legend = ob.frame_axes.legend(
        handles=legend_elements,
        facecolor="black",
        ncols=3,
        loc="lower right",
        fancybox=True,
        shadow=True,
        fontsize=width,
    )

    legend.get_frame().set_visible(False)

    for text in legend.get_texts():
        text.set_color("white")

    adaptive_mask_plot = f"{io_generator.prefix}adaptive_mask.svg"
    fig.savefig(os.path.join(io_generator.out_dir, "figures", adaptive_mask_plot))


def plot_gscontrol(
    *,
    io_generator: io.OutputGenerator,
    gscontrol: list,
    png_cmap: str,
):
    """Plot the results of the gscontrol steps.

    Parameters
    ----------
    io_generator : :obj:`~tedana.io.OutputGenerator`
        The output generator for this workflow.
    gscontrol : list
        List of gscontrol methods applied.
    """
    import pandas as pd

    if "gsr" in gscontrol or "mir" in gscontrol:
        confounds_file = io_generator.get_name("confounds tsv")
        confounds_df = pd.read_table(confounds_file)

        # Get repetition time from reference image
        tr = io_generator.reference_img.header.get_zooms()[-1]

    if "gsr" in gscontrol:
        gsr_img = nb.load(io_generator.get_name("gs img"))

        # Get fft and freqs for this component
        # adapted from @dangom
        timeseries = confounds_df["global_signal"].values
        spectrum, freqs = utils.get_spectrum(timeseries, tr)

        plot_name = f"{io_generator.prefix}gsr_boldmap.svg"
        plot_name = os.path.join(io_generator.out_dir, "figures", plot_name)

        plot_component(
            stat_img=gsr_img,
            component_timeseries=timeseries,
            power_spectrum=spectrum,
            frequencies=freqs,
            tr=tr,
            classification_color="red",
            png_cmap=png_cmap,
            title="Global Signal Regression",
            out_file=plot_name,
        )

    if "mir" in gscontrol:
        mir_img = nb.load(io_generator.get_name("t1 like img"))

        # Get fft and freqs for this component
        # adapted from @dangom
        timeseries = confounds_df["mir_global_signal"].values
        spectrum, freqs = utils.get_spectrum(timeseries, tr)

        plot_name = f"{io_generator.prefix}mir_boldmap.svg"
        plot_name = os.path.join(io_generator.out_dir, "figures", plot_name)

        plot_component(
            stat_img=mir_img,
            component_timeseries=timeseries,
            power_spectrum=spectrum,
            frequencies=freqs,
            tr=tr,
            classification_color="red",
            png_cmap=png_cmap,
            title="Minimum Image Regression",
            out_file=plot_name,
        )


def plot_heatmap(
    *,
    correlation_df: pd.DataFrame,
    component_table: pd.DataFrame,
    out_file: str,
):
    """Plot a heatmap of correlations between external regressors and ICA components.

    Parameters
    ----------
    correlation_df : (E x C) :obj:`pandas.DataFrame`
        A DataFrame where rows are external regressor names, columns are component names,
        and values are the Pearson correlation coefficients.
    component_table : pandas.DataFrame
        Component table.
    out_file : str
        The output file name.
    """
    import re

    import scipy.cluster.hierarchy as spc
    import seaborn as sns

    corr_df = correlation_df.copy()
    regressors = corr_df.index.tolist()

    # Perform hierarchical clustering on rows
    corr = corr_df.T.corr().values
    pdist_uncondensed = 1.0 - corr
    pdist_condensed = np.concatenate([row[i + 1 :] for i, row in enumerate(pdist_uncondensed)])
    linkage = spc.linkage(pdist_condensed, method="complete")
    cluster_assignments = spc.fcluster(linkage, 0.5 * pdist_condensed.max(), "distance")
    idx = np.argsort(cluster_assignments)
    new_regressor_order = [regressors[i] for i in idx]
    corr_df = corr_df.loc[new_regressor_order]

    # Get the metrics for the models from the component table
    pattern = "(R2stat .* model)"
    searches = [re.search(pattern, col) for col in component_table.columns]
    models = [search.group(1) for search in searches if search is not None]
    models_df = component_table[models]
    # Remove the R2stat string from the models_df column names
    models_df.columns = models_df.columns.str.replace("R2stat ", "").str.replace(" model", "")
    models_df = models_df.T  # transpose so components are columns

    n_regressors = corr_df.shape[0]
    n_components = corr_df.shape[1]
    n_models = models_df.shape[0]
    ratio = n_regressors / n_models

    fig, axes = plt.subplots(
        figsize=(n_components * 0.25, (n_regressors * 0.25) + (n_models * 0.25) + 0.5),
        nrows=2,
        height_ratios=[n_regressors, n_models],
        sharex=True,
    )
    sns.heatmap(
        corr_df,
        cmap="seismic",
        center=0,
        vmax=1,
        vmin=-1,
        square=True,
        linewidths=0.5,
        cbar_kws={
            "shrink": 0.85,
            "label": "Correlation",
            "ticks": [-1, 0, 1],
            "pad": 0.01,
            "aspect": 20,
        },
        ax=axes[0],
    )
    axes[0].tick_params(axis="y", labelrotation=0)
    axes[0].set_xticks([])
    axes[0].tick_params(axis="x", bottom=False)

    sns.heatmap(
        models_df,
        cmap="Reds",
        center=0.5,
        vmax=1,
        vmin=0,
        square=True,
        linewidths=0.5,
        cbar_kws={
            "shrink": 0.85,
            "label": "R-Squared",
            "ticks": [0, 1],
            "pad": 0.01,
            "aspect": 20 / ratio,
        },
        ax=axes[1],
    )
    axes[1].tick_params(axis="y", labelrotation=0)
    axes[1].set_xlabel("Component", fontsize=16)

    fig.savefig(out_file, bbox_inches="tight")


def _correlate_dataframes(df1, df2):
    """Correlate each column in two DataFrames using numpy.corrcoef.

    Parameters
    ----------
    df1 : pandas.DataFrame of shape (T, C)
        The first DataFrame.
    df2 : pandas.DataFrame of shape (T, E)
        The second DataFrame. Rows must align with df1.

    Returns
    -------
    correlation_df : pandas.DataFrame of shape (C, E)
        A DataFrame where rows are columns from df1, columns are columns from df2,
        and values are the Pearson correlation coefficients.
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")

    if df1.shape[0] != df2.shape[0]:
        raise ValueError("DataFrames must have the same number of rows.")

    # Convert DataFrames to numpy arrays
    arr1 = df1.values
    arr2 = df2.values

    # Concatenate arrays column-wise
    # This creates an array where the first df1.shape[1] columns are from df1
    # and the subsequent columns are from df2.
    combined_arr = np.hstack((arr1, arr2))

    # Calculate the full correlation matrix.
    # np.corrcoef expects variables as rows, so we transpose combined_arr.
    # If df1 has m columns and df2 has n columns, combined_arr.T has m+n rows.
    # full_corr_matrix will be an (m+n) x (m+n) matrix.
    full_corr_matrix = np.corrcoef(combined_arr.T)

    # Extract the part of the matrix that corresponds to correlations
    # between columns of df1 and columns of df2.
    # This is the block from row 0 to df1.shape[1]-1,
    # and from column df1.shape[1] to the end.
    num_cols_df1 = df1.shape[1]
    cross_corr_matrix = full_corr_matrix[:num_cols_df1, num_cols_df1:]

    # Convert the result back to a DataFrame with appropriate labels
    correlation_df = pd.DataFrame(cross_corr_matrix, index=df1.columns, columns=df2.columns)
    return correlation_df


def plot_decay_variance(
    *,
    io_generator: io.OutputGenerator,
):
    """Plot the variance of the T2* and S0 estimates.

    Parameters
    ----------
    io_generator : :obj:`~tedana.io.OutputGenerator`
        The output generator for this workflow.
    """
    mask_img = io_generator.get_name("adaptive mask img")
    # At least 2 good echoes
    mask_img = image.binarize_img(
        mask_img,
        threshold=1.5,
        two_sided=False,
        copy_header=True,
    )

    names = [
        "stat-variance_desc-t2star_statmap",
        "stat-variance_desc-s0_statmap",
        "stat-covariance_desc-t2star+s0_statmap",
    ]
    imgs = ["t2star variance img", "s0 variance img", "t2star-s0 covariance img"]
    for name, img in zip(names, imgs):
        in_file = io_generator.get_name(img)
        data = masking.apply_mask(in_file, mask_img)
        data_p02, data_p98 = np.percentile(data, [2, 98])
        plot_name = f"{io_generator.prefix}{name}.svg"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A non-diagonal affine.*",
                category=UserWarning,
            )
            plotting.plot_stat_map(
                in_file,
                bg_img=None,
                display_mode="mosaic",
                symmetric_cbar=False,
                black_bg=True,
                cmap="Reds",
                vmin=data_p02,
                vmax=data_p98,
                annotate=False,
                output_file=os.path.join(io_generator.out_dir, "figures", plot_name),
                resampling_interpolation="nearest",
            )
