
import matplotlib

matplotlib.use("AGG")

import logging

LGR = logging.getLogger("GENERAL")
MPL_LGR = logging.getLogger("matplotlib")
MPL_LGR.setLevel(logging.WARNING)
RepLGR = logging.getLogger("REPORT")

import json
import os
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from tedana.workflows import tedana as tedana_cli

from mapca import MovingAveragePCA

repoid = "ds000258"
wdir = "/home/portega/Downloads"

repo = os.path.join(wdir, repoid)
if not os.path.isdir(repo):
    os.mkdir(repo)

print(">>> INSTALLING DATASET:")
subprocess.run(
    f"cd {wdir} && datalad install git@github.com:OpenNeuroDatasets/{repoid}.git", shell=True
)

subjects = []

# Create pandas dataframe to store results with the following columns:
# Subject, maPCA_AIC, GIFT_AIC, maPCA_KIC, GIFT_KIC, maPCA_MDL, GIFT_MDL
results_df = pd.DataFrame(
    columns=["Subject", "maPCA_AIC", "GIFT_AIC", "maPCA_KIC", "GIFT_KIC", "maPCA_MDL", "GIFT_MDL"]
)


for sbj in os.listdir(repo):
    sbj_dir = os.path.join(repo, sbj)

    # Access subject directory
    if os.path.isdir(sbj_dir) and "sub-" in sbj_dir:
        echo_times = []
        func_files = []

        subjects.append(os.path.basename(sbj_dir))

        print(">>>> Downloading subject", sbj)

        subprocess.run(f"datalad get {sbj}/func", shell=True, cwd=repo)

        print(">>>> Searching for functional files and echo times")

        # Get functional filenames and echo times
        for func_file in os.listdir(os.path.join(repo, sbj, "func")):
            if func_file.endswith(".json"):
                with open(os.path.join(repo, sbj, "func", func_file)) as f:
                    data = json.load(f)
                    echo_times.append(data["EchoTime"])
            elif func_file.endswith(".nii.gz"):
                func_files.append(os.path.join(repo, sbj, "func", func_file))

        # Sort echo_times values from lowest to highest and multiply by 1000
        echo_times = np.array(sorted(echo_times)) * 1000

        # Sort func_files
        func_files = sorted(func_files)
        print(func_files)

        # Tedana output directory
        tedana_output_dir = os.path.join(sbj_dir, "tedana")

        print("Running tedana")

        # Run tedana
        tedana_cli.tedana_workflow(
            data=func_files,
            tes=echo_times,
            out_dir=tedana_output_dir,
            tedpca="mdl",
        )

        # Find tedana optimally combined data and mask
        tedana_optcom = os.path.join(tedana_output_dir, "desc-optcom_bold.nii.gz")
        tedana_mask = os.path.join(tedana_output_dir, "desc-adaptiveGoodSignal_mask.nii.gz")

        # Read tedana optimally combined data and mask
        tedana_optcom_img = nib.load(tedana_optcom)
        tedana_mask_img = nib.load(tedana_mask)

        # Save tedana optimally combined data and mask into mat files
        tedana_optcom_mat = os.path.join(sbj_dir, "optcom_bold.mat")
        tedana_mask_mat = os.path.join(sbj_dir, "mask.mat")
        print("Saving tedana optimally combined data and mask into mat files")
        scipy_io.savemat(tedana_optcom_mat, {"data": tedana_optcom_img.get_fdata()})
        scipy_io.savemat(tedana_mask_mat, {"mask": tedana_mask_img.get_fdata()})

        # Run mapca
        print("Running mapca")
        pca = MovingAveragePCA(normalize=True)
        _ = pca.fit_transform(tedana_optcom_img, tedana_mask_img)

        # Get AIC, KIC and MDL values
        aic = pca.aic_
        kic = pca.kic_
        mdl = pca.mdl_

        # Remove tedana output directory and the anat and func directories
        subprocess.run(f"rm -rf {tedana_output_dir}", shell=True, cwd=repo)
        subprocess.run(f"rm -rf {sbj}/anat", shell=True, cwd=repo)
        subprocess.run(f"rm -rf {sbj}/func", shell=True, cwd=repo)

        # Here run matlab script with subprocess.run
        print("Running GIFT version of maPCA")

        # Append AIC, KIC and MDL values to a pandas dataframe
        print("Appending AIC, KIC and MDL values to a pandas dataframe")
        results_df = results_df.append(
            {
                "Subject": sbj,
                "maPCA_AIC": aic,
                "GIFT_AIC": 0,
                "maPCA_KIC": kic,
                "GIFT_KIC": 0,
                "maPCA_MDL": mdl,
                "GIFT_MDL": 0,
            },
            ignore_index=True,
        )

        print("Subject", sbj, "done")

# Save pandas dataframe to csv file
results_df.to_csv(os.path.join(wdir, "results.csv"), index=False)


