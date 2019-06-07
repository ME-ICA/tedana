"""
Apply AROMA to tedana output folder.
"""
import os.path as op
import logging

import aroma
import numpy as np
import nibabel as nib

from tedana.utils import apply_xforms

LGR = logging.getLogger(__name__)


def aroma_workflow(tedana_dir, motpars_file, t_r, xforms=None):
    """
    Apply AROMA to a tedana output directory to further identify components
    that are motion-related from the original ICA. Overwrites original
    component table with updated component classifications.

    Parameters
    ----------
    tedana_dir : :obj:`str`
        Directory containing tedana run outputs.
    motpars_file : :obj:`str`
        Motion parameters file.
    t_r : :obj:`float`
        Repetition time, in seconds.
    xforms : :obj:`list` of :obj:`str`, optional
        List of transforms to apply to warp component maps in tedana_dir into
        standard space.
    """
    betas_file = op.join(tedana_dir, 'betas_OC.nii')
    mix_file = op.join(tedana_dir, 'meica_mix.1D')
    betas_file2 = op.join(tedana_dir, 'betas_OC_std.nii')
    apply_xforms(betas_file, betas_file2, xforms)
    betas_data = nib.load(betas_file2)
    # variance normalize betas to make pseudo-z-values
    z_data = betas_data / np.std(betas_data, axis=-1)
    z_thresh_idx = np.abs(z_data) > np.median(np.abs(z_data))
    z_thresh_data = z_data * z_thresh_idx
    clf_df = aroma.run_aroma(mix_file, z_thresh_data, motpars_file, t_r)
    clf_df.loc[clf_df['classification'] == 'rejected']
    comptable = pd.read_csv(op.join(tedana_dir, 'comp_table_ica.txt'),
                            sep='\t', index_col='component')
    comptable.rename(columns={'classification': 'original_classification',
                              'rationale': 'original_rationale'},
                     inplace=True)
    comptable = pd.merge(comptable, clf_df, on='component', how='outer')
    # AROMA is only used for rejection, so original classifications for any
    # components not rejected by AROMA from tedana.
    LGR.info('Overriding original classifications for AROMA-rejected '
             'components. All other components will retain their original '
             'classifications.')
    not_rej_idx = comptable.loc[comptable['classification'] != 'rejected'].index.values
    comptable.loc[not_rej_idx, 'classification'] = comptable.loc[
        not_rej_idx, 'original_classification']
    comptable.loc[not_rej_idx, 'rationale'] = comptable.loc[
        not_rej_idx, 'original_rationale']

    # Overwrite original component table
    LGR.info('Overwriting original component table.')
    comptable.to_csv(op.join(tedana_dir, 'comp_table_ica.txt'), sep='\t',
                     index_label='component')
