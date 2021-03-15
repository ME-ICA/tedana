"""
Constants for tedana
"""

allowed_conventions = ('orig', 'bidsv1.5.0')

bids = 'bidsv1.5.0'

# filename tables
img_table = {
    'adaptive mask': {
        'orig': 'adaptive_mask',
        'bidsv1.5.0': 'desc-adaptiveGoodSignal_mask',
    },
    't2star map': {
        'orig': 't2sv',
        'bidsv1.5.0': 'T2starmap',
    },
    's0 map': {
        'orig': 's0v',
        'bidsv1.5.0': 'S0map',
    },
    'combined': {
        'orig': 'ts_OC',
        'bidsv1.5.0': 'desc-optcom_bold',
    },
    'ICA components': {
        'orig': 'ica_components',
        'bidsv1.5.0': 'desc-ICA_components',
    },
    'z-scored PCA components': {
        'orig': 'pca_components',
        'bidsv1.5.0': 'desc-PCA_stat-z_components',
    },
    'z-scored ICA components': {
        'orig': 'betas_OC',
        'bidsv1.5.0': 'desc-ICA_stat-z_components',
    },
    'ICA accepted components': {
        'orig': 'betas_hik_OC',
        'bidsv1.5.0': 'desc-ICAAccepted_components',
    },
    'z-scored ICA accepted components': {
        'orig': 'feats_OC2',
        'bidsv1.5.0': 'desc-ICAAccepted_stat-z_components',
    },
    'denoised ts': {
        'orig': 'dn_ts_OC',
        'bidsv1.5.0': 'desc-optcomDenoised_bold',
    },
    'high kappa ts': {
        'orig': 'hik_ts_OC',
        'bidsv1.5.0': 'desc-optcomAccepted_bold',
    },
    'low kappa ts': {
        'orig': 'lowk_ts_OC',
        'bidsv1.5.0': 'desc-optcomRejected_bold',
    },
    # verbose outputs
    'full t2star map': {
        'orig': 't2svG',
        'bidsv1.5.0': 'desc-full_T2starmap',
    },
    'full s0 map': {
        'orig': 's0vG',
        'bidsv1.5.0': 'desc-full_S0map',
    },
    'whitened': {
        'orig': 'ts_OC_whitened',
        'bidsv1.5.0': 'desc-optcomPCAReduced_bold',
    },
    'echo weight PCA map split': {
        'orig': 'e{0}_PCA_comp',
        'bidsv1.5.0': 'echo-{0}_desc-PCA_components',
    },
    'echo R2 PCA split': {
        'orig': 'e{0}_PCA_R2',
        'bidsv1.5.0': 'echo-{0}_desc-PCAR2ModelPredictions_components',
    },
    'echo S0 PCA split': {
        'orig': 'e{0}_PCA_S0',
        'bidsv1.5.0': 'echo-{0}_desc-PCAS0ModelPredictions_components',
    },
    'PCA component weights': {
        'orig': 'pca_weights',
        'bidsv1.5.0': 'desc-PCAAveragingWeights_components',
    },
    'PCA reduced': {
        'orig': 'oc_reduced',
        'bidsv1.5.0': 'desc-optcomPCAReduced_bold',
    },
    'echo weight ICA map split': {
        'orig': 'e{0}_ICA_comp',
        'bidsv1.5.0': 'echo-{0}_desc-ICA_components',
    },
    'echo R2 ICA split': {
        'orig': 'e{0}_ICA_R2',
        'bidsv1.5.0': 'echo-{0}_desc-ICAR2ModelPredictions_components',
    },
    'echo S0 ICA split': {
        'orig': 'e{0}_ICA_S0',
        'bidsv1.5.0': 'echo-{0}_desc-ICAS0ModelPredictions_components',
    },
    'ICA component weights': {
        'orig': 'ica_weights',
        'bidsv1.5.0': 'desc-ICAAveragingWeights_components',
    },
    'high kappa ts split': {
        'orig': 'hik_ts_e{0}',
        'bidsv1.5.0': 'echo-{0}_desc-Accepted_bold',
    },
    'low kappa ts split': {
        'orig': 'lowk_ts_e{0}',
        'bidsv1.5.0': 'echo-{0}_desc-Rejected_bold',
    },
    'denoised ts split': {
        'orig': 'dn_ts_e{0}',
        'bidsv1.5.0': 'echo-{0}_desc-Denoised_bold',
    },
    # global signal outputs
    'gs map': {
        'orig': 'T1gs',
        'bidsv1.5.0': 'desc-globalSignal_map',
    },
    'has gs combined': {
        'orig': 'tsoc_orig',
        'bidsv1.5.0': 'desc-optcomWithGlobalSignal_bold',
    },
    'removed gs combined': {
        'orig': 'tsoc_nogs',
        'bidsv1.5.0': 'desc-optcomNoGlobalSignal_bold',
    },
    't1 like': {
        'orig': 'sphis_hik',
        'bidsv1.5.0': 'desc-T1likeEffect_min',
    },
    'ICA accepted mir denoised': {
        'orig': 'hik_ts_OC_MIR',
        'bidsv1.5.0': 'desc-optcomAcceptedMIRDenoised_bold',
    },
    'mir denoised': {
        'orig': 'dn_ts_OC_MIR',
        'bidsv1.5.0': 'desc-optcomMIRDenoised_bold',
    },
    'ICA accepted mir component weights': {
        'orig': 'betas_hik_OC_MIR',
        'bidsv1.5.0': 'desc-ICAAcceptedMIRDenoised_components',
    },
}
