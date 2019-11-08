
{'tree_id': 'mytree',
    'report': 'my own report for my decision tree',
    'decision_tree_full': ['see below...']
# 'tree_refs_log'
# 'tree_info_log'
# 'tree_necessary_metrics'
}

# NOTE: I'll want to be able to manually classify components by number


# 'Decision_tree_full' : [{'nodeidx': # will be filled in at runtime
#                          'functionname': 'RhoGtKappa',
#                          'args':metrics_used':, # will be filled in at runtime
#                          'decide_comps': 'all',
#                          'iftrue': 'reject',
#                          'iffalse': 'no_change',
#                          'additionalparameters': [],
#                          'report_extra_log': [], # optionally defined by user
#                          'numfalse': [], # will be filled in at runtime
#                          'numtrue': [], # will be filled in at runtime
#                          }

#                          {'nodeidx': # will be filled in at runtime
#                          'functionname': KappaGtElbow,
#                          'metrics_used': # will be filled in at runtime
#                          'decide_comps': 'unclassified',
#                          'iftrue': 'provisionallyaccept',
#                          'iffalse': 'provisionallyreject',
#                          'additionalparameters': [],
#                          'report_extra_log': [], # optionally defined by user
#                          'numfalse': [], # will be filled in at runtime
#                          'numtrue': [], # will be filled in at runtime
#                          }

#                          {'nodeidx': # will be filled in at runtime
#                          'functionname': RhoGtElbow,
#                          'metrics_used': # will be filled in at runtime
#                          'decide_comps': ['provisionallyaccept'],
#                          'iftrue': 'provisionallyreject',
#                          'iffalse': 'no_change',
#                          'additionalparameters': [],
#                          'report_extra_log': [], # optionally defined by user
#                          'numfalse': [], # will be filled in at runtime
#                          'numtrue': [], # will be filled in at runtime
#                          }
# ]
# if refs is None:
#             refs = (
#             "Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., "
#                 "Vértes, P. E., Inati, S. J., ... & Bullmore, E. T. "
#                 "(2013). Integrated strategy for improving functional "
#                 "connectivity mapping using multiecho fMRI. Proceedings "
#                 "of the National Academy of Sciences, 110(40), "
#                 "16187-16192."
#             )

# necessary_metrics = list(set(['kappa', 'rho',
#                                       # currently countsigR2 and countsigS0
#                                       'count_sig_in_T2cluster', 'count_sig_in_S0cluster',
#                                       'DICE_FT2', 'DICE_FS0',
#                                       'T2sig_inside-outside_clusters_T',  # currently sigalnoise_t
#                                       'varexp']))

# config['nodes'] = (  # Step 1: Rho>Kappa
#     (comptable,
#      decision_tree_steps,
#      tmp_necessary_metrics)=RhoGtKappa(comptable, iftrue='reject', iffalse='no_change',
#                                        decide_comps='all', kappa_scale=1,
#                                        decision_tree_steps=None)
#     used_metrics=list(set(used_metrics + tmp_necessary_metrics))

#     # Step 2:
#     # (comptable,
#     #  decision_tree_steps,
#     #  tmp_necessary_metrics) = KappaGtElbow(comptable, iftrue='reject', iffalse='no_change',
#     #                                        decide_comps='all', kappa_scale=1,
#     #                                        decision_tree_steps=decision_tree_steps)
#     # used_metrics = list(set(used_metrics + tmp_necessary_metrics))
)
