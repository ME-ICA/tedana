
Init dict also contains
'tree_report_log'
'tree_refs_log'
'tree_info_log'
'tree_necessary_metrics'

Decision_tree_full = [{'nodeidx': # will be filled in at runtime
                         'functionname': RhoGtKappa,
                         'metrics_used':, # will be filled in at runtime
                         'decide_comps': 'all',
                         'iftrue': 'reject',
                         'iffalse': 'no_change',
                         'additionalparameters': [],
                         'report_extra_log': [], # optionally defined by user
                         'numfalse': [], # will be filled in at runtime
                         'numtrue': [], # will be filled in at runtime
                         } 

                         {'nodeidx': # will be filled in at runtime
                         'functionname': KappaGtElbow,
                         'metrics_used': # will be filled in at runtime
                         'decide_comps': 'unclassified',
                         'iftrue': 'provisionallyaccept',
                         'iffalse': 'provisionallyreject',
                         'additionalparameters': [],
                         'report_extra_log': [], # optionally defined by user
                         'numfalse': [], # will be filled in at runtime
                         'numtrue': [], # will be filled in at runtime
                         } 

                         {'nodeidx': # will be filled in at runtime
                         'functionname': RhoGtElbow,
                         'metrics_used': # will be filled in at runtime
                         'decide_comps': ['provisionallyaccept'],
                         'iftrue': 'provisionallyreject',
                         'iffalse': 'no_change',
                         'additionalparameters': [],
                         'report_extra_log': [], # optionally defined by user
                         'numfalse': [], # will be filled in at runtime
                         'numtrue': [], # will be filled in at runtime
                         } 



]