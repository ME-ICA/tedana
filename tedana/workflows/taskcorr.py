import numpy as np
import os as os
import scipy.stats as sct
import sys
import argparse


def _get_parser():
	"""
	Parses command line inputs for this function

	Returns
	-------
	parser.parse_args() : argparse dict
	"""
	parser = argparse.ArgumentParser()
	# Argument parser follow template provided by RalphyZ, also used by tedana.
	# https://stackoverflow.com/a/43456577
	optional = parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	required.add_argument('-td','--ted-dir',
						  dest='out_dir',
						  type=str,
						  help=('Output directory of tedana, i.e. where '
								'the mixing matrix and the component'
								'table are located.'),								
						  required=True)
	optional.add_argument('--mix',
						  dest='mixm',
						  nargs=1,
						  metavar='FILE',
						  type=lambda x: is_valid_file(parser, x),
						  help=('File containing ME-ICA mixing matrix.'),
						  default='meica_mix.1D')
	optional.add_argument('--ctab',
						  dest='ctab',
						  metavar='FILE',
						  type=lambda x: is_valid_file(parser, x),
						  help='File containing TEDICA component table',
						  default='comp_table_ica.txt')
	optional.add_argument('-gts','--good_ts',
						  dest='good_ts',
						  metavar='FILE',
						  type=lambda x: is_valid_file(parser, x),
						  help=('File containing timeseries that should'
								'be accepted. It can be a 1D array or a'
								'matrix, where columns are the timeseries'),
						  default=None)
	optional.add_argument('-bts','--bad_ts',
						  dest='bad_ts',
						  metavar='FILE',
						  type=lambda x: is_valid_file(parser, x),
						  help=('File containing timeseries that should'
								'be rejected. It can be a 1D array or a'
								'matrix, where columns are the timeseries'),
						  default=None)
	optional.add_argument('-t','--thr',
						  dest='thr',
						  type=float,
						  help='Threshold to apply for component selection',
						  default=0.6)
	parser._action_groups.append(optional)
	return parser

def import_file(file_name):
	"""
	Importing files and normalising them for Normalised Cross-Correlation
	"""
	timeseries = np.genfromtxt(file_name)
	norm_ts = sct.zscore(timeseries,axis=0)
	return norm_ts

def check_dimensionality(ts,mixmat):
	"""
	Just checking ts is rightly oriented
	"""
	ntr,_ = mixmat.shape
	ts_shape = ts.shape
	if ts_shape[0] != ntr:
		ts = ts.T

	return ts

def correlate_ts(ts,mixmat):
	"""
	Run (normalised) cross correlation of the signals
	And then selects maximum NCC to decide.
	"""
	_,ncomp = mixmat.shape

	if np.size(ts.shape) != 1:
		_,nts = ts.shape
	else:
		nts = 1

	corr_mtx = np.empty((ncomp,nts))

	for ts_num in range(0,nts):
		for comp_num in range(0,ncomp):
			ncc = np.correlate(mixmat[:,comp_num],ts[:,ts_num],mode='same')
			corr_mtx[comp_num,ts_num] = np.max(ncc)

	return corr_mtx

def select_comp(corr_mtx,thr=0.6):
	"""
	threshold components and then save their position
	"""
	thr_mtx = corr_mtx > thr
	selcomp = [i for i, x in enumerate(t) if x]

	return selcomp

def modify_comp_table(out_dir,mixm,ctab,ts,thr,flag='Accepted'):
	


def _main(argv=None):
    """Tedana entry point"""
    options = _get_parser().parse_args(argv)
    modify_comp_table(options[out_dir],options[mixm],options[ctab],options[good_ts],options[thr],'Accepted')
    modify_comp_table(options[out_dir],options[mixm],options[ctab],options[bad_ts],options[thr],'Rejected')


if __name__ == '__main__':
    _main()
