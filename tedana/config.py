"""Setting default values for ICA decomposition."""

DEFAULT_ICA_METHOD = "robustica"
DEFAULT_N_ROBUST_RUNS = 30
DEFAULT_N_MAX_ITER = 500
DEFAULT_N_MAX_RESTART = 10
DEFAULT_SEED = 42


"""Setting extreme values for number of robust runs."""

MIN_N_ROBUST_RUNS = 5
MAX_N_ROBUST_RUNS = 500
WARN_N_ROBUST_RUNS = 200


"""Setting the warning threshold for the index quality."""

WARN_IQ = 0.6
