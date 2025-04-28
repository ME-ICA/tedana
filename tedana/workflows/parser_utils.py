"""Functions for parsers."""

import argparse
import os.path as op

from tedana.config import MAX_N_ROBUST_RUNS, MIN_N_ROBUST_RUNS
from tedana.io import fname_to_component_list, str_to_component_list


def check_tedpca_value(value, is_parser=True):
    """
    Check tedpca argument.

    Check if argument is a float in range (0,1),
    a positive integer,
    or one of a list of strings.
    """
    valid_options = ("mdl", "aic", "kic", "kundu", "kundu-stabilize")
    if value in valid_options:
        return value

    error = argparse.ArgumentTypeError if is_parser else ValueError
    dashes = "--" if is_parser else ""

    def check_float(floatvalue):
        assert isinstance(floatvalue, float)
        if not (0.0 < floatvalue < 1.0):
            msg = f"Floating-point argument to {dashes}tedpca must be in the range (0.0, 1.0)."
            raise error(msg)
        return floatvalue

    def check_int(intvalue):
        assert isinstance(intvalue, int)
        if intvalue < 1:
            msg = f"Integer argument to {dashes}tedpca must be positive"
            raise error(msg)
        elif intvalue == 1:
            # Assume user meant 100% (1.0), as it is highly unlikely they meant to get a
            # single component
            return 1.0
        return intvalue

    if isinstance(value, float):
        return check_float(value)
    if isinstance(value, int):
        return check_int(value)

    try:
        floatvalue = float(value)
    except ValueError:
        msg = (
            f"Argument to {dashes}tedpca must be either a number, "
            f"or one of: {', '.join(valid_options)}"
        )
        raise error(msg)

    try:
        intvalue = int(value)
    except ValueError:
        return check_float(floatvalue)
    return check_int(intvalue)


def check_n_robust_runs_value(string, is_parser=True):
    """
    Check n_robust_runs argument.

    Check if argument is an int between MIN_N_ROBUST_RUNS  and MAX_N_ROBUST_RUNS.
    """
    error = argparse.ArgumentTypeError if is_parser else ValueError
    try:
        intarg = int(string)
    except ValueError:
        msg = (
            f"Argument to n_robust_runs must be an integer "
            f"between {MIN_N_ROBUST_RUNS} and {MAX_N_ROBUST_RUNS}."
        )
        raise error(msg)

    if not (MIN_N_ROBUST_RUNS <= intarg <= MAX_N_ROBUST_RUNS):
        raise error(
            f"n_robust_runs must be an integer between {MIN_N_ROBUST_RUNS} "
            f"and {MAX_N_ROBUST_RUNS}."
        )
    else:
        return intarg


def is_valid_file(parser, arg):
    """Check if argument is existing file."""
    if not op.isfile(arg) and arg is not None:
        parser.error(f"The file {arg} does not exist!")

    return arg


def parse_manual_list_int(manual_list):
    """
    Parse the list of components to accept or reject into a list of integers.

    Parameters
    ----------
    manual_list : :obj:`str` :obj:`list[str]` :obj:`int` :obj:`list[int]` or [] or None
        String of integers separated by commas
        A file name for a file that contains integers

    Returns
    -------
    manual_nums : :obj:`list[int]`
        A list of integers or an empty list.

    Note
    ----
    Do not need to check if integers are less than 0 or greater than the total
    number of components here, because it is later checked in selectcomps2use
    and a descriptive error message will appear there
    """
    if isinstance(manual_list, str):
        manual_list = [manual_list]

    if not manual_list:
        manual_nums = []
    elif op.exists(op.expanduser(str(manual_list[0]).strip(" "))):
        # filename was given
        manual_nums = fname_to_component_list(op.expanduser(str(manual_list[0]).strip(" ")))
    elif len(manual_list) == 1 and isinstance(manual_list[0], str) and "," in manual_list[0]:
        # Break up a comma delimited list into multiple values
        possible_list = manual_list[0].split(",")
        manual_list = [s.strip() for s in possible_list]

    if "manual_nums" not in locals():
        if len(manual_list) > 1:
            # Assume that this is a list of integers, but raise error if not
            manual_nums = []
            for x in manual_list:
                if float(x) == int(float(x)):
                    manual_nums.append(int(x))
                else:
                    raise ValueError(
                        "parse_manual_list_int expected a list of integers, "
                        f"but the input is {manual_list}"
                    )
        elif isinstance(manual_list[0], str):
            # arbitrary string was given, length of list is 1
            manual_nums = str_to_component_list(manual_list[0])
        elif isinstance(manual_list[0], int):
            # Is a single integer and should remain a list with a single integer
            manual_nums = manual_list
        else:
            raise ValueError(
                "parse_manual_list_int expected integers or a filename, "
                f"but the input is {manual_list}"
            )

    return manual_nums


def parse_manual_list_str(manual_list):
    """
    Parse the list of components tags into a comma delimited list of strings.

    Parameters
    ----------
    manual_list : :obj:`str` :obj:`list[str]` or [] or None
        Strings (classification tags) separated by commas

    Returns
    -------
    manual_vals : :obj:`str`
        A comma delimited

    Note
    ----
    Unlike _parse_manual_list_int, only ',' is a permitted delimiter.
    Classification tags can use spaces so that cannot be a delimiter.
    Classification tags cannot include commas.
    Those strings would be split at other points in the code.
    """
    if not manual_list:
        manual_vals = []
    elif not isinstance(manual_list, list):
        manual_vals = [manual_list]
    else:
        manual_vals = manual_list

    if len(manual_vals) > 1:
        for x in manual_vals:
            if not isinstance(x, str):
                raise ValueError(
                    "parse_manual_list_str expected a string or a list of strings, "
                    f"but the input is {manual_list}"
                )
            elif "," in x:
                raise ValueError(
                    "parse_manual_list_str includes a comma in a list of multiple strings. "
                    "Input can include a comma delimited string, but not multiple strings. "
                    f"Input is {manual_list}"
                )

    # separate string by commas and remove leading & training whitespace
    if len(manual_vals) == 1:
        if isinstance(manual_vals[0], str):
            possible_list = manual_vals[0].split(",")
            manual_vals = [s.strip() for s in possible_list]
        else:
            raise ValueError(
                "parse_manual_list_str expected a string or a list of strings, "
                f"but the input is {manual_list}"
            )

    # Convert the list of strings back to a single comma delimited string with no trailing spaces
    manual_string = ",".join(str(s).strip() for s in manual_vals)

    return manual_string
