"""Documentation to be injected into docstrings."""

import sys

###################################
# Standard documentation entries
docdict = dict()

docdict[
    "selector"
] = """
selector : :obj:`tedana.selection.component_selector.ComponentSelector`
    The selector to perform decision tree-based component selection with.
"""

docdict[
    "if_true"
] = """
if_true : :obj:`str`
    If the condition in this step is True, give the component classification this
    label. Use 'nochange' if no label changes are desired.
"""

docdict[
    "if_false"
] = """
if_false : :obj:`str`
    If the condition in this step is False, give the component classification this
    label. Use 'nochange' to indicate if no label changes are desired.
"""

docdict[
    "decide_comps"
] = """
decide_comps : :obj:`str` or :obj:`list[str]`
    What classification(s) to operate on. using default or
    intermediate_classification labels. For example: decide_comps='unclassified'
    means to operate only on unclassified components. Use 'all' to include all
    components.
"""

docdict[
    "log_extra_info"
] = """
log_extra_info : :obj:`str`
    Additional text to the information log. Default="".
"""

docdict[
    "only_used_metrics"
] = """
only_used_metrics : :obj:`bool`
    If True, only return the component_table metrics that would be used. Default=False.
"""

docdict[
    "custom_node_label"
] = """
custom_node_label : :obj:`str`
    A short label to describe what happens in this step. If "" then a label is
    automatically generated. Default="".
"""

docdict[
    "tag_if_true"
] = """
tag_if_true : :obj:`str`
    The classification tag to apply if a component is classified True. Default="".
"""

docdict[
    "tag_if_false"
] = """
tag_if_false : :obj:`str`
    The classification tag to apply if a component is classified False. Default="".
"""

docdict[
    "selector"
] = """
selector : :obj:`~tedana.selection.component_selector.ComponentSelector`
    If only_used_metrics is False, the updated selector is returned.
"""

docdict[
    "used_metrics"
] = """
used_metrics : :obj:`set(str)`
    If only_used_metrics is True, the names of the metrics used in the
    function are returned.
"""

docdict_indented = {}


def _indentcount_lines(lines):
    """Minimum indent for all lines in line list.

    >>> lines = [' one', '  two', '   three']
    >>> _indentcount_lines(lines)
    1
    >>> lines = []
    >>> _indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> _indentcount_lines(lines)
    1
    >>> _indentcount_lines(['    '])
    0
    """
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{str(exp)}")
    return f
