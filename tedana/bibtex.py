"""Utilities for managing the tedana bibliography."""

import logging
import os.path as op
import re

import numpy as np
import pandas as pd

from tedana.utils import get_resource_path

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def find_braces(string):
    """Search a string for matched braces.

    This is used to identify pairs of braces in BibTeX files.
    The outside-most pairs should correspond to BibTeX entries.

    Parameters
    ----------
    string : :obj:`str`
        A long string to search for paired braces.

    Returns
    -------
    : obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces.
    """
    toret = {}
    pstack = []

    for idx, char in enumerate(string):
        if char == "{":
            pstack.append(idx)
        elif char == "}":
            if len(pstack) == 0:
                raise IndexError(f"No matching closing parens at: {idx}")

            toret[pstack.pop()] = idx

    if len(pstack) > 0:
        raise IndexError(f"No matching opening parens at: {pstack.pop()}")

    toret = list(toret.items())
    return toret


def reduce_idx(idx_list):
    """Identify outermost brace indices in list of indices.

    The purpose here is to find the brace pairs that correspond to BibTeX entries,
    while discarding brace pairs that appear within the entries
    (e.g., braces around article titles).

    Parameters
    ----------
    idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces.

    Returns
    -------
    reduced_idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces corresponding to BibTeX entries.
    """
    idx_list2 = [idx_item[0] for idx_item in idx_list]
    idx = np.argsort(idx_list2)
    idx_list = [idx_list[i] for i in idx]

    df = pd.DataFrame(data=idx_list, columns=["start", "end"])

    good_idx = []
    df["within"] = False
    for i, row in df.iterrows():
        df["within"] = df["within"] | ((df["start"] > row["start"]) & (df["end"] < row["end"]))
        if not df.iloc[i]["within"]:
            good_idx.append(i)

    idx_list = [idx_list[i] for i in good_idx]
    return idx_list


def index_bibtex_identifiers(string, idx_list):
    """Identify the BibTeX entry identifier before each entry.

    The purpose of this function is to take the raw BibTeX string and a list of indices of entries,
    starting and ending with the braces of each entry, and then extract the identifier before each.

    Parameters
    ----------
    string : :obj:`str`
        The full BibTeX file, as a string.
    idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces corresponding to BibTeX entries.

    Returns
    -------
    idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of BibTeX entries,
        from the starting @ to the final }.
    """
    at_idx = [(a.start(), a.end() - 1) for a in re.finditer("@[a-zA-Z0-9]+{", string)]
    df = pd.DataFrame(at_idx, columns=["real_start", "false_start"])
    df2 = pd.DataFrame(idx_list, columns=["false_start", "end"])
    df = pd.merge(left=df, right=df2, left_on="false_start", right_on="false_start")
    new_idx_list = list(zip(df.real_start, df.end))
    return new_idx_list


def find_citations(description):
    r"""Find citations in a text description.

    It looks for cases of \\citep{} and \\cite{} in a string.

    Parameters
    ----------
    description : :obj:`str`
        Description of a method, optionally with citations.

    Returns
    -------
    all_citations : :obj:`list` of :obj:`str`
        A list of all identifiers for citations.
    """
    paren_citations = re.findall(r"\\citep{([a-zA-Z0-9,_/\.]+)}", description)
    intext_citations = re.findall(r"\\cite{([a-zA-Z0-9,_/\.]+)}", description)
    inparen_citations = re.findall(r"\\citealt{([a-zA-Z0-9,_/\.]+)}", description)
    all_citations = ",".join(paren_citations + intext_citations + inparen_citations)
    all_citations = all_citations.split(",")
    all_citations = sorted(list(set(all_citations)))
    return all_citations


def reduce_references(citations, reference_list):
    """Reduce the list of references to only include ones associated with requested citations.

    Parameters
    ----------
    citations : :obj:`list` of :obj:`str`
        A list of all identifiers for citations.
    reference_list : :obj:`list` of :obj:`str`
        List of all available BibTeX entries.

    Returns
    -------
    reduced_reference_list : :obj:`list` of :obj:`str`
        List of BibTeX entries for citations only.
    """
    reduced_reference_list = []
    for citation in citations:
        citation_found = False
        for reference in reference_list:
            check_string = "@[a-zA-Z]+{" + citation + ","
            if re.match(check_string, reference):
                reduced_reference_list.append(reference)
                citation_found = True
                continue

        if not citation_found:
            LGR.warning(f"Citation {citation} not found.")

    return reduced_reference_list


def get_description_references(description):
    """Find BibTeX references for citations in a methods description.

    Parameters
    ----------
    description : :obj:`str`
        Description of a method, optionally with citations.

    Returns
    -------
    bibtex_string : :obj:`str`
        A string containing BibTeX entries, limited only to the citations in the description.
    """
    bibtex_file = op.join(get_resource_path(), "references.bib")
    with open(bibtex_file) as fo:
        bibtex_string = fo.read()

    braces_idx = find_braces(bibtex_string)
    red_braces_idx = reduce_idx(braces_idx)
    bibtex_idx = index_bibtex_identifiers(bibtex_string, red_braces_idx)
    citations = find_citations(description)
    reference_list = [bibtex_string[start : end + 1] for start, end in bibtex_idx]
    reduced_reference_list = reduce_references(citations, reference_list)

    bibtex_string = "\n".join(reduced_reference_list)
    return bibtex_string
