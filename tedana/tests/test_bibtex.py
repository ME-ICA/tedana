"""Tests for bibtex."""

from tedana import bibtex


def test_warn_no_citation_found(caplog):
    citations = ["Nonexistent et al, 0 AD"]
    ref_list = []
    bibtex.reduce_references(citations, ref_list)
    assert f"Citation {citations[0]} not found." in caplog.text
