from pathlib import Path
from html import unescape
from string import Template
from tedana.info import __version__
from tedana.externals import tempita


def _update_template_about(call, methods):
    """
    Populate a report with content.

    Parameters
    ----------
    call : str
        Call used to execute tedana
    methods : str
        Generated methods for specific tedana call
    Returns
    -------
    HTMLReport : an instance of a populated HTML report
    """
    resource_path = Path(__file__).resolve().parent.joinpath('data', 'html')
    body_template_name = 'report_body_template.html'
    body_template_path = resource_path.joinpath(body_template_name)
    tpl = tempita.HTMLTemplate.from_filename(str(body_template_path),
                                             encoding='utf-8')
    subst = tpl.substitute(content=methods,
                           javascript=None)
    body = unescape(subst)
    return body


def _update_template_bokeh(bokeh_id, bokeh_js):
    """
    Populate a report with content.

    Parameters
    ----------
    bokeh_id : str
        HTML div created by bokeh.embed.components
    bokeh_js : str
        Javascript created by bokeh.embed.components
    Returns
    -------
    HTMLReport : an instance of a populated HTML report
    """
    resource_path = Path(__file__).resolve().parent.joinpath('data', 'html')

    body_template_name = 'report_body_template.html'
    body_template_path = resource_path.joinpath(body_template_name)
    tpl = tempita.HTMLTemplate.from_filename(str(body_template_path),
                                             encoding='utf-8')
    subst = tpl.substitute(content=bokeh_id,
                           javascript=bokeh_js)
    body = unescape(subst)
    return body


def _save_as_html(body):
    """
    Save an HTML report out to a file.

    Parameters
    ----------
    body : str
        Body for HTML report with embedded figures
    """
    resource_path = Path(__file__).resolve().parent.joinpath('data', 'html')
    head_template_name = 'report_head_template.html'
    head_template_path = resource_path.joinpath(head_template_name)
    with open(str(head_template_path), 'r') as head_file:
        head_tpl = Template(head_file.read())

    html = head_tpl.substitute(version=__version__, body=body)
    return html


def generate_report(bokeh_id, bokeh_js, file_path=None):
    """
    Generate and save an HTML report.

    Parameters
    ----------
    bokeh_id : str
        HTML div created by bokeh.embed.components
    bokeh_js : strs
        Javascript created by bokeh.embed.components
    file_path : str
        The file path on disk to write the HTML report

    Returns
    -------
    HTML : file
        A generated HTML report
    """
    body = _update_template_bokeh(bokeh_id, bokeh_js)
    html = _save_as_html(body)

    if file_path is not None:
        with open(file_path, 'wb') as f:
            f.write(html.encode('utf-8'))
    else:
        with open('./tedana_report.html', 'wb') as f:
            f.write(html.encode('utf-8'))
