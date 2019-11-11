from pathlib import Path
from string import Template
from tedana.externals import tempita
from tedana.info import __version__


def _update_template(bokeh_id, bokeh_js):
    """
    Populate a report with content.
    Parameters
    ----------
    title : str
        The title for the report
    docstring : str
        The introductory docstring for the reported object
    content : img
        The content to display
    overlay : img
        Overlaid content, to appear on hover
    parameters : dict
        A dictionary of object parameters and their values
    description : str
        An optional description of the content
    Returns
    -------
    HTMLReport : an instance of a populated HTML report
    """
    resource_path = Path(__file__).resolve().parent.joinpath('data', 'html')

    body_template_name = 'report_body_template.html'
    body_template_path = resource_path.joinpath(body_template_name)
    tpl = tempita.HTMLTemplate.from_filename(str(body_template_path),
                                             encoding='utf-8')
    body = tpl.substitute(version=__version__,
                          bokeh_id=bokeh_id,
                          bokeh_js=bokeh_js)

    head_template_name = 'report_head_template.html'
    head_template_path = resource_path.joinpath(head_template_name)
    with open(str(head_template_path), 'r') as head_file:
        head_tpl = Template(head_file.read())

    html = head_tpl.substitute(body=body)
    return html
