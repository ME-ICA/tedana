#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
""" tedana setup script """


def main():
    """ Install entry-point """
    from io import open
    from os import path as op
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages

    this_path = op.dirname(op.abspath(getfile(currentframe())))

    # For Python 3: use a locals dictionary
    # http://stackoverflow.com/a/1463370/6820620
    ldict = locals()
    # Get version and release info, which is all stored in tedana/info.py
    module_file = op.join(this_path, 'tedana', 'info.py')
    with open(module_file) as infofile:
        pythoncode = [line for line in infofile.readlines() if not
                      line.strip().startswith('#')]
        exec('\n'.join(pythoncode), globals(), ldict)

    setup(
        name=ldict['__packagename__'],
        version=ldict['__version__'],
        description=ldict['__description__'],
        long_description=ldict['__longdesc__'],
        author=ldict['__author__'],
        author_email=ldict['__email__'],
        maintainer=ldict['__maintainer__'],
        maintainer_email=ldict['__email__'],
        url=ldict['__url__'],
        license=ldict['__license__'],
        classifiers=ldict['CLASSIFIERS'],
        download_url=ldict['DOWNLOAD_URL'],
        # Dependencies handling
        install_requires=ldict['REQUIRES'],
        tests_require=ldict['TESTS_REQUIRES'],
        extras_require=ldict['EXTRA_REQUIRES'],
        entry_points={'console_scripts': [
            't2smap=tedana.cli.run_t2smap:main',
            'tedana=tedana.cli.run_tedana:main'
        ]},
        packages=find_packages(exclude=("tests",)),
        zip_safe=False,
    )


if __name__ == '__main__':
    main()
