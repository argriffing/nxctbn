#!/usr/bin/env python
"""Simulate branching trajectories on a continuous time Bayesian network (CTBN).

"""

DOCLINES = __doc__.split('\n')

# This setup script is written according to
# http://docs.python.org/2/distutils/setupscript.html
#
# It is meant to be installed through github using pip.

from distutils.core import setup

# This idiom is used by scipy to check if it is running during the setup.
__NXMCTREE_SETUP__ = True

setup(
        name='nxctbn',
        version='0.1',
        description=DOCLINES[0],
        author='alex',
        url='https://github.com/argriffing/nxctbn/',
        download_url='https://github.com/argriffing/nxctbn/',
        packages=['nxctbn'],
        test_suite='nose.collector',
        package_data={'nxctbn' : ['tests/test_*.py']},
        )


