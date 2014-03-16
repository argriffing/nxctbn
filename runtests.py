#!/usr/bin/env python
"""
runtests.py

Run tests, building the project first.
This is from the scipy runtests.py script.

"""

from __future__ import division, print_function, absolute_import


PROJECT_MODULE = 'nxctbn'
PROJECT_ROOT_FILES = ['nxctbn', 'setup.py']

import sys
import os
import argparse

# If we are run from the source directory
# we do not want to import the project from there.
sys.path.pop(0)

import shutil
import subprocess

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def main(args):

    # Build the project.
    site_dir = build_project()
    sys.path.insert(0, site_dir)
    os.environ['PYTHONPATH'] = site_dir

    # Define the test directory.
    test_dir = os.path.join(ROOT_DIR, 'build', 'test')

    __import__(PROJECT_MODULE)
    test = sys.modules[PROJECT_MODULE].test

    # Run the tests under build/test
    try:
        shutil.rmtree(test_dir)
    except OSError:
        pass
    try:
        os.makedirs(test_dir)
    except OSError:
        pass

    cwd = os.getcwd()
    try:
        os.chdir(test_dir)
        result = test(args.mode)
    finally:
        os.chdir(cwd)

    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)


def build_project():
    """
    Build a dev version of the project.
    Returns the site-packages directory where it was installed.
    """
    root_ok = [os.path.exists(os.path.join(ROOT_DIR, fn))
            for fn in PROJECT_ROOT_FILES]
    if not all(root_ok):
        print('To build the project, run runtests.py in '
                'git checkout or unpacked source')
        sys.exit(1)

    dst_dir = os.path.join(ROOT_DIR, 'build', 'testenv')
    env = dict(os.environ)
    cmd = [sys.executable, 'setup.py']
    cmd += ['install', '--prefix=' + dst_dir]
    log_filename = os.path.join(ROOT_DIR, 'build.log')
    print('Building, see build.log...')
    with open(log_filename, 'w') as log:
        p = subprocess.Popen(
                cmd, env=env, stdout=log, stderr=log, cwd=ROOT_DIR)
        ret = p.wait()

    if ret == 0:
        print('Build OK')
    else:
        with open(log_filename, 'r') as f:
            print(f.read())
        print('Bulid failed!')

    from distutils.sysconfig import get_python_lib
    site_dir = get_python_lib(prefix=dst_dir, plat_specific=True)

    return site_dir


#TODO GCOV

#TODO LCOV

#TODO python 3 support


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', default='fast',
            help="'fast', 'full', or something that could be passed "
                    "to nosetests -A [default: fast]")
    main(parser.parse_args())

