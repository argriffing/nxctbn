from __future__ import division, print_function, absolute_import

from numpy.testing import run_module_suite, decorators, assert_equal

def test_pass_me():
    assert_equal(42, 42.0)

@decorators.skipif(True, 'skipping a dummy test')
def test_skip_me():
    assert_equal(0, 1)

@decorators.knownfailureif(True, 'known failing a dummy test')
def test_known_fail_me():
    assert_equal(0, 1)
