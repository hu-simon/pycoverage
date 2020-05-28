import os
import sys

from setuptools import setup
from distutils.sysconfig import get_python_lib

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)

# Check to make sure that the user's Python version is compatible.
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write(
        """
        ==========================
        UNSUPPORTED PYTHON VERSION
        ==========================
        
        This version of pycoverage requires Python {}.{} but you are attempting to install it on Python {}.{}.

        You may also be using a version of pip that doesn't understand the python_requires classifier. Make sure that you have pip >= 9.0 and setuptools >= 24.2, and try installing the package again. The following commands allow you to do this.

        $ python -m pip install --upgrade pip setuptools
        $ pip install -e .

        This will install the latest version of pip, setuptools, and pycoverage. Older versions of pycoverage may be available but are at end of life support. 
        """.format(
            *(REQUIRED_PYTHON + CURRENT_PYTHON)
        )
    )
    sys.exit(1)

# Install pycoverage.
setup(
    name="pycoverage",
    version="0.2",
    description="Package containing utilities to handle coverage control algorithms. These tools were developed for the MinAU project.",
    url="https://nodes.ucsd.edu/svn/repos/codes/2019d_Coverage-python",
    author="Jun Hao (Simon) Hu",
    author_email="simonhu@ieee.org",
    license="MPL-2.0",
    packages=[
        "pycoverage",
        "pycoverage.vorutils",
        "pycoverage.simutils",
        "pycoverage.visutils",
        "pycoverage3d",
        "pycoverage3d.vorutils",
    ],
    zip_safe=False,
)
