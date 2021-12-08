"""
© 2021. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
"""

from setuptools import setup, find_packages
__version__ = "1.0.0"

# add readme
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='pyQBTNs',
    version=__version__,
    author='Elijah Pelofske, Hristo Djidjev, Dan O\'Malley, Maksim E. Eren, Boian S. Alexandrov',
    author_email='epelofske@lanl.gov',
    description='pyQBTNs: Python Quantum Boolean Tensor Networks',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    package_dir={'pyQBTNs': 'pyQBTNs/'},
    platforms = ["Linux", "Mac", "Windows"],
    include_package_data=True,
    url='https://github.com/lanl/pyQBTNs',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.9.7', # put the correct python version here
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.9.7', # put the correct python version here
    install_requires=INSTALL_REQUIRES,
    license='License :: BSD3 License',
)
