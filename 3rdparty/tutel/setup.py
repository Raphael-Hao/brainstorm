#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module.

Reference:
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import os, sys
import subprocess
from typing import List, Tuple

from setuptools import setup, find_packages, Command
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION
except:
    IS_HIP_EXTENSION = False

if len(sys.argv) <= 1:
    sys.argv += ['install', '--user']

root_path = os.path.dirname(sys.argv[0])
root_path = root_path if root_path else '.'

os.chdir(root_path)

class Tester(Command):
    """Cmdclass for `python setup.py test`.
    Args:
        Command (distutils.cmd.Command):
            Abstract base class for defining command classes.
    """

    description = 'test the code using pytest'
    user_options: List[Tuple[str, str, str]] = []

    def initialize_options(self):
        """Set default values for options that this command supports."""
        pass

    def finalize_options(self):
        """Set final values for options that this command supports."""
        pass

    def run(self):
        """Run pytest."""
        subprocess.check_call('python3 -m pytest -v -s tests/', shell=True)

def install(use_nccl):
    ext_libs = ['dl', 'cuda', 'nvrtc'] if not IS_HIP_EXTENSION else []
    ext_args = {'cxx': ['-Wno-sign-compare', '-Wno-unused-but-set-variable']}
    if use_nccl:
        if IS_HIP_EXTENSION:
            ext_libs += ['rccl']
        else:
            ext_libs += ['nccl']
        ext_args['cxx'] += ['-DUSE_NCCL']

    setup(
        name='tutel',
        version='0.1',
        description='An Optimized Mixture-of-Experts Implementation.',
        url='https://github.com/microsoft/Tutel',
        author='Microsoft',
        author_email='tutel@microsoft.com',
        license='MIT',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Environment :: GPU',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        keywords=['Mixture of Experts', 'MoE', 'Optimization'],
        packages=find_packages(exclude=['tests']),
        python_requires='>=3.6, <4',
        install_requires=[
        ],
        extras_require={
            'test': [
                'GPUtil>=1.4.0',
                'pytest-subtests>=0.4.0',
                'pytest>=6.2.2',
            ],
        },
        ext_modules=[
            CUDAExtension('tutel_custom_kernel', [
                './tutel/custom/custom_kernel.cpp',
            ],
            library_dirs=['/usr/local/cuda/lib64/stubs'],
            libraries=ext_libs,
            extra_compile_args=ext_args)
        ],
        cmdclass={
            'build_ext': BuildExtension,
            'test': Tester,
        },
        project_urls={
            'Source': 'https://github.com/microsoft/Tutel',
            'Tracker': 'https://github.com/microsoft/Tutel/issues',
        },
    )

try:
    install(use_nccl=True)
except:
    print('Try installing without NCCL extension..')
    install(use_nccl=False)
