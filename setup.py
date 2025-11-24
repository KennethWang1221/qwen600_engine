"""
Setup script for Qwen3-0.6B CUDA Inference Engine Python package.

Installation:
    pip install -e .  # Development mode
    pip install .     # Regular installation
"""

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(Extension):
    """Custom extension that uses CMake to build."""
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build extension that runs CMake."""
    
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build this package. "
                "Please install CMake 3.18+ and try again."
            )
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # CMake configuration arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        
        # Build arguments
        build_args = ['--config', 'Release', '--', '-j4']
        
        # Create build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Run CMake
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )
        
        # Build
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )


# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Qwen3-0.6B CUDA Inference Engine"


setup(
    name='qwen-cuda-engine',
    version='2.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='High-performance CUDA inference engine for Qwen3-0.6B',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/qwen600_engine',
    
    # Python package
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    
    # C++ extension
    ext_modules=[CMakeExtension('qwen_engine._qwen_core')],
    cmdclass={'build_ext': CMakeBuild},
    
    # Requirements
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
    ],
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'mypy>=0.900',
        ],
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
    ],
    
    # Keywords
    keywords='llm inference cuda gpu qwen transformer deep-learning',
    
    # Include additional files
    include_package_data=True,
    zip_safe=False,
)

