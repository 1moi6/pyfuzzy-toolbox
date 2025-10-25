"""
Setup script for Fuzzy Systems
"""

from setuptools import setup, find_packages
import os
import sys

# Lê o README para descrição longa
def read_file(filename):
    """Lê conteúdo de um arquivo."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Import version from package
sys.path.insert(0, os.path.dirname(__file__))
from fuzzy_systems import __version__
VERSION = __version__

# Dependências principais
INSTALL_REQUIRES = [
    'numpy>=1.20.0',
    'matplotlib>=3.3.0',
    'scipy>=1.6.0',
]

# Dependências opcionais
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=6.0',
        'pytest-cov>=2.12',
        'black>=21.0',
        'flake8>=3.9',
        'mypy>=0.910',
    ],
    'docs': [
        'sphinx>=4.0',
        'sphinx-rtd-theme>=0.5',
        'nbsphinx>=0.8',
    ],
    'ml': [
        'scikit-learn>=0.24',
        'pandas>=1.2',
    ],
}

# Adiciona 'all' que instala todas as dependências extras
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name='fuzzy-systems',  # Nome no PyPI (com hífen)
    version=VERSION,
    author='Fuzzy Systems Contributors',
    author_email='fuzzy.systems@example.com',
    description='Comprehensive Fuzzy Logic library: Inference, Learning, Dynamics',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/fuzzy-systems',  # TODO: Atualizar com URL real do GitHub
    packages=find_packages(exclude=['tests', 'docs']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    keywords='fuzzy logic, fuzzy inference, mamdani, sugeno, anfis, wang-mendel, machine learning, fuzzy ode, p-fuzzy, control systems',
    python_requires='>=3.8',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
    license='MIT',
    project_urls={
        'Homepage': 'https://github.com/yourusername/fuzzy-systems',
        'Bug Tracker': 'https://github.com/yourusername/fuzzy-systems/issues',
        'Source Code': 'https://github.com/yourusername/fuzzy-systems',
        'Changelog': 'https://github.com/yourusername/fuzzy-systems/blob/main/CHANGELOG.md',
        'Documentation': 'https://github.com/yourusername/fuzzy-systems#readme',
    },
)
