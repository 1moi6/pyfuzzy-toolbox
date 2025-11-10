"""
Setup script for Fuzzy Systems
"""

from setuptools import setup, find_packages
import os

# Lê o README para descrição longa
def read_file(filename):
    """Lê conteúdo de um arquivo."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Lê a versão sem importar o módulo (evita dependências no setup)
def get_version():
    """Extrai __version__ do __init__.py sem importar o módulo."""
    version_file = os.path.join(os.path.dirname(__file__), 'fuzzy_systems', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                # Extrai a string entre aspas
                return line.split('=')[1].strip().strip('"').strip("'")
    raise RuntimeError('Unable to find version string.')

VERSION = get_version()

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
        'joblib>=1.0',
    ],
    'ui': [
        'streamlit>=1.20.0',
        'plotly>=5.0.0',
        'typer[all]>=0.9.0',
        'rich>=13.0.0',
    ],
}

# Adiciona 'all' que instala todas as dependências extras
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name='pyfuzzy-toolbox',  # Nome no PyPI (com hífen)
    version=VERSION,
    author='Moiseis Cecconello',
    author_email='moiseis@gmail.com',
    description='A comprehensive Python library for fuzzy systems: inference (Mamdani, Sugeno), learning (ANFIS, Wang-Mendel), and dynamics (p-fuzzy, fuzzy ODEs)',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/1moi6/pyfuzzy-toolbox',
    packages=find_packages(exclude=['tests', 'docs']),
    package_data={
        'fuzzy_systems': ['streamlit_app/**/*.py', 'streamlit_app/**/*.toml', 'streamlit_app/**/*.md'],
    },
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
    entry_points={
        'console_scripts': [
            'pyfuzzy=fuzzy_systems.cli:app',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    license='MIT',
    project_urls={
        'Homepage': 'https://github.com/1moi6/pyfuzzy-toolbox',
        'Bug Tracker': 'https://github.com/1moi6/pyfuzzy-toolbox/issues',
        'Source Code': 'https://github.com/1moi6/pyfuzzy-toolbox',
        'Changelog': 'https://github.com/1moi6/pyfuzzy-toolbox/blob/main/CHANGELOG.md',
        'Documentation': 'https://github.com/1moi6/pyfuzzy-toolbox#readme',
    },
)
