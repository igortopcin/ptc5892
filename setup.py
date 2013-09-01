try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'PTC5892 Kuan Filter',
    'author': 'Igor Topcin',
    'url': 'https://github.com/igortopcin/ptc5892',
    'download_url': 'https://github.com/igortopcin/ptc5892',
    'author_email': 'topcin@ime.usp.br',
    'version': '0.1',
    'install_requires': [
        'nose',
        'PIL',
        'numpy',
        'scipy',
        'matplotlib' ],
    'packages': ['NAME'],
    'scripts': [],
    'name': 'kuan'
}

setup(**config)

