from distutils.core import setup

setup(
    name='atomics',
    version='1',
    packages=[
        'atomics',
    ],
    install_requires=[
        'numpy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'openmdao',
        'pint',
        'guppy3',
        'sphinx-rtd-theme',
        'sphinx-code-include',
        'jupyter-sphinx',
        'numpydoc',
    ],
)
