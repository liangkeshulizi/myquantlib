
from setuptools import setup, find_packages

setup(
    name = 'myquantlib',
    version = '0.1',
    packages = find_packages(exclude=['tests*']),
    license = 'MIT',
    description = 'A native and hackable neural network quantization library based on pytorch for research purposes.',
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires = ['pytorch>=2.5'],
    url='',
    author='LIYIZHOU',
    author_email='liangkeshulizi@gmail.com'
)
