from setuptools import setup, find_packages
import sys
if sys.version_info < (2,7):
    sys.exit('Sorry, Python < 2.7 is not supported')

setup(
    name='wsitools',
    version='0.1dev',
    description='Whole slide image processing tools',
    packages=find_packages(),
    author="Jun Jiang",
    author_email="Jiang.Jun@mayo.edu",
    url='https://github.com/smujiang/smujiang.github.io',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    install_requires=[
        'numpy',
        'matplotlib',
        'opencv-python',
        'xlrd',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'openslide-python',
        'shapely',
        'pillow',
        'joblib',
        'tensorflow'
    ]
)
