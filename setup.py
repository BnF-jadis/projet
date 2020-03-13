'''
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
	'imutils >= 0.5.3',
	'networkx >= 2.3',
	'numpy >= 1.17.2',
	'opencv-python == 3.4.2.16',
	'opencv-contrib-python == 3.4.2.17',
	'overpass >= 0.7',
	'pandas >= 0.25.2',
	'Pillow >= 6.2.0',
	'pyxDamereauLevenshtein >= 1.5.3',
	'scikit-image >= 0.15.0',
	'scikit-learn >= 0.22.1',
	'tensorflow == 1.13.0'
]

setuptools.setup(
    name="jadis-RPetitpierre", # Replace with your own username
    version="0.0.1",
    author="Remi Petitpierre",
    author_email="remi.petitpierre@epfl.ch",
    description="JADIS Program for automatic geolocalisation of city maps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
'''

import os

os.system('pip install --upgrade pip')
os.system('conda update -n base -c defaults conda')
os.system('conda create -n jadis python=3.6')
os.system('conda activate jadis')
os.system('pip install -r utils/requirements.txt')


