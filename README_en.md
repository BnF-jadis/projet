Ceci est la version anglaise. [**\[Lire en français\]**](README.md)

[![license](https://img.shields.io/badge/license-CC%20BY%203.0%20FR-brightgreen)](https://creativecommons.org/licenses/by/3.0/fr/)
[![open issues](https://isitmaintained.com/badge/open/BnF-jadis/project.svg)](https://github.com/BnF-jadis/projet/issues)
[![DOI](https://zenodo.org/badge/247013885.svg)](https://zenodo.org/badge/latestdoi/247013885)

# JADIS Project

The JADIS project stems from a scientific collaboration between the BnF, French National Library, and EPFL, Swiss Federal Institute of Technology in Lausanne. The objectives of the project are the following:

* Develop an algorithm to automatically geolocate and realign map collections with street-level precision.
* Realign the results to the historical street name database to allow the search of maps of Paris by period street names.

The algorithm has two main axes. The first one aims at automatically vectorizing the maps using a neural network. The second one aims at creating a similarity network to realign the maps between them. The core algorithm identifies geometric similarities between the vectorized maps to realign them on a contemporary reference. 

## Documentation (in French)

* [Installation and User Manual](https://github.com/BnF-jadis/projet/blob/master/Jadis_manuel.pdf). The program is designed to be fully usable by non-programmers.
* [Simplified description of how the segmentation and realignment algorithms work](https://github.com/BnF-jadis/projet/blob/master/documentation_Jadis.pdf).

## Easy installation

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)
2. [Download the JADIS program (https://github.com/BnF-jadis/projet/archive/master.zip), unzip it, and place it in the folder of your choice, e.g. on the desktop.
in the folder of your choice, for example on the Desktop
3. 3. [Download the trained neural network](https://drive.google.com/file/d/13iRsEwFv9tTe68v5d_dXlEAJj9sn0qsb/view?usp=sharing),
unzip it and put it in the main JADIS folder
4.	If necessary, install a compiler for C, such as GCC (https://gcc.gnu.org/install/binaries.html). On Mac, install XCode (https://apps.apple.com/us/app/xcode/id497799835)
5. Open a command prompt. On Windows, you can use the Command Prompt application
application, installed by default and accessible via the search. Alternatively, you can use for example Anaconda prompt. On Linux/Fedora or Mac, use Terminal.
6. You are in one of the folders of your computer, usually the source folder of your account. The name of the folder is shown to the left of your cursor, for example
For example, ``Users\Remi'' or ``~Remi $```. On Unix (Mac, Linux), you can type ``ls`` to list the files in your directory.
7. Navigate to the JADIS folder on your computer. For example, if it is on your
Desktop, the path will probably be Desktop/project-master :
``` cd Desktop/project-master ````
8.	Create a new conda environment for the project: ```` conda create -n jadis python=3.6 ````
9.	Activate your conda environment: ```` conda activate jadis ```. This command must be repeated for each new user session. 
10. Use the setup.py function to install the program:
```python setup.py ````
11. When the command prompt asks if you want to continue, type ```y ``` and then _back to line_.
12. Check your internet connection and try again if the installation fails.

## Licence
CC BY 3.0 FR (Summary here under)

You are permitted to:
* Share - copy, distribute, and communicate the material by any means and in any format
* Adapt - remix, transform and create from the material for any purpose, including commercial use.

This license is acceptable for free cultural works.
* The Licensor may not withdraw the permissions granted by the license as long as you are complying with the terms of this license.

Under the following conditions:
* Attribution - You must credit the Work, link to the license, and indicate whether any modifications have been made to the Work. You must indicate this information by all reasonable means, but not suggest that the Licensor endorses you or the way you have used the Work.
* No Additional Restrictions - You may not enforce any legal requirements or technical measures that would legally restrict others from using the Work under the conditions described in the license.

### Citation
Rémi Petitpierre. (2020). BnF-jadis/projet: v1.0-beta (beta). Zenodo. https://doi.org/10.5281/zenodo.6594483
```
@software{petitpierre_bnf-jadis_2020,
  author       = {R{\'{e}}mi Petitpierre},
  title        = {BnF-jadis/projet: v1.0-beta},
  month        = Mar,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {beta},
  doi          = {10.5281/zenodo.6594483},
  url          = {https://doi.org/10.5281/zenodo.6594483}
}
```

