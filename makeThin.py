import argparse, glob, cv2, tqdm, os, json

from utils.thin import thinImage
from utils.utils import *


print('\n','Description :')
print('Calcule le réseau viaire filaire, et l\'exporte dans le format choisi.')

print('\n','Options :')
print('--toPNG : export au format PNG')
print('--toJSON : export au format JSON')
print('--toSHP : export au format Shapefile')
print('--toSVG : export au format SVG')
print('--geoloc: géolocalise l\'export JSON (si la géolocalisation de la carte est disponible)', '\n')

parser = argparse.ArgumentParser()

parser.add_argument('--toPNG', action="store_true", dest="toPNG", default=False)
parser.add_argument('--toJSON', action="store_true", dest="toJSON", default=False)
parser.add_argument('--toSHP', action="store_true", dest="toSHP", default=False)
parser.add_argument('--toSVG', action="store_true", dest="toSVG", default=False)
parser.add_argument('--geoloc', action="store_true", dest="geoloc", default=False)

options = parser.parse_args()

if (not(options.toPNG) and not(options.toJSON) and not(options.toSVG) and not(options.toSVG)):
	raise ValueError("Vous n'avez pas spécifié de format d'export. \n")

project_name = getProjectName()

paths = getPathsToProcess(os.path.join('export', project_name, 'segmented'), 
                          os.path.join('export', project_name, 'vectorized'))

for path in tqdm.tqdm(paths, desc='Squelettisation des images'):
    
    image = cv2.imread(path, 0)
    name = getImageName(path)
    
    simple_segments, full_segments, nodes_grid = thinImage(image, name, path.replace('segmented', 'vectorized'), 
    	exportSVG = options.toSVG, exportJSON = options.toJSON, exportPNG = options.toPNG, exportSHP = options.toSHP,
    	geoloc = options.geoloc)



print('__________________________________________________________________________ \n')



