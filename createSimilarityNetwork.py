from utils.networking import *
from utils.utils import getProjectName, getImageName
from updateCorpus import updateCorpus
import json, os, glob
import numpy as np
import networkx as nx

print('\n','Description :')
print('Création du réseau de similarité entre les cartes, basé sur les caractéristiques visuelles et les métadonnées.', '\n')

project_name = getProjectName()

print('Pour modifier les poids et les colonnes prises en compte par le réseau de similarité, veuillez vous référer aux paramètres avancés, \
disponibles dans settings/{0}/settings.json, sous l\'entrée "networking".'.format(project_name), '\n')

df_metadata = loadMetadata(os.path.join('data', project_name, 'data.xlsx'))

files = glob.glob(os.path.join('data', project_name, 'maps', '*.*'))

arks, leaflets = [], []
for file in files:
    arks.append(getImageName(file)[:-2])
    leaflets.append(getImageName(file)[-1])
df_files = pd.DataFrame({'path': files, 'ark': arks, 'leaflet': leaflets})
df_files['leaflet'] = df_files['leaflet'].astype('int')

df_metadata = pd.merge(df_metadata, df_files, on=['ark', 'leaflet'], how='inner')

with open(os.path.join('settings', project_name, 'settings.json')) as settings:
    settings = json.load(settings)

similarity_matrix = computeSimilarityMatrix(settings = settings['networking'], df_metadata = df_metadata)

print('Matrice de similarité sauvegardée.')
np.save(os.path.join('save', project_name, 'network_matrix', 'matrix.npy'), similarity_matrix)

graph = createGraph(array = similarity_matrix, N_neighbours = settings['networking']['gephi']['n_neighbours'])
nx.write_gexf(graph, os.path.join('export', project_name, 'gephi'))
print('Réseau exporté au format Gephi')
print(nx.info(graph))


print('__________________________________________________________________________ \n')
