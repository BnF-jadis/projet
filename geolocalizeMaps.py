import json, glob, os, gc, tqdm
import numpy as np

from utils.utils import getProjectName, getImageName, getPathsToProcess
from utils.match import computeMapExportIF, loadAnchorIF, computeTransform

print('\n','Description :')
print('Géolocalisation des cartes par réalignement avec l\'ancre et entre les cartes.', '\n')


project_name = getProjectName()

with open(os.path.join('settings', project_name, 'settings.json')) as settings:
    settings = json.load(settings)


computeMapExportIF(export_path = os.path.join('export', project_name, 'segmented'), 
                   save_IF_path = os.path.join('save', project_name, 'IF'))

computeMapExportIF(export_path = os.path.join('export', project_name, 'anchor'), 
                   save_IF_path = os.path.join('save', project_name, 'IF'), anchor = True)


feat2, dir2, cent2, shape2 = loadAnchorIF(folder = os.path.join('save', project_name, 'IF'))

gc.collect()

paths = getPathsToProcess(os.path.join('save', project_name, 'IF'), 
                          os.path.join('save', project_name, 'match', 'primary'))

thresholds_list, lowes_list, score_list = [], [], []

for path in tqdm.tqdm(paths, desc='Appariement des cartes à l\'ancre actuelle'):
    if not('anchor' in path):
        with open(path) as data:
            data = json.load(data)
            feat1, dir1 = np.asarray(data['feat']), np.asarray(data['dir'])
            cent1, shape1 = np.asarray(data['cent']), data['shape']
            name = getImageName(path)

            if len(data['cent']) > 1:
                result = computeTransform(feat1, feat2, cent1, cent2, dir1, dir2, shape1, name, 'anchor', settings, primary = True)
            else:
                result = {'dst': [], 'score': 999, 'subscore': [333, 333, 333],
                         'threshold': 1000, 'lowes': 1, 'M': [], "deformation_coef": 1., 
                          "shape": shape1, "on": "anchor"}

            with open(os.path.join('save', project_name, 'match', 'primary', name + '.json'), 'w') as outfile:
                json.dump(result, outfile)

            gc.collect()

paths = glob.glob(os.path.join('save', project_name, 'match', 'primary', '*.json'))

anchored, secondary_anchors = [], []
scores, proj = [], []

for path in paths:
  with open(path) as data:
    data = json.load(data)
    if data['score'] <= settings['matching']['anchoring_score']:
      anchored.append(1)
      if data['score'] <= settings['matching']['secondary_anchor_score']:
        secondary_anchors.append(1)
      else:
        secondary_anchors.append(0)
    else:
      anchored.append(0)
      secondary_anchors.append(0)
    scores.append(data['score'])
    proj.append(data['dst'])

scores = np.asarray(scores)
proj = np.asarray(proj)


similarity_matrix = np.load(os.path.join('save', project_name, 'network_matrix', 'matrix.npy'), allow_pickle=True)
indice = np.arange(len(similarity_matrix))
secondary_anchors_id = indice[secondary_anchors]
IF_save_path = os.path.join('save', project_name, 'IF')
output_path = os.path.join('save', project_name, 'match', 'secondary')

for i, row in tqdm.tqdm(enumerate(similarity_matrix), desc='Appariement des cartes aux ancres secondaires'):
  name1 = getImageName(paths[i])

  if (anchored[i] != 1) and not(os.path.isfile(os.path.join(output_path, name1 + '.json'))):

    pathIF1 = os.path.join(IF_save_path, name1 + '.json')

    with open(pathIF1) as IF1:
      IF1 = json.load(IF1)
      feat1, dir1 = np.asarray(IF1['feat']), np.asarray(IF1['dir'])
      cent1, shape1 = np.asarray(IF1['cent']), IF1['shape']

    best_sec_anchors = secondary_anchors_id[np.flip(np.argpartition(row, (-1, -10)))[:10]]
    best_score,  best_result = np.inf, {}

    for j in best_sec_anchors:
      name2 = getImageName(paths[j])
      pathIF2 = os.path.join(IF_save_path, name2 + '.json')

      with open(pathIF2) as IF2:
        IF2 = json.load(IF2)
        feat2, dir2 = np.asarray(IF2['feat']), np.asarray(IF2['dir'])
        cent2, shape2 = np.asarray(IF2['cent']), IF2['shape']

      result = computeTransform(feat1, feat2, cent1, cent2, dir1, dir2, shape1, name1, name2, settings, primary = False)

      if result['score'] < best_score:
        best_score = result['score']
        best_result = result

      gc.collect()

    with open(os.path.join(output_path, name1 + '.json'), 'w') as outfile:
      json.dump(best_result, outfile)



print('__________________________________________________________________________ \n')

