import json, glob, os, gc, tqdm, cv2

from utils.utils import *
from utils.segment import makeImagePatches, dhSegmentPrediction, reconstituteMap, postProcessing

print('\n','Description :')
print('Segmentation des cartes à partir du modèle entraîné.', '\n')


project_name = getProjectName()

with open(os.path.join('settings', project_name, 'settings.json')) as settings:
    settings = json.load(settings)

classes = settings['segment']['classes']
workshop_image = os.path.join('workshop', 'images')
workshop_prediction = os.path.join('workshop', 'prediction')

maps_paths = getPathsToProcess(os.path.join('data', project_name, 'maps'), os.path.join('export', project_name, 'segmented'))

for path in tqdm.tqdm(maps_paths, desc='Maps segmented'):

    try:
        resetFolders([workshop_image, workshop_prediction])
    except:
        os.makedirs(workshop_image, exist_ok=True)
        os.makedirs(workshop_prediction, exist_ok=True)

    image = cv2.imread(path, 1)
    rows, cols = makeImagePatches(image = image, patches_path = workshop_image)

    dhSegmentPrediction(os.path.join('model', settings['segment']['model'], 'export'), 
                        workshop_image, workshop_prediction, classes)
    
    prediction_map = reconstituteMap(image = image, prediction_path = workshop_prediction, 
                                     classes = classes, rows = rows, cols = cols)

    postprocessed_map = postProcessing(prediction_map, component_min_area = 100)

    try:
        assert len(postprocessed_map.shape) == 3
    except:
        postprocessed_map = grayToColor(prediction_map, postprocessed_map)

    cv2.imwrite(os.path.join('export', project_name, 'segmented', getImageName(path) + '.png'), postprocessed_map)

    gc.collect()


print('__________________________________________________________________________ \n')

