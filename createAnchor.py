from utils.utils import getProjectName
from utils.match import createAnchorMap
import os, json


print('\n','Description :')
print('Création de l\'ancre actuelle.', '\n')


project_name = getProjectName()

with open(os.path.join('settings', project_name, 'settings.json')) as settings:
    settings = json.load(settings)

createAnchorMap(settings)


print('__________________________________________________________________________ \n')
