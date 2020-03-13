import argparse, os, json
from utils.utils import cleanProjectName, updateProjectName

print('\n','Description :')
print('Création d\'un nouveau projet JADIS.')

print('\n','Options :')
print('--cityName : (obligatoire) Nom de la ville', '\n')

parser = argparse.ArgumentParser()

parser.add_argument('--cityName', action="store", dest="cityName")

options = parser.parse_args()

if (options.cityName) == None:
    raise ValueError("Vous n'avez pas spécifié le nom de la ville concernée par votre projet. \n")

project_name = cleanProjectName(options.cityName, is_input = True)

try:
    os.makedirs(os.path.join('data', project_name, 'maps'), exist_ok=False)
    os.makedirs(os.path.join('export', project_name, 'anchor'), exist_ok=False)
    os.makedirs(os.path.join('export', project_name, 'deformation'), exist_ok=False)
    os.makedirs(os.path.join('export', project_name, 'deformed'), exist_ok=False)
    os.makedirs(os.path.join('export', project_name, 'gephi'), exist_ok=False)
    os.makedirs(os.path.join('export', project_name, 'vectorized'), exist_ok=False)
    os.makedirs(os.path.join('export', project_name, 'segmented'), exist_ok=False)
    os.makedirs(os.path.join('export', project_name, 'corrected'), exist_ok=False)
    os.makedirs(os.path.join('save', project_name, 'IF'), exist_ok=False)
    os.makedirs(os.path.join('save', project_name, 'projection'), exist_ok=False)
    os.makedirs(os.path.join('save', project_name, 'match', 'primary'), exist_ok=False)
    os.makedirs(os.path.join('save', project_name, 'match', 'secondary'), exist_ok=False)
    os.makedirs(os.path.join('save', project_name, 'HHOG'), exist_ok=False)
    os.makedirs(os.path.join('save', project_name, 'network_matrix'), exist_ok=False)
    os.makedirs(os.path.join('model', 'train', project_name, 'images'), exist_ok=False)
    os.makedirs(os.path.join('model', 'train', project_name, 'labels'), exist_ok=False)
    os.makedirs(os.path.join('model', 'train', project_name, 'eval', 'images'), exist_ok=False)
    os.makedirs(os.path.join('model', 'train', project_name, 'eval', 'labels'), exist_ok=False)
    os.makedirs(os.path.join('settings', project_name), exist_ok=False)
    
    print('Votre nouveau projet :', project_name, 'a bien été créé. Vous pouvez désormais commencer.')
except:
    print('Un projet à ce nom existe déjà. Veuillez réessayer')

with open(os.path.join('utils', 'settings.json')) as settings:
    settings = json.load(settings)

settings['corpus']['city_name'] = options.cityName

with open(os.path.join('settings', project_name,  'settings.json'), 'w') as outfile:
    json.dump(settings, outfile, indent=2)

updateProjectName(project_name)


print('__________________________________________________________________________ \n')

