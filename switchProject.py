import argparse
import os, json, glob
from utils.utils import updateProjectName, cleanProjectName

print('\n','Description :')
print('Changement du projet sélectionné dans l\'espace de travail.')

print('\n','Options :')
print('--cityName : (obligatoire) Nom de la ville', '\n')

parser = argparse.ArgumentParser()

parser.add_argument('--cityName', action="store", dest="cityName")

options = parser.parse_args()

if (options.cityName) == None:
    raise ValueError("\nVous n'avez pas spécifié le nom de la ville concernée par votre projet. \n")

project_paths = glob.glob(os.path.join('data', '*'))
available_projects = []

for path in project_paths:
    available_projects.append(cleanProjectName(path[5:], is_input = True))

project_name = cleanProjectName(options.cityName, is_input = True)

if not(project_name in available_projects):
    raise ValueError("Projet inconnu. \nLes projets existants sont les suivants :\n{0}\n".format(str(available_projects)))

print("Le projet {} a bien été sélectionné.".format(project_name))
updateProjectName(project_name)


print('__________________________________________________________________________ \n')

