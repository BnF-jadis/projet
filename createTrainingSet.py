import argparse, json, glob, os, cv2, random, time, tqdm

from utils.utils import *
from utils.segment import makeImagePatches

print('\n','Description :')
print('Création d\'un jeu de données d\'entraînement à partir du corpus.', '\n')

print('Options :')
print('--n : nombre de patches d\'entraînement désiré')

parser = argparse.ArgumentParser()
parser.add_argument('--n', action="store", dest="n")
options = parser.parse_args()

if (options.n == None):
    raise ValueError("Vous n'avez pas spécifié le nombre de patches désiré. \n")

project_name = getProjectName()

with open(os.path.join('settings', project_name, 'settings.json')) as settings:
    settings = json.load(settings)

if settings["cnn"]["patch_size"] != 1000:
    print('\n Attention, la tailles des patches est différente de 1000. Par conséquent, le réseau neuronal\
une fois entraîné ne sera plus adapté à la taille standard JADIS. Voulez-vous continuer ? [O(ui)/N(on)]')
    answer = str(input()).lower()[:1]

    if (answer != 'o') and (answer != 'y'):
        raise ValueError("Arrêt du programme. \n")


paths = glob.glob(os.path.join('data', project_name, 'maps', '*.*'))

random_select = []
n_eval = int(np.around(0.1*int(options.n)))

for i in range(int(options.n) + n_eval):
    random_select.append(random.randrange(0, len(paths)))
    
paths = np.asarray(paths)[random_select]

for i, path in tqdm.tqdm(enumerate(paths), 'Découpage des patches d\'entraînement'):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    rdY, rdX = random.randrange(0, h-1000), random.randrange(0, w-1000)
    patch = image[rdY:rdY + 1000, rdY:rdY + 1000]

    name = str(time.time())[4:15].replace('.', '')

    if i < n_eval:
        cv2.imwrite(os.path.join('model', 'train', project_name, 'eval', 'images', name + '.png'), patch)
        cv2.imwrite(os.path.join('model', 'train', project_name, 'eval', 'labels', name + '.png'), patch)
    else:
        cv2.imwrite(os.path.join('model', 'train', project_name, 'images', name + '.png'), patch)
        cv2.imwrite(os.path.join('model', 'train', project_name, 'labels', name + '.png'), patch)
    


print('__________________________________________________________________________ \n')


