# 2020, JADIS 
# Coded by Remi Petitpierre (https://github.com/RPetitpierre)
# From Ecole Polytechnique Fédérale de Lausanne (EPFL)
# For Bibliothèque nationale de France (BnF)


import cv2, glob, os, shutil, string, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getImageName(path_name: str):
    ''' Get image name from path.
    Input(s):
        path_name: path name
    Output(s):
        image_name: image name, without folder and without extension
    '''

    image_name = path_name.split('.')[0].split('\\')[-1].split('/')[-1]

    return image_name


def normalizeArray(array: np.ndarray):
    ''' Central limit theorem normalization.
    Input(s):
        input_array: input array to normalize
    Output(s):
        array: normalized array
    '''
    
    raise Warning('deprecated')
    
    normalized = array.copy()
    normalized = (array - np.mean(array)) / np.std(array)
                  
    return normalized


def normalizeArrayRange(input_array: np.ndarray):
    ''' Normalize an array, so that its values take place between 0 and 1.
    Input(s):
        input_array: input array to normalize
    Output(s):
        array: normalized array
    '''
    
    array = input_array.copy()
    array = (array - np.min(array))/(np.max(array) - np.min(array))
    
    return array

def getImagesPaths(folder_path: str):
    ''' Given the path of the folder, find all the image files contained and that are supported by Opencv.
    Input(s):
        folder_path: path of the folder where the images are contained
    Output(s):
        path: list of the image paths
    '''
    
    opencv_formats = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg', 'jpg', 'jpe', 'jp2', 'tiff', 
                      'tif', 'png']
    
    paths = []
    for img_format in opencv_formats:
        paths += glob.glob(os.path.join(folder_path, '*.' + img_format))

    try:
        assert len(paths) > 0
    except:
        raise AssertionError('No image found in folder {}. Please check the corresponding folder path in settings. Consider that the supported image extensions are bmp, pbm, pgm, ppm, sr, ras, jpeg, jpg, jpe, jp2, tif, tiff, and png.'.format(folder_path))
    
    return paths


def cleanColors(image: np.ndarray, threshold: float=127.5):
    ''' Thresholds colors to pure 0 or 255 RGB colors.
    Input(s):
        image: generally a label or prediction image, eventually a binary image
        threshold: generally 127.5 (= 255/2)
    Output(s):
        img: the same image, thresholded
    '''

    img = image.copy()
    img[img < threshold] = 0
    img[img >= threshold] = 255
                        
    return img


def grayToColor(orig: np.ndarray, gray: np.ndarray):
    ''' Convert a grayscale label image to color, given a colored original model.
    Input(s):
        orig: original model image (doesn't need to be the same than grayscale !). Only the color pannel should be similar.
        gray: grayscale image
    Output(s):
        colored: the colored image
    '''
    
    g = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    shades = sorted(np.unique(g))
    colored = np.zeros(orig.shape)

    for shade in shades:
        color = (orig[g == shade][0]).tolist()
        colored[gray == shade] = color
        
    return colored
    
    
def imageShow(image: np.ndarray):
    ''' Simple plotting function.
    Input(s):
        image: the image to plot
    Output(s):
        fig, ax: the matplotlib plot
    '''
    
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    return fig, ax


def trinarize(image: np.ndarray, to_color: list = [0, 255, 255]):
    ''' Trinarize the image, keeping only white, black, and a customized color replacing all other.
    Input(s):
        image: the image to trinarize
        to_color: the color used to replace all other

    Output(s):
        img: the trinarized image
    '''
    
    try:
        assert len(image.shape) == 2
    except AssertionError:
        print('The input image should not be grayscale')

    try:
        assert len(to_color)
        for channel in to_color:
            assert isinstance(channel, int) 
            assert ((channel >= 0) & (channel <= 255))
    except:
        print('The destination color should be a length 3 list of integers, comprised between 0 and 255')
    
    img = np.zeros((image.shape[0], image.shape[1], 3))
    img[image <= 0] = [0, 0, 0]
    img[image >= 255] = [255, 255, 255]
    img[(image > 0) & (image < 255)] = to_color
                        
    return img


def normalizeColor(image, means):
    
    raise Warning('Deprecated')
    img = image.copy().astype('int64')
    
    for i in range(3):
        img[:,:,i] = img[:,:,i] + (means[i]-np.median(img[:,:,i]))
        img[:,:,i][img[:,:,i] < 0] = 0
        img[:,:,i][img[:,:,i] > 255] = 255
        
    return img.astype('uint8')


def resetFolders(folders: list):
    
    for folder in folders:
        os.system("rm -r {}".format(folder))
        os.makedirs(folder, exist_ok=True)

        
def save_df(df: pd.core.frame.DataFrame, path: str, name: str):
    
    df.to_json(os.path.join(path, name + '.json'))
    df.to_csv(os.path.join(path, name + '.csv'))
    df.to_excel(os.path.join(path, name + '.xls'))
    df.to_excel(os.path.join(path, name + '.xlsx'))


def cleanProjectName(project_name: str, is_input: bool = False):
    
    name = project_name.lower().capitalize()
    name = name.replace(' ', '_').replace('\n', '_').replace('\s', '_').replace('\t', '_')
    
    for p in string.punctuation:
        name = name.replace(p, '')
    
    if is_input and (project_name != name):
        print('Le nom a légèrement été modifié pour éviter des erreurs de chemin.')
    
    return name


def updateProjectName(project_name: str):

    with open(os.path.join('utils', 'project.json'), 'w') as outfile:
        json.dump({"project": project_name}, outfile)



def getProjectName():
    
    with open(os.path.join('utils', 'project.json')) as project_name:
        project_name = json.load(project_name)['project']
        
    return project_name


def getPathsToProcess(input_path, output_path, input_format = '*', output_format = '*'):
    
    input_paths = glob.glob(os.path.join(input_path, '*.' + input_format))
    output_paths = glob.glob(os.path.join(output_path, '*.' + output_format))
    
    output_names = []
    for path in output_paths:
        output_names.append(getImageName(path))
    
    paths_to_process = []
    for path in input_paths:
        if not(getImageName(path) in output_names):
            paths_to_process.append(path)

    if len(paths_to_process) < len(input_paths):
        print('Certains calculs ont déjà été effectués. Reprise.')
            
    return paths_to_process
    
