# 2020, BackThen Maps 
# Coded by Remi Petitpierre https://github.com/RPetitpierre
# For Bibliothèque nationale de France (BnF)

import cv2, os, imutils

import numpy as np
import pandas as pd
import tensorflow as tf;

from skimage.morphology import remove_small_objects, area_closing, dilation

from utils.utils import *
from utils.external.dhsegment.inference import LoadedModel


def dhSegmentPrediction(model_path: str, images_path: str, prediction_path: str, classes: list):
    ''' dhSegment-based prediction function.
    Input(s):
        model_path: path to the tensorflow model
        images_path: path to the folder where the images patches used for prediction are found
        prediction_path: path to the folder where the predicted patches will be saved
        classes: color classes used to train the model
    '''

    input_files = glob.glob(os.path.join(images_path, '*'))

    with tf.Session():
        m = LoadedModel(model_path, predict_mode='filename')

        for filename in input_files:

            prediction_outputs = m.predict(filename)
            probs = prediction_outputs['probs'][0]

            prediction_map = np.zeros((probs.shape[0], probs.shape[1], 3))
            for i in range(probs.shape[0]):
                for j in range(probs.shape[1]):
                    prediction_map[i,j] = classes[np.argmax(probs[i,j])]

            prediction_map = prediction_map.astype(np.uint8, copy=False)

            basename = os.path.basename(filename).split('.')[0]
            np.save(os.path.join(prediction_path, '{}.npy'.format(basename)), probs)

            
def reconstituteMap(image: np.ndarray, prediction_path: str, classes: list, rows: int, cols: int):
    ''' Reconstitution of the map based on the prediction on image patches.
    Input(s):
        image: original full-size map
        prediction_path: path to the folder where the predicted patches are found
        classes: color classes used to train the model
        rows: number of rows of patches
        cols: number of columns of patches
    Output(s):
        prediction_map: image of the full_sized predicted segmentation of the map
    '''
    
    reconstitution = np.zeros((1000+(rows*800), 1000+(cols*800), 3))
    
    for row in range(rows):
        for col in range(cols):
            
            probs = np.load(os.path.join(prediction_path, str(row) + '_' + str(col) + '.npy'))
            probs = probs

            pre_row, pre_col = 50, 50
            post_row, post_col = 950, 950

            if row == 0:
                pre_row = 0
            elif row == rows-1:
                post_row = 1000
            if col == 0:
                pre_col = 0
            elif col == cols-1:
                post_col = 1000
            
            insert = cv2.resize(probs,(1000, 1000))
            reconstitution[pre_row+(row*800):(row*800)+post_row, 
                           pre_col+(col*800):(col*800)+post_col] = insert[
                pre_row:post_row, pre_col:post_col]
            
    prediction_map = np.zeros((reconstitution.shape[0], reconstitution.shape[1], 3))
    
    labels = np.argmax(reconstitution, axis = 2)
    for i in range(len(classes)):
        prediction_map[labels == i] = classes[i]
            
    prediction_map = prediction_map.astype(np.uint8, copy=False)
    prediction_map = cv2.cvtColor(prediction_map, cv2.COLOR_BGR2RGB)
    prediction_map = prediction_map[:image.shape[0], :image.shape[1]]
    
    prediction_map = cv2.resize(prediction_map, (int(prediction_map.shape[1]//(1000/848)), 
                                                 int(prediction_map.shape[0]//(1000/848))))
    
    return prediction_map.astype('uint8')


def makeImagePatches(image: np.ndarray, patches_path: str = '', export: bool = True):
    ''' Reconstitution of the map based on the prediction on image patches.
    Input(s):
        image: original full-size map
        patches_path: path to the folder where the image patches will be saved
        export: if True, the patches are saved to patches_path
    Output(s):
        rows: number of rows of patches
        cols: number of columns of patches
        patches: if they are not exported, the image patches are returned
    '''
    
    rows = 1 + ((image.shape[0]-201)//800)
    cols = 1 + ((image.shape[1]-201)//800)
    
    patches = []
    for row in range(rows):
        for col in range(cols):
            patch = image[0+row*800:1000+row*800, 0+col*800:1000+col*800]
            if patch.shape[:2] != (1000, 1000):
                background = np.zeros((1000, 1000, 3))
                background[0:patch.shape[0], 0:patch.shape[1]] = patch
                patch = background
            
            if export:
                cv2.imwrite(os.path.join(patches_path, str(row) + '_' + str(col) + '.png'), patch)
            else:
                patches.append(patch.astype('uint8'))
    if export:        
        return rows, cols
    else:
        return rows, cols, patches


def removeSmallComponents(image: np.ndarray, component_min_area: int = 100):
    ''' Remove the components smaller than a defined area.
    Input(s):
        image: segmented map image
        component_min_area: threshold below which the components should be removed
    Output(s):
        image: grayscale cleaned segmented map image
    '''
    
    def labelsToGray(orig_gray: np.ndarray, orig_labels: np.ndarray, labeled: np.ndarray):
    
        new_gray = np.zeros(orig_gray.shape)

        for shade in np.unique(orig_gray):
            shade_labels = np.unique(orig_labels[orig_gray == shade])
            shade_labels = shade_labels[shade_labels != 0]

            for shade_label in shade_labels:
                new_gray[labeled == shade_label] = shade

        return new_gray
    
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    labels = findAllComponents(gray)
    cleared = remove_small_objects(labels, 100)
    closed = area_closing(dilation(dilation(cleared)))
    labels[cleared == 0] = closed[cleared == 0]
    
    new_gray = labelsToGray(gray, cleared, labels)
    
    return new_gray


"""
def cleanAnnotations(images_folder_path: str = 'patches/input/BnF/', label_folder_path: str = 'patches/output/BnF/',
                    new_images_folder_path: str = 'patches/images/', new_label_folder_path: str = 'patches/labels/'):
    ''' Clean the photoshop-annotated label images, in order to make them ready for CNN training
    Input(s):
        images_folder_path: folder containing the original images
        label_folder_path: folder containing the dirty label images
        new_images_folder_path: folder containing the copy of original images
        new_label_folder_path: folder containing the cleaned label images
    '''
    
    label_paths = glob.glob(os.path.join(label_folder_path, '*.png'))
    
    os.makedirs(new_images_folder_path, exist_ok=True)
    os.makedirs(new_label_folder_path, exist_ok=True)

    for label_path in label_paths:
        label = cv2.imread(label_path)
        label = cleanColors(label)
        label = removeSmallComponents(label, threshold=18)

        image_path = images_folder_path + label_path[len(label_folder_path):]
        image = cv2.imread(image_path)

        counter = 0
        
        for angle in [0, 90, 180, 270]:
            rotated_label = imutils.rotate_bound(label, angle)
            rotated_image = imutils.rotate_bound(image, angle)
            new_label_path = new_label_folder_path + label_path[len(label_folder_path):-4]
            new_image_path = new_images_folder_path + image_path[len(images_folder_path):-4]
            
            cv2.imwrite(new_label_path + str(counter) + '.png', rotated_label)
            cv2.imwrite(new_image_path + str(counter) + '.png', rotated_image)
            counter += 1
    
    # Remove the s tail of sRGB images, this removes a warning in TensorFlow
    command = 'mogrify ' + os.path.join(new_images_folder_path, '*.png')
    os.system(command)

    command = 'mogrify ' + os.path.join(new_label_folder_path, '*.png')
    os.system(command)
"""


def massTrinarize(input_folder: str, output_folder: str):
    ''' Allows to convert a whole folder of label or prediction images in a trinary color format
    Input(s):
        input_folder: multi-color image folder
        output_folder: trinary color image folder
    '''

    paths = glob.glob(os.path.join(input_folder, '*.png'))
    for path in paths:
        img = cv2.imread(path, 1)
        if label:
            img = cleanColors(img)
            img = removeSmallComponents(img, threshold = 18)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = trinarize(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        cv2.imwrite(path.replace(input_folder, output_folder), img)
        
    # Remove the s tail of sRGB images, this removes a warning in TensorFlow    
    command = 'mogrify ' + os.path.join(output_folder, '*.png')


def findAllComponents(image: np.ndarray):
    ''' Allows to find all connected components in an image, resolving the defects from the standard opencv connectedComponents function
    Input(s):
        image: original label or prediction image containing clear connected components
    Output(s):
        labels: image labelled with connected components unique ids
    '''
    
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, labels = cv2.connectedComponents(gray, connectivity=4)
    _, labels_bg = cv2.connectedComponents((labels == 0).astype('uint8'), connectivity=4)
    _, labels_inv = cv2.connectedComponents(255-gray, connectivity=4)
    _, labels_inv_bg = cv2.connectedComponents((labels_inv == 0).astype('uint8'), connectivity=4)
    
    max_labels, max_labels_bg = int(np.max(labels)), int(np.max(labels_bg))
    max_labels_inv = int(np.max(labels_inv))
        
    labels_bg = max_labels + labels_bg
    labels_bg[labels_bg == max_labels] = 0

    labels_inv_bg = max_labels_inv + labels_inv_bg
    labels_inv_bg[labels_inv_bg == max_labels_inv] = 0

    labels = labels + labels_bg
    labels_inv = labels_inv + labels_inv_bg
    labels = labels + (max_labels + max_labels_bg)*labels_inv
    
    return labels


def removeBlackFlakes(image: np.ndarray):
    ''' Clean the map by removing frame/background flakes from the image.
    Input(s):
        image: segmented map image
    Output(s):
        image: cleaned segmented map image (grayscale)
    '''

    if len(image.shape) > 2:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img = image.copy()
        
    labels = findAllComponents(img)
    components = np.unique(labels[img == 0])
    background = [labels[0, 0], labels[-1, 0], labels[0, -1], labels[-1, -1]]
    
    for corner in background:
        components = components[components != corner]

    for ind, component in enumerate(components):
        kernel = np.ones((3, 3), np.uint8) 
        dilated = cv2.dilate((labels == component).astype('uint8'), kernel, iterations=1).astype('bool')
        replace_color = img[dilated & (labels != component)]
        unique, count = np.unique(replace_color, return_counts = True)
        img[labels == component] = unique[np.argmax(count)]
            
    return img

def reconstituteImage(orig: np.ndarray, rows: int, cols: int, patches: list):
    ''' Reconstitution of the map based on the prediction on image patches.
    Input(s):
        orig: original full-size map
        rows: number of rows of patches
        cols: number of columns of patches
        patches: image patches
    Output(s):
        image: full-sized reconstituted image
    '''
    
    reconstitution = np.zeros((1000+(800*rows),1000+(800*cols)))
    
    iterator = 0
    for row in range(rows):
        for col in range(cols):
            
            pre_row, pre_col = 100, 100
            post_row, post_col = 900, 900

            if row == 0:
                pre_row = 0
            elif row == rows-1:
                post_row = 1000
            if col == 0:
                pre_col = 0
            elif col == cols-1:
                post_col = 1000

            reconstitution[(row*800):(row*800)+1000, (col*800):(col*800)+1000] = patches[iterator]
            iterator += 1
            
    reconstitution = reconstitution[:orig.shape[0], :orig.shape[1]]
    
    return reconstitution.astype('uint8')


def postProcessing(prediction: np.ndarray, component_min_area: int = 100):
    ''' Global function of segmented images post-processing.
    Input(s):
        prediction: segmented map image
        component_min_area: threshold below which the components should be removed
    Output(s):
        colored: cleaned segmented map image
    '''

    assert component_min_area >= 0, ('component_min_area must be a positive integer')
    
    processed = prediction.copy()
    
    for i in range(10):
        processed = cv2.blur(processed, (3, 3))
        processed = cleanColors(processed)
    
    rows, cols, patches = makeImagePatches(processed, export = False)
    cleared_patches = []
    for patch in patches:
        cleared_patches.append(removeSmallComponents(patch, component_min_area = 100))
    processed = reconstituteImage(processed, rows, cols, patches = cleared_patches)
    
    processed = cleanBackground(processed)
    processed = removeBlackFlakes(processed)
    
    colored = grayToColor(prediction, processed)
    
    return colored


def cleanBackground(image: np.ndarray):
    ''' Clean the frame by removing color flakes and smoothing the border.
    Input(s):
        image: segmented map image
    Output(s):
        image: cleaned segmented map image (grayscale)
    '''

    if len(image.shape) > 2:
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    mask = (gray != 0).astype('uint8')*255

    for i in range(5):
        mask = cv2.blur(mask, (15, 15))
        mask[mask < 127.5] = 0
        mask[mask >= 127.5] = 255
        
    labels = findAllComponents(mask)
    unique, counts = np.unique(labels, return_counts = True)
    if len(unique) > 1:
        main = unique[1:][np.argmax(counts[1:])]
        mask[(mask != 0) & (labels != main)] = 0
        image[mask == 0] = 0

    return image

