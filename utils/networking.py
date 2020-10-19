# 2020, BackThen Maps 
# Coded by Remi Petitpierre https://github.com/RPetitpierre
# For Bibliothèque nationale de France (BnF)

import cv2, os, glob
import numpy as np, pandas as pd
from skimage.feature import hog
import scipy.stats as stats
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_ndarray as dl_dist
import networkx as nx
from tqdm import tqdm

from utils.utils import *



def matchPathToMetadata(metadata: pd.core.frame.DataFrame, maps_folder_path: str,
                        filesnames_column: str):
    ''' Match the images found in path with the metadata.
    Input(s):
        metadata: dataframe containing metadata
        maps_folder_path: path where the maps images are found
        filesnames_column: column containing the images names, in the metadata dataframe
    Output(s):
        df_metadata: metadata dataframe containing the images paths, when they were found
    '''

    maps_paths = getImagesPaths(maps_folder_path)
    
    maps_names = []
    for path in maps_paths:
        maps_names.append(getImageName(path))
    maps_names = np.asarray(maps_names)

    _maps_paths = []
    for ind, filename in metadata[filesnames_column].iteritems():
        image_name = getImageName(filename)
        match = (image_name == maps_names).astype('int')

        if np.sum(match) == 1:
            _maps_paths.append(maps_paths[np.argmax(match)])
        else:
            _maps_paths.append(np.nan)
            
    metadata['path'] = _maps_paths
    df_metadata = metadata.dropna(subset = ['path']).reset_index(drop=True)
    
    return df_metadata


def loadMetadata(metadata_path: str):
    ''' Load metadata from path and shape them in a dataframe.
    Input(s):
        metadata_path: path where the data file (json, csv, xls or xlsx) is found
    Output(s):
        df_metadata: metadata dataframe
    '''
    
    dataformat = metadata_path.split('.')[-1]
    if dataformat == 'json':
        df_metadata = pd.read_json(metadata_path)
    elif dataformat == 'csv':
        df_metadata = pd.read_csv(metadata_path)
    elif (dataformat == 'xls') or (dataformat == 'xlsx'):
        df_metadata = pd.read_excel(metadata_path)
    else:
        raise InputError("The format .{} is not supported for metadata. Supported formats are .json, .csv, \
                          .xls and .xlsx".format(dataformat))
    
    # Correction of a bug frequently occuring when exporting with pandas
    if ('Unnamed: 0' in df_metadata.columns) and not ('Unnamed: 1' in df_metadata.columns):
        df_metadata = df_metadata.drop(columns = 'Unnamed: 0')
        
    return df_metadata


def orientedGraphNeighbours(array: np.ndarray, N_neighbours: int):
    ''' Finds the N neighbours for each item in the weighted graph.
    Input(s):
        array: matrix of the edges weights
        N_neighbours: number of neighbours to return for each item
    Output(s):
        neighbourhood: list of the N neighbours, for each item
    '''
    
    neighbourhood = []
    
    for row in array:
        neighbours = np.argpartition(row, np.arange(-N_neighbours, 0))[-N_neighbours:]
        neighbourhood.append(neighbours)
        
    return neighbourhood


def numericalDistance(array: np.ndarray, df: pd.core.frame.DataFrame, col: str, weight: float=1, log: bool=False):
    ''' Computes a distance for numerical elements. The metric is based on the distance normalized with regard
    to the maximal distance found between 2 elements in the matrix.
    Input(s):
        array: input matrix to which the result will be added. Could be a zeros-filled matrix
        df: metadata dataframe
        col: columns containing the numerical variable
        weight: importance given to that variable
        log: wheter the variable should be considered in log scale
    Output(s):
        array: updated matrix, to which the distance matrix for the "col" variable was added
    '''
    
    df_ = df[col].dropna()
    if log:
        df_ = np.log10(df_)
        
    distance = np.max(df_) - np.min(df_)
    
    for ind1, item1 in df_.iteritems():
        for ind2, item2 in df_.iteritems():
            if not(np.isnan(item1)) and not(np.isnan(item2)):
                array[ind1, ind2] += weight*(1 - (np.abs(item1-item2))/distance)
            
    return array


def categoricalDistance(array: np.ndarray, df: pd.core.frame.DataFrame, col: str, weight: float=1, ignore: list=[]):
    ''' Computes a distance for categorical elements. The metric is based on the number of categorical elements in common
    Input(s):
        array: input matrix to which the result will be added. Could be a zeros-filled matrix
        df: metadata dataframe
        col: columns containing the categorical variable
        weight: importance given to that variable
        ignore: list of categories to ignore
    Output(s):
        array: updated matrix, to which the distance matrix for the "col" variable was added
    '''
    
    df_ = df[col].dropna()
    
    for ind1, item1 in df_.iteritems():
        i1 = item1
        if str(type(item1)) == "<class 'str'>":
            i1 = [item1]
            
        for ind2, item2 in df_.iteritems():
            i2 = item2
            if str(type(item2)) == "<class 'str'>":
                i2 = [item2]
            
            for i1_ in i1:
                for i2_ in i2:
                    if str(i1_) == str(i2_):
                        if not i1_ in ignore:
                            array[ind1, ind2] += weight
    return array


def HHOG(df: pd.core.frame.DataFrame, angle_bins: int=180, reduce_size_factor: float=4, filesnames_column: str = 'path'):
    ''' Compute the HHOG (histogram of histogram of oriented gradient) for each image in the dataset
    Input(s):
        df: metadata dataframe
        angle_bins: number of angle bins for the HOG (the greater, the more detailed the HOG features)
        reduce_size_factor: the higher, the smaller the scale of interest of the HOG. Also speeds-up the computation.
        load: in the case where some histograms where already computed for the very same dataset
    Output(s):
        histograms: list HHOG (histogram of HOG), for each image
    '''
    project_name = getProjectName()
    save_HHOG_path = os.path.join('save', project_name, 'HHOG')
    print('PRINT', filesnames_column, '\n')
    paths = df[filesnames_column].values
            
    histograms = []
    
    for path in tqdm(paths, desc='Extraction des caractéristiques visuelles'):

        name = getImageName(path)

        if not(os.path.isfile(os.path.join(save_HHOG_path, name + '.npy'))):

            image = cv2.imread(path, 0)
            image = cv2.resize(image, (int(np.around(image.shape[1])/reduce_size_factor), 
                                       int(np.around(image.shape[0])/reduce_size_factor)))
            fd = hog(image, orientations=angle_bins, pixels_per_cell=(32, 32), cells_per_block=(1, 1), 
                                visualize=False, feature_vector=False)

            histogram = np.sum(np.sum(fd[:,:,0,0], axis=0), axis=0)

            np.save(os.path.join(save_HHOG_path, name), histogram)
            
        else:
            histogram = np.load(os.path.join(save_HHOG_path, name + '.npy'))
        
        histograms.append(histogram)
    
    np.save('workshop/histograms.npy', histograms)
    
    return histograms


def HOGSimilarityDistance(input_array: np.ndarray, df: pd.core.frame.DataFrame, weight: int = 1, 
                          angle_bins: int = 180, reduce_size_factor: int = 4, 
                          out_p_value: bool = False, filesnames_column: str = 'path'):
    ''' Compute the correlation between the HHOG of each image in the dataset and return it as graph normalized weighted edges, giving a measure of the visual similarity between images
    Input(s):
        input_array: input matrix to which the result will be added. Could be a zeros-filled matrix
        df: metadata dataframe
        maps_paths_column: columns containing the images paths
        weight: importance given to the visual features
        angle_bins: number of angle bins for the HOG (the greater, the more detailed the HOG features)
        reduce_size_factor: the higher, the smaller the scale of interest of the HOG. Also speeds-up the computation.
        out_p_value: also return the p_value of the computed correlation
        load: in the case where the histograms where already computed for the very same dataset
    Output(s):
        array: input matrix, to which the visual similarity weights were added
    '''
    
    save_HHOG_path = os.path.join('save', getProjectName(), 'HHOG')

    histograms = HHOG(df, angle_bins=angle_bins, reduce_size_factor=reduce_size_factor, filesnames_column=filesnames_column)
    
    r_corr = np.zeros((len(df), len(df)))
    p_value = np.zeros((len(df), len(df)))

    for ind1 in range(len(histograms)):
        for ind2 in range(len(histograms)):
            if (ind1 != ind2) and (r_corr[ind1, ind2] == 0) and (r_corr[ind2, ind1] == 0):
                r_corr[ind1, ind2], p_value[ind1, ind2] = stats.pearsonr(histograms[ind1], histograms[ind2])

    for ind1 in range(len(histograms)):
        for ind2 in range(len(histograms)):           
            if r_corr[ind1, ind2] == 0:
                r_corr[ind1, ind2] += r_corr[ind2, ind1]
                p_value[ind1, ind2] += p_value[ind2, ind1]
            if ind1 == ind2:
                r_corr[ind1, ind2] = 1
    
    array = normalizeArrayRange(r_corr)*weight + input_array
    
    if out_p_value:
        return array, p_value
    
    else:
        return array
    
    
def computeSimilarityMatrix(settings: dict, df_metadata: pd.core.frame.DataFrame, load: bool = False):
    ''' Compute the similarity matrix
    Input(s):
        settings: to be stored in a json-like dict
        df_metadata: metadata dataframe
        load: in the case where the histograms where already computed for the very same dataset
    Output(s):
        array: matrix containing the edges weights between all items
    '''
    
    array = np.zeros((len(df_metadata), len(df_metadata)))

    if settings['visual_features']['weight'] > 0:
        array = HOGSimilarityDistance(input_array = array, df = df_metadata, 
                                      weight = settings['visual_features']['weight'], 
                                      angle_bins = settings['visual_features']['HOG']['angle_bins'],
                                      reduce_size_factor = settings['visual_features']['HOG']['reduce_size_factor'],
                                      filesnames_column = settings['metadata']['maps']['filesnames_column'])
    
    for i, column in enumerate(settings['metadata']['columns']):
        if column['name'] in df_metadata.columns:
            name = column['name']
        else:
            alternative_names = dl_dist(column['name'], np.asarray(df_metadata.columns))
            name = df_metadata.columns[np.argmin(alternative_names)]
            raise NameError('No column named {0} found in data/{1}/data.xlsx. Do you mean {2} ? Please correct the name of column {3} in settings/{1}/networking.json'.format(
                             column['name'], getProjectName(), name, i+1))
            
        if column['distance'] in ['categorical', 'numerical']:
            distance = column['distance']
        else:
            column_type = df_metadata[name].dtype.name
            if ('float' in column_type) or ('int' in column_type) or ('bool' in column_type):
                distance = 'numerical'
            else:
                distance = 'categorical'
                
        weight = column['weight']
        assert isinstance(weight, int) or isinstance(weight, float)
                
        if distance == 'numerical':
            try:
                log = column['log']
                if not(log in [0, 1]):
                    log = 0
            except:
                log = 0
            array = numericalDistance(array, df_metadata, name, weight=weight, log=log)
        elif distance == 'categorical':
            array = categoricalDistance(array, df_metadata, name, weight=weight, ignore = ['', ' '])
            
    return array


def createGraph(array: np.ndarray, N_neighbours:int = 10, name:str = 'Similarity matrix'):
    ''' Compute a networkx graph from the input array of edges weights
    Input(s):
        array: input matrix containing the edges weights between items
        name: name of the graph
    Output(s):
        G: networkx graph
    '''

    G = nx.Graph()
    G.add_nodes_from(np.arange(0, len(array)))
    neighbourhood = orientedGraphNeighbours(array, N_neighbours+1)

    list_edges = []

    for ind, item in enumerate(neighbourhood):
        for neighbour in item:
            if neighbour != ind:
                list_edges.append((ind, neighbour, array[ind, neighbour]))
                
    G.add_weighted_edges_from(list_edges)
    G.name = name
                
    return G
