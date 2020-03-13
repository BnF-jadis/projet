# 2020, BackThen Maps 
# Coded by Remi Petitpierre https://github.com/RPetitpierre
# For Bibliothèque nationale de France (BnF)

import cv2, thinning, os

import numpy as np
import pandas as pd
import shapefile as shp

from skimage.measure import approximate_polygon
from PIL import Image, ImageDraw

from utils.utils import *
from utils.match import toLatLon

Image.MAX_IMAGE_PIXELS = 500000000


def skeletonize(road_network: np.ndarray, path: str = "workshop/vectorized.png", largest_component: bool = False):
    ''' Thinning/skeletonization of the road network image to a wired model.
    Input(s):
        road_network: black and white image of the road network (streets in white)
        path: path where the skeletonized image should be saved
        largest_component: if True, only the largest road network component will be kept
    Output(s):
        vectorized: skeletonized image
    '''
    
    assert len(road_network.shape) == 2, 'ERROR: road_network must be grayscale image'
    
    img = cv2.resize(road_network, (road_network.shape[1]//2, road_network.shape[0]//2))
    vectorized = thinning.guo_hall_thinning(img)
    vectorized[vectorized > 100] = 255
    vectorized[vectorized <= 100] = 0
    
    if largest_component:
        try:
            _, labels, stats, _ = cv2.connectedComponentsWithStats(vectorized.copy(), connectivity=8, stats=cv2.CC_STAT_AREA)
            stats = stats[1:]
            main_component = (np.argmax(stats[:,4])+1).astype('int32')
            vectorized = (labels == main_component).astype('uint8')*255
        except:
            'Warning: Skeletonization failed to apply largest_component = True param. Skipping.'
    
    cv2.imwrite(path, vectorized)
    
    return vectorized


def findNodes(image: np.ndarray):
    ''' Find the nodes in the road network skeleton image.
    Input(s):
        image: skeletonized image
    Output(s):
        nodes: array of nodes coordinates (x, y)
        degree: degrees of the nodes (2=endpoint, 4=crossroads of 3 streets, 5=crossroads of 4 streets, etc.)
        addresses: directions of the crossing roads, with regard to the node
    '''

    img = image.copy()

    # Find row and column locations that are non-zero
    (rows, cols) = np.nonzero(img)
    nodes, degree, addresses = [], [], []

    for (r,c) in zip(rows, cols):
        if r > 0 and c > 0 and r < image.shape[0]-1 and c < image.shape[1]-1:
            # Extract an 8-connected neighbourhood
            (col_neigh, row_neigh) = np.meshgrid(np.array([c-1, c, c+1]), np.array([r-1, r, r+1]))

            # Cast to int to index into image
            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')

            # Convert into a single 1D array and check for non-zero locations
            pix_neighbourhood = img[row_neigh, col_neigh].ravel() != 0

            # If the number of non-zero locations equals 2, add this to our list of coordinates
            n_neighbours = np.sum(pix_neighbourhood)
            if (n_neighbours == 2) or (n_neighbours >= 4):
                nodes.append((r, c))
                degree.append(n_neighbours)
                direction_set = np.where(pix_neighbourhood == True)[0]
                direction_set = direction_set[direction_set != 4]
                addresses.append(direction_set)

    nodes = np.asarray(nodes)
    
    return nodes, degree, addresses


def cleanNodesEdges(df_nodes: pd.DataFrame):

    df = df_nodes.copy()

    new_addresses, new_degree = [], []

    for ind, address in df['address'].iteritems():        
        new_address = avoidDiagonalEdges(address)
        new_addresses.append(new_address)
        new_degree.append(len(new_address) + 1)

    df['address'] = new_addresses
    df['degree'] = new_degree
    
    return df


def avoidDiagonalEdges(address: list, direction: int = None):
    
    right, diagonal = [1, 3, 5, 7], {0: [1, 3], 2: [1, 5], 6: [3, 7], 8: [5, 7]}
    new_address = []
    
    for r in right:
        if r in address:
            new_address.append(r)
            
    for d in diagonal.keys():   
        if d in address:
            if not(diagonal[d][0] in address) and not(diagonal[d][1] in address):
                if direction != None:
                    if not((8-direction) in diagonal[d]):
                        new_address.append(d)
                else:
                    new_address.append(d)                
    
    return new_address


def explorePath(start_x: int, start_y: int, start_dir: int, image: np.ndarray, nodes_grid: np.ndarray):
    
    ''' Follow the path from one given start node and direction until the next node, and stores the pixels
        on the way.
    Input(s):
        start_x: start node x-coordinate
        start_y: start node y-coordinate
        start_dir: starting direction ({0, 1, 2,
                                        3, -, 5,
                                        6, 7, 8})
        image: skeletonized image of the road network
        nodes_grid: grid of the nodes of the skeletonized image
    Output(s):
        way: list of pixel coordinates on the way
        direction: last direction to reach the 2nd node
        nodes_grid[x, y]: degree of the arrival node
    '''
    
    def absoluteWay(x: int, y: int, way: int):
        
        if way == 0:
            x_, y_ = x-1, y-1
        elif way == 1:
            x_, y_ = x-1, y
        elif way == 2:
            x_, y_ = x-1, y+1
        elif way == 3:
            x_, y_ = x, y-1
        elif way == 5:
            x_, y_ = x, y+1
        elif way == 6:
            x_, y_ = x+1, y-1
        elif way == 7:
            x_, y_ = x+1, y
        elif way == 8:
            x_, y_ = x+1, y+1
        else:
            raise AttributeError('Parameters invalid: (' + str(x) + ',' + str(y) + ',' + str(way) + '), way \
            should be comprised between 0 and 8, and != 4. x, y and way should be of type int.')

        return x_, y_
    
    def noTurnBack(direction: int):
    
        wrong_paths = []
        if direction == 0:
            wrong_paths = [5, 7]
        elif direction == 1:
            wrong_paths = [6, 8]
        elif direction == 2:
            wrong_paths = [3, 7]
        elif direction == 3:
            wrong_paths = [2, 8]
        elif direction == 5:
            wrong_paths = [0, 6]
        elif direction == 6:
            wrong_paths = [1, 5]
        elif direction == 7:
            wrong_paths = [0, 2]
        elif direction == 8:
            wrong_paths = [1, 3]
            
        return wrong_paths
        
    direction = start_dir
    x, y = start_x, start_y
    assert image[x, y] != 0, 'ERROR: start point is not white'
    end = False
    way = [(x, y)]
    
    # First iteration
    new_x, new_y = absoluteWay(x, y, direction)
    assert image[new_x, new_y] != 0, 'ERROR: 2nd point is not white'
    way.append((new_x, new_y))
    x, y = new_x, new_y
    
    wrong_paths = noTurnBack(direction)
    wrong_paths_active = True
    
    if nodes_grid[x, y]:
        end = True
        direction = 8-start_dir

    while not(end):
        if x > 0 and y > 0 and x < image.shape[0]-1 and y < image.shape[1]-1:
            # Extract an 8-connected neighbourhood
            (row_neigh, col_neigh) = np.meshgrid(np.array([x-1, x, x+1]), np.array([y-1, y, y+1]))

            # Cast to int to index into image
            col_neigh, row_neigh = col_neigh.astype('int'), row_neigh.astype('int')

            # Convert into a single 1D array and check for non-zero locations
            try:
                pix_neighbourhood = image[row_neigh, col_neigh].transpose().ravel() != 0
            except:
                print(x, y, image.shape, )
                raise AssertionError()
        
            # If the number of non-zero locations equals 2, add this to our list of coordinates
            n_neighbours = np.sum(pix_neighbourhood)
            direction_set = np.where(pix_neighbourhood == True)[0]
            last_ds = [wrong_paths]
            last_ds.append(direction_set)
            
            direction_set = direction_set[direction_set != 4]
            last_ds.append(direction_set)
            direction_set = direction_set[direction_set != (8-direction)]
            last_ds.append(direction_set)
            direction_set = np.asarray(avoidDiagonalEdges(direction_set, direction))
            last_ds.append(direction_set)
            
            if wrong_paths_active:
                for wrong_path in wrong_paths:
                    direction_set = direction_set[direction_set != wrong_path]
                wrong_paths_active = False                

            if len(direction_set) != 1:
                end = True
                break
            
            direction = direction_set[0]
                
            new_x, new_y = absoluteWay(x, y, direction)
            way.append((new_x, new_y))
            x, y = new_x, new_y

            if nodes_grid[x, y]:
                end = True
        else:
            end = True
             
    return way, direction, nodes_grid[x, y]


def findSegments(df_nodes: pd.DataFrame, image: np.ndarray, min_length: int = 30, return_simple_ways: bool = True):
    ''' Find all the road segments in the network. Keep the ones that are longer than a given length or non-terminal. 
        Optionally, compute the Douglas-Peucker simple itinerary of each segment and return it.
    Input(s):
        df_nodes: list of nodes
        image: skeletonized image of the road network
        min_length: min segment length if the segment is terminal
        return_simple_ways: if True, compute the Douglas-Peucker simple itinerary of each segment and return it
    Output(s):
        (Optional)(simple_ways: the Douglas-Peucker simple itinerary of each segmenty)
        ways: list of segments, containing all the pixels on the way between each couple of nodes
        nodes_grid: image containing all the nodes found in the image and their degree
    '''
    
    img = image.copy()
    done, ways = [], []
    df_nodes = df_nodes.sort_values(by='degree').reset_index(drop=True)
    nodes_grid = np.zeros(image.shape)
    
    for ind, row in df_nodes[['x', 'y', 'degree']].iterrows():
        nodes_grid[row['x'], row['y']] = row['degree']
    nodes_grid = nodes_grid.astype('int')

    for ind, node in df_nodes.iterrows():
        for direct in node['address']:
            code = str(node['x']) + '_' + str(node['y']) + '_' + str(direct)
            if not(code in done):
                way, last_direct, degree = explorePath(start_x=node['x'], start_y=node['y'], 
                                           start_dir=direct, image=img, nodes_grid=nodes_grid)
                if not((len(way) <= min_length) and ((node['degree'] == 2) or (degree == 2))):
                    done.append(str(way[-1][0]) + '_' + str(way[-1][1]) + '_' + str(8-last_direct))
                    ways.append(way)
                    
    if return_simple_ways:
        simple_ways = []
        for way in ways:
            inv_way = np.asarray([np.asarray(way)[:,1], image.shape[0]-np.asarray(way)[:,0]]).transpose()
            simple_ways.append(approximate_polygon(np.asarray(inv_way), tolerance=1.6).tolist())

        return simple_ways, ways, nodes_grid
    
    else:
        return ways, nodes_grid


def thinImage(image: np.ndarray, image_name: str, export_file_path: str, exportPNG: bool = False, 
              exportJSON: bool = False, exportSVG: bool = False, exportSHP: bool = False, geoloc: bool = False):
    
    assert (exportPNG or exportJSON or exportSVG or exportSHP)
    
    # Convert to B&W
    road_network = image.copy()
    road_network[road_network < 254] = 0
    road_network[road_network < 255/2] = 0
    road_network[road_network >= 255/2] = 255

    vectorized = skeletonize(road_network, largest_component = True)
    
    nodes, degree, addresses = findNodes(vectorized)

    if len(degree) < 0:
        return [], [], np.zeros((image.shape[1], image.shape[0]))

    df_nodes = pd.DataFrame({'x': nodes[:,0], 'y': nodes[:,1], 'degree': degree, 'address': addresses })
    df_nodes = df_nodes.sort_values(by='degree').reset_index(drop=True)
    df_nodes = cleanNodesEdges(df_nodes)
    df_nodes = df_nodes[df_nodes['degree'] != 3]

    if (exportJSON or exportSHP):
        simple_segments, full_segments, nodes_grid = findSegments(df_nodes, vectorized, min_length = 15, 
                                                                  return_simple_ways = True)
    else:
        full_segments, nodes_grid = findSegments(df_nodes, vectorized, min_length = 15, 
                                                 return_simple_ways = False)
        simple_segments = []

    if exportPNG:
        toPNG(full_segments, vectorized, export_file_path)
    elif exportSVG:
        toPNG(full_segments, vectorized, os.path.join('workshop', 'thin.png'))

    if geoloc:
        if exportJSON:

            project_name = getProjectName()
            
            try:    
                with open(os.path.join('save', project_name, 'match' , 'primary', image_name + '.json')) as data:
                    data = json.load(data)

                    M = np.asarray(data['M'])

                    simple_segments_JSON = []
                    for segment in simple_segments:
                        s = np.asarray([2*np.asarray(segment)[:,0], image.shape[0]-(2*np.asarray(segment)[:,1])]).T
                        simple_segments_JSON.append(toLatLon((s@M[:, :2]) + M[:, 2:3].transpose()).tolist())

            except:
                print("La géolocalisation de l'image {} n'a pas encore été calculée. Par conséquent, \
il n'est pas possible de calculer la géolocalisation de son réseau filaire".format(image_name))
                simple_segments_JSON = simple_segments

        else:
            print('La géolocalisation du réseau filaire ne fonctionne que pour le format JSON actuellement.')
    else:
        simple_segments_JSON = simple_segments
            
    if exportJSON:
        with open(export_file_path.replace('png', 'json'), 'w') as outfile:
            json.dump(simple_segments_JSON, outfile)
            
    if exportSHP:
        os.makedirs(export_file_path.replace('.png', ''), exist_ok=True)
        toShapefile(simple_segments, os.path.join(export_file_path.replace('.png', ''), image_name))
        
    if exportSVG:
        print("\nAvertissement: Si vous n'avez jamais utilisé cette commande, \
installez d'abord Homebrew, ImageMagick et Potrace via le terminal.\n")
        print('Pour installer Homebrew:\n', 
              '  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"')
        print('Pour installer ImageMagick:\n', '  brew install imagemagick')
        print('Pour installer Potrace: \n', ' brew install potrace\n')
        
        if exportPNG:
            png_path = export_file_path
        else:
            png_path = os.path.join('workshop', 'thin.png')
            
        pnm_path = os.path.join('workshop', 'thin.pnm')
        svg_path = export_file_path.replace('png', 'svg')
        os.system('convert ' + png_path + pnm_path)
        os.system('potrace ' + pnm_path + ' -s -o ' + svg_path)
    
    return simple_segments, full_segments, nodes_grid


def toPNG(segments: list, vectorized: np.ndarray, out_path: str):
    ''' Save a given set of segments as a bitmap image from the road network.
    Input(s):
        segments: list of segments, containing all the pixels on the way between each couple of nodes
        vectorized: skeletonized image of the road network
        out_path: the path, where the output bitmap image should be save
    '''
    
    canvas = (np.ones(vectorized.shape)*255).astype('uint8')
    cv2.imwrite('workshop/canvas.png', canvas);
    bitmap = Image.open('workshop/canvas.png')
    draw = ImageDraw.Draw(bitmap)

    for segment in segments:
        coords = []
        for point in segment:
            coords.append((point[1], point[0]))
            
        draw.line(coords, fill = 'black', width=0)

    bitmap.save(out_path)


def toShapefile(simple_ways, out_path):
    
    w = shp.Writer(out_path)
    w.field('DeletionFlag', 'C', 1, 0)
    w.field('gid', 'N', 11, 0)
    w.field('streetname', 'C', 41, 0)
    w.field('note', 'C', 32, 0)
    
    for i in range(len(simple_ways)):
        w.line([simple_ways[i]])
        w.record('01', i, '', '')
    w.close()

