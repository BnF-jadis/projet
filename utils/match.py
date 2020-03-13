# 2020, BackThen Maps 
# Coded by Remi Petitpierre https://github.com/RPetitpierre
# For Bibliothèque nationale de France (BnF)

import numpy as np
import pandas as pd
import cv2, os, tqdm, glob
import overpass, json
from PIL import Image, ImageDraw

from skimage.feature import hog
from sklearn.neighbors import KDTree
from scipy.spatial.distance import euclidean
from matplotlib.tri import Triangulation
from scipy.ndimage.interpolation import map_coordinates

from utils.utils import *
from utils.segment import removeSmallComponents, reconstituteImage, makeImagePatches

Image.MAX_IMAGE_PIXELS = 500000000
    

def createAnchorMap(settings: dict):
    
    def requestOSMAPI(city_name: str, osm_geoloc_maxsize: int, admin_level: int):
    
        api = overpass.API(timeout=3600)
        data = api.get('area[name="{0}"]["admin_level"="{1}"];way(area)[highway];out geom;'.format(city_name, str(admin_level)), 
            responseformat = "json")

        bounds, highway_type, ids, geometry = [], [], [], []
        for element in data['elements']:
            try:
                bounds.append(element['bounds'])
                highway_type.append(element['tags']['highway'])
                ids.append(element['id'])
                geometry.append(element['geometry'])
            except:
                pass

        df = pd.DataFrame(bounds)
        df['id'] = ids
        df['highway_type'] = highway_type
        df['geometry'] = geometry

        # Eliminate homonym cities
        for i in range(2):
            df = df[(df['minlat'] > df['minlat'].mean()-(1.5*df['minlat'].std())) &
                    (df['minlon'] > df['minlon'].mean()-(1.5*df['minlon'].std())) &
                    (df['maxlat'] < df['maxlat'].mean()+(1.5*df['maxlat'].std())) &
                    (df['maxlon'] < df['maxlon'].mean()+(1.5*df['maxlon'].std()))]


        highway_types = {'footway': 1, 'residential': 2, 'service': 2, 'steps': 1, 'pedestrian': 2, 'primary': 4,
             'secondary': 3, 'tertiary': 2, 'cycleway': 1, 'path': 1, 'trunk_link': 2, 'living_street': 2,
             'trunk': 3, 'track': 2, 'primary_link': 2, 'corridor': 1, 'construction': 1,
             'motorway_link': 2, 'secondary_link': 2, 'tertiary_link': 1, 'motorway': 4, 'road': 2}

        df['width'] = 1
        for type_ in highway_types.keys():
            df['width'].loc[df['highway_type'] == type_] = 1 + highway_types[type_]

        return df
    
    
    def drawAnchorMap(anchor: np.ndarray, df: pd.DataFrame, coef: float, streetwidth_coef: float):
    
        def drawStreet(draw: ImageDraw.ImageDraw, geometry: list, coef: float, width: int):

            street = pd.DataFrame(geometry)
            street['lon'] = coef*(street['lon']-minlon)
            street['lat'] = coef*(street['lat']-minlat)

            points = np.ravel(np.asarray((street['lon'].tolist(), street['lat'].tolist())).T).tolist()
            draw.line(points, fill = 255, width = width)

        minlat, maxlat = df['minlat'].min(), df['maxlat'].max()
        minlon, maxlon = df['minlon'].min(), df['maxlon'].max()

        anchor = Image.fromarray(anchor)

        draw = ImageDraw.Draw(anchor)
        fdrawStreet = np.vectorize(drawStreet)
        fdrawStreet(draw, df['geometry'].values, coef, df['width'].values*streetwidth_coef)
        del draw

        anchor = anchor.transpose(Image.FLIP_TOP_BOTTOM)
        anchor.save('workshop/PIL.png')
        anchor = cv2.imread('workshop/PIL.png')

        rows, cols, patches = makeImagePatches(anchor, export = False)
        cleared_patches = []
        for patch in patches:
            cleared_patch = patch
            for i in range(10):
                cleared_patch = cv2.blur(cleared_patch, (3, 3))
                cleared_patch[cleared_patch < 255/2] = 0
                cleared_patch[cleared_patch >= 255/2] = 255
            cleared_patches.append(removeSmallComponents(cleared_patch, component_min_area = 200))
        anchor = reconstituteImage(anchor, rows, cols, patches = cleared_patches)

        return anchor

    project_name = getProjectName()

    save_path = os.path.join('save', project_name, 'projection', 'anchor.json')

    if len(glob.glob(save_path)) > 0:
        print("\nL'ancre a déjà été calculée. Vous la trouverez à l'emplacement suivant: {0}\n".format(save_path))

    else:    
        df = requestOSMAPI(settings['corpus']['city_name'], settings['anchor']['image_maxsize'], settings['anchor']['admin_level'])

        minlat, maxlat = df['minlat'].min(), df['maxlat'].max()
        minlon, maxlon = df['minlon'].min(), df['maxlon'].max()

        h, w = maxlat-minlat, maxlon-minlon
        coef = settings['anchor']['image_maxsize']/np.max([h, w])

        anchor = np.zeros((int(np.around(h*coef)), int(np.around(w*coef)))).astype('uint8')

        anchor = drawAnchorMap(anchor, df, coef, settings['anchor']['streetwidth_coef'])

        citylat = np.mean([minlat, maxlat])
        if maxlat > 0:
            top_lon_deformation = np.cos(np.pi*maxlat/180)
            bot_lon_deformation = np.cos(np.pi*minlat/180)
        else:
            top_lon_deformation = np.cos(np.pi*minlat/180)
            bot_lon_deformation = np.cos(np.pi*maxlat/180)

        lon_deformation = np.max([top_lon_deformation, bot_lon_deformation])

        anchor = cv2.resize(anchor, (int(np.around(anchor.shape[1]*lon_deformation)), anchor.shape[0])).astype('uint8')   
        pixel_offset = (lon_deformation-top_lon_deformation)*anchor.shape[1]/2
        
        (h, w) = anchor.shape

        X = np.asarray([0, 0, h, h])
        Y = np.asarray([0, w, 0, w])
        Zx = np.asarray([0, 0, 0, 0])
        if top_lon_deformation <  bot_lon_deformation:
            Zy = np.asarray([-pixel_offset, pixel_offset, 0, 0])
        else:
            Zy = np.asarray([0, 0, -pixel_offset, pixel_offset])

        dx, dy = computeDeformation(X, Y, Zx, Zy, (h, w))
        transform = elasticTransform(anchor, dx, dy)

        transform[transform < 255/2] = 0
        transform[transform > 0] = 255 

        cv2.imwrite(os.path.join('export', project_name, 'anchor', 'anchor.png'), transform)
        
        with open(os.path.join('save', project_name, 'projection', 'anchor.json'), 'w') as outfile:
            json.dump({'bot_lon_deformation': bot_lon_deformation, 
                       'top_lon_deformation': top_lon_deformation, 
                       'minlon': minlon, 'maxlon': maxlon, 
                       'minlat': minlat, 'maxlat': maxlat, 
                       'coef': coef, 'shape': [h, w]}, outfile)
        

def loadProjectionParams():
    ''' Load anchor reprojection parameters '''

    project_name = getProjectName()

    with open(os.path.join('save', project_name, 'projection' ,'anchor.json')) as data:

        data = json.load(data)
        bot_lon_deformation = data['bot_lon_deformation']
        top_lon_deformation = data['top_lon_deformation']
        minlon = data['minlon']
        maxlon = data['maxlon']
        minlat = data['minlat']
        maxlat = data['maxlat']
        coef = data['coef']
        shape = data['shape']
        
    return bot_lon_deformation, top_lon_deformation, minlon, maxlon, minlat, maxlat, coef, shape


def toLatLon(coords):
    
    bot_lon_deformation, top_lon_deformation, minlon, maxlon, minlat, maxlat, coef, shape = loadProjectionParams()
    
    latitude, longitude = [], []
    lon_def_coef = bot_lon_deformation-top_lon_deformation
    mid_w = shape[1]/2
    mid_lon = np.mean([minlon, maxlon])

    for i in range(len(coords)):
        lat = coords[i, 1]/coef
        lon = coords[i, 0]
                
        if maxlat > 0:
            latitude.append(maxlat-lat)
        else:
            latitude.append(minlat-lat)

        lon_def = top_lon_deformation + lon_def_coef*lat/shape[0]
        longitude.append(mid_lon+((lon - mid_w)/lon_def)/coef)

    geolocalisation = np.asarray([latitude, longitude]).T
    
    return geolocalisation


def addFringePoints(src_pts_matched: np.ndarray, errors: np.ndarray, shape: tuple):
    ''' Adds static points along the fringe of the image to avoid deformation of the frame. Then, 
        computes and returns the X and Y deformation for all the points, including fringe points.
    Input(s):
        src_pts_matched: keypoints matched in the source image
        errors: error of the keypoints with regard to the reference map
        shape: shape of the image to deform
    Output(s):
        X: vertical coordinate of the keypoints, including fringe keypoints
        Y: horizontal coordinate of the keypoints, including fringe keypoints
        Zx: vertical deformation of the keypoints
        Zy: horizontal deformation of the keypoints
    '''

    fringe = (int(np.around(shape[0]/100)), int(np.around(shape[1]/100)))
    x = np.linspace(0, shape[0], fringe[0]).tolist()
    x += np.linspace(0, shape[0], fringe[0]).tolist()
    x += np.zeros(fringe[1]).tolist()
    x += (shape[0]*np.ones(fringe[1])).tolist()

    y = np.zeros(fringe[0]).tolist()
    y += (shape[1]*np.ones(fringe[0])).tolist()
    y += np.linspace(0, shape[1], fringe[1]).tolist()
    y += np.linspace(0, shape[1], fringe[1]).tolist()

    x, y = np.asarray(x).astype('int'), np.asarray(y).astype('int')
    z = np.zeros(2*(fringe[0]+fringe[1])).astype('int')
    
    X = np.concatenate((src_pts_matched[:, 1], x))
    Y = np.concatenate((src_pts_matched[:, 0], y))
    Zx = np.concatenate((errors[:, 0], z))
    Zy = np.concatenate((errors[:, 1], z))

    return X, Y, Zx, Zy


def computeErrors(src_pts_matched: np.ndarray, dst_pts_matched: np.ndarray, M: np.ndarray):
    ''' Computes the error, or the deformation between the matched keypoints of both images, 
    with regard to the homography.
    Input(s):
        src_pts_matched: keypoints matched in the source image
        dst_pts_matched: keypoints matched in the destination image
        M: matrix of transformation src->dst
    Output(s):
        errors: error of the keypoints in the second image with regard to the reference '''
    
    errors = dst_pts_matched - ((src_pts_matched@M[:,:2]) + M[:, 2:3].transpose())
    
    errors = np.asarray(errors)
    
    return errors


def computeDeformation(X, Y, Zx, Zy, shape):
    ''' Computes the error, or the deformation between the matched keypoints of both images, 
        at each pixel position in the second image.
    Input(s):
        X: vertical coordinate of the keypoints, including fringe keypoints
        Y: horizontal coordinate of the keypoints, including fringe keypoints
        Zx: vertical deformation of the keypoints
        Zy: horizontal deformation of the keypoints
        shape: shape of the image to deform
    Output(s):
        dx: map of the vertical deformation, at each pixel position in the 2nd image
        dy: map of the horizontal deformation, at each pixel position in the 2nd image
    '''

    triangulation = Triangulation(X, Y)
    finder = triangulation.get_trifinder()

    triangle = np.zeros(shape)
    j_coords = np.arange(shape[1])

    for i in range(shape[0]):
        triangle[i] = finder(i*np.ones(shape[1]).astype('int64'), j_coords) 

    array_x = triangulation.calculate_plane_coefficients(Zx)
    array_y = triangulation.calculate_plane_coefficients(Zy)

    n_triangle = array_x.shape[0]
    dx, dy = np.zeros(shape), np.zeros(shape)
    indices = np.indices(shape)

    dx = indices[0]*array_x[:,0][triangle.astype('int16')] + indices[1]*array_x[:, 1][triangle.astype('int16')] + \
                    array_x[:,2][triangle.astype('int16')]
    dy = indices[0]*array_y[:,0][triangle.astype('int16')] + indices[1]*array_y[:, 1][triangle.astype('int16')] + \
                        array_y[:,2][triangle.astype('int16')]
            
    return dx, dy


def elasticTransform(image, dx, dy):
    '''Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    '''
    def transformChannel(channel, dx, dy):
        
        x, y = np.meshgrid(np.arange(channel.shape[0]), np.arange(channel.shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        
        return map_coordinates(channel, indices, order=1).reshape(channel.shape)
        
    if len(image.shape)==2:
        
        return transformChannel(image, dx, dy)
    
    else:
        r, g, b = cv2.split(image)
        r_ = transformChannel(r, dx, dy)
        g_ = transformChannel(g, dx, dy)
        b_ = transformChannel(b, dx, dy)
        
        transformed_image = cv2.merge([r_, g_, b_])
        
        return transformed_image


def mapInvariantFeatures(image, n_orientations = 18):
        
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    blur = cv2.blur(cv2.blur(gray, (3, 3)), (5, 5))
    
    gray[gray == 0] = 255

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats((255-gray).astype('uint8'), connectivity=4)
    
    features, directions = [], []
    radius_list = [2, 2, 4, 8, 16]
    scale_list = [4, 8, 16, 32]
    unique_labels = np.unique(labels)

    kernel_list = []
    for radius in radius_list:
        kernel_list.append(np.ones((radius, radius), np.uint8))
    
    if len(unique_labels) > 2**16:
        labels_format = 'uint64'
    else:
        labels_format = 'uint16'
        
    direction = 0
    empty_vector = np.zeros(n_orientations)

    for scale in scale_list:

        scale_vectors, scale_directions = [], []
        fd = hog(blur.astype('uint8'), feature_vector=False, orientations = n_orientations, 
                 cells_per_block=(1, 1), pixels_per_cell=(scale, scale))
        
        small_labels = cv2.resize(labels.astype(labels_format), (fd.shape[1], fd.shape[0]),
                                 interpolation = cv2.INTER_NEAREST)
        
        for label in unique_labels:
            vector = empty_vector.copy()
            shape_mask = (small_labels == label).astype('uint8')

            x, y, w, h = cv2.boundingRect(shape_mask)
            y_inf, x_inf = y-32, x-32
            y_sup, x_sup = y+h+32, x+w+32
            shape_mask = shape_mask[y_inf:y_sup, x_inf:x_sup]

            shape_mask_shape = shape_mask.shape
            if (shape_mask_shape[0] > 0) and (shape_mask_shape[1] > 0):
                for k, kernel in enumerate(kernel_list):

                    shape_mask = cv2.dilate(shape_mask.astype('uint8'), kernel).astype('bool')
                    block_features = np.sum(fd[y_inf:y_sup, x_inf:x_sup][shape_mask][:, 0, 0], axis = 0)
                    if k == 0:
                        direction = np.argmax(block_features)
                    vector += block_features

                vector = np.roll(n_orientations*vector/np.sum(vector), -direction)
                scale_vectors.append(vector.tolist())
                scale_directions.append(direction)

            else:
                scale_vectors.append((empty_vector.copy()).tolist())
                scale_directions.append(0)

        features.append(scale_vectors)
        directions.append(scale_directions)

    return np.asarray(features), np.asarray(directions), centroids


def computeMapExportIF(export_path, save_IF_path, anchor = False):

    paths = getPathsToProcess(export_path, save_IF_path)

    for path in tqdm.tqdm(paths, desc='Calcul des caractéristiques géométriques invariantes'):
        
        name = getImageName(path)     
        image = cv2.imread(path, 0)
        image[image >= 254] = 255
        image[image <= 1] = 0
        
        if anchor:
            image[image == 0] = np.int(255/2)
            image[0, :] = 0
            image[image.shape[0]-1, :] = 0
            image[:, 0] = 0
            image[:, image.shape[1]-1] = 0
            
        feat, dir_, cent = mapInvariantFeatures(image)

        with open(os.path.join(save_IF_path, name + '.json'), 'w') as outfile:
            json.dump({'feat': feat.tolist(), 'dir': dir_.tolist(), 
                       'cent': cent.tolist(), 'shape': image.shape}, outfile)


def loadAnchorIF(folder: str):

    with open(os.path.join(folder, 'anchor.json')) as data:
        data = json.load(data)
        feat, dir_ = np.asarray(data['feat']), np.asarray(data['dir'])
        cent, shape = np.asarray(data['cent']), data['shape']
        
        return feat, dir_, cent, shape
    
    
def computeTransform(feat1, feat2, cent1, cent2, dir1, dir2, shape1, img1_name, img2_name, 
                     settings, visualize_deformation = True, primary = True):
    
    def matchKeypoints(feat1, feat2, cent1, cent2, dir1, dir2, lowes_ratio = 0.85):

        X = pd.DataFrame(feat1).dropna()
        Y = pd.DataFrame(feat2).dropna()
                
        kp1_id = cent1[X.index]
        dir1 = dir1[X.index]
        kp2_id = cent2[Y.index]
        dir2 = dir2[Y.index]
        kp2_id = kp2_id.reshape(-1, 1, 2)

        X = X.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        tree = KDTree(X)
        if len(X) >= 2:
            dist, ind = tree.query(Y, k=2)
        else:
            dist, ind = [], np.zeros((len(Y), 2)).astype('uint8')

        good_mask = []

        for m1, m2 in dist:
            good_mask.append((m1 < m2*lowes_ratio))

        src_pts = kp1_id[ind[:,0]][good_mask]
        dst_pts = kp2_id[good_mask]
        
        dir1 = dir1[ind[:,0]][good_mask]
        dir2 = dir2[good_mask]
        
        return src_pts, dst_pts, dir1, dir2
    
    
    def computeMatchScore(rotation_vector, scale_ratio_vector, mask, dst, h, w, n_orientations):
        
        scale_ratio_vector = np.concatenate(scale_ratio_vector)
        rotation_vector = np.concatenate(rotation_vector)
        
        rotation = rotation_vector[mask.astype('bool')]
        scale_ratio = scale_ratio_vector[mask.astype('bool')]
        h_ = euclidean(dst[0,0], dst[1,0])
        w_ = euclidean(dst[0,0], dst[2,0])
        angle = 0.

        if (h_ > 0) and (w_ > 0):
            scale_coherence = 0.5*np.abs(np.log2(np.mean([h/h_, w/w_])) - np.mean(scale_ratio))

            opp = euclidean([dst[0,0,0], 0], [dst[1,0,0], 0])
            alpha = (180*np.arcsin(opp/h_))/np.pi
            angle = alpha/(360/n_orientations)
            rot_coherence = (1-(len(rotation[(rotation >= angle - 1) & (
                rotation <= angle + 1)])/(len(rotation)))**3)
        else:
            scale_coherence = 10
            rot_coherence = 1.

        regularizer = 2/np.log(np.sum(mask))

        coherence_score = scale_coherence + rot_coherence + regularizer
        
        return coherence_score, scale_coherence, rot_coherence, regularizer, rotation_vector, angle
    
    
    def matchMaps(feat1, feat2, cent1, cent2, dir1, dir2, shape1, lowes_ratio = 0.85, threshold = 200):
        
        (h, w) = shape1

        src_pts_list, dst_pts_list = [], []
        scale_ratio_vector, rotation_vector = [], []

        # Match keypoints
        for i in range(4):
            for j in range(4):
                src_pts, dst_pts, dir1_, dir2_ = matchKeypoints(feat1[i], feat2[j], cent1, cent2,
                                                              dir1[i].copy(), dir2[j].copy(), lowes_ratio)

                src_pts_list.append(src_pts)
                dst_pts_list.append(dst_pts)
                scale_ratio_vector.append((np.ones(len(src_pts))*i-j).astype('int8'))
                rotation_vector.append(dir1_-dir2_)

        # Merge both and compute transformation
        src_pts = np.concatenate(src_pts_list)
        dst_pts = np.concatenate(dst_pts_list)

        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold = threshold)

        mask = np.ravel(mask)

        pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
        dst = (pts@M[:,:2]) + M[:, 2:3].transpose()

        coherence_score, scale_coherence, rot_coherence, regularizer, rotation_vector, angle = computeMatchScore(
            rotation_vector, scale_ratio_vector, mask, dst, h, w, feat1.shape[-1])
        
        return coherence_score, scale_coherence, rot_coherence, regularizer, dst, M, src_pts, dst_pts, rotation_vector, angle

    
    best = {'dst': [], 'score': np.inf, 'subscore': (np.nan, np.nan, np.nan),
           'threshold': np.nan, 'lowes': np.nan, 'M': np.nan, 'src_pts': [], 'dst_pts': []}

    min_Lowes = settings['matching']['Lowes']['min']
    max_Lowes = settings['matching']['Lowes']['max']
    
    ratios = np.arange(min_Lowes, max_Lowes, (max_Lowes-min_Lowes)/settings['matching']['Lowes']['bins']).tolist()
    
    (h, w) = shape1
    
    for lowes_ratio in ratios:
        for threshold in settings['matching']['RANSAC']:

            coherence_score, scale_coherence, rot_coherence, regularizer, dst, M, src_pts, dst_pts, rotation, angle = matchMaps(
                feat1, feat2, cent1, cent2, dir1.copy(), dir2.copy(), shape1, lowes_ratio, threshold)
            
            if coherence_score < best['score']:
                
                src_pts_ = src_pts[(rotation >= angle - 1) & (rotation <= angle + 1)]
                dst_pts_ = dst_pts[(rotation >= angle - 1) & (rotation <= angle + 1)]
                
                if (len(src_pts_) > 0) and (len(dst_pts_) > 0):
                    M_, mask = cv2.estimateAffinePartial2D(src_pts_, dst_pts_, cv2.RANSAC, 
                                                          ransacReprojThreshold = threshold)

                    if not(M_ is None):
                        M = M_
                    
                    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
                    dst = (pts@M[:,:2]) + M[:, 2:3].transpose()
                    
                    mask = np.ravel(mask).astype('bool')
                    
                    best = {'dst': dst, 'score': coherence_score, 
                            'subscore': (scale_coherence, rot_coherence, regularizer),
                            'threshold': threshold, 'lowes': lowes_ratio, 'M': M, 
                            'src_pts': src_pts_[mask], 'dst_pts': dst_pts_[mask]}
                else:
                    best = {'dst': dst, 'score': coherence_score, 
                            'subscore': (scale_coherence, rot_coherence, regularizer),
                            'threshold': threshold, 'lowes': lowes_ratio, 'M': M, 
                            'src_pts': src_pts, 'dst_pts': dst_pts}

            if (coherence_score > 3) or (coherence_score > best['score']*1.25):
                break
                    
    errors = computeErrors(best['src_pts'], best['dst_pts'][:,0], best['M'])

    X, Y, Zx, Zy = addFringePoints(best['src_pts'], errors, shape1)
    
    assert((np.max(X) == shape1[0]) & (np.max(Y) == shape1[1])), 'There might be a confusion between \
source and destination images.'
    
    df = pd.DataFrame({'X': X, 'Y': Y, 'Zx': Zx, 'Zy': Zy})
    df = df.drop_duplicates(subset=['X', 'Y'], keep='first')
    X, Y = df['X'].values, df['Y'].values
    Zx, Zy = df['Zx'].values, df['Zy'].values
    
    dx, dy = computeDeformation(X, Y, Zx, Zy, shape1)

    project_name = getProjectName()

    deformation_coef = saveDeformation(dx, dy, img1_name, primary)
                                              
    best['deformation_coef'] = deformation_coef

    del best['src_pts']
    del best['dst_pts']

    best['dst'] = best['dst'][:,0].tolist()
    best['shape'] = shape1
    best['on'] = img2_name
    best['M'] = best['M'].tolist()

    return best
    

def saveDeformation(dx: np.ndarray, dy: np.ndarray, name: str, primary: bool = True):
    ''' Converts the dx and dy deformation to an image for storage. Lossy.
    Input(s):
        path: folder to store the deformation
        dx: map of the vertical deformation, at each pixel position in the 2nd image
        dy: map of the horizontal deformation, at each pixel position in the 2nd image
    '''

    project_name = getProjectName()

    if primary:
        export_path = os.path.join('export', project_name, 'deformation', 'primary')
    else:
        export_path = os.path.join('export', project_name, 'deformation', 'secondary')
    
    sign_channel = ((dx > 0) + 2*(dy > 0)).astype('uint8')
    coef = 255/np.max([np.max(np.abs(dx)), np.max(np.abs(dy))])
    if coef > 1:
        coef = 1
        
    dx_ = np.abs(np.around(dx*coef)).astype('uint8')
    dy_ = np.abs(np.around(dy*coef)).astype('uint8')
        
    compact = cv2.merge([dx_, sign_channel, dy_])
    cv2.imwrite(os.path.join(export_path, name + '.png'), compact)
    
    return coef


def loadDeformation(path):
    ''' Loads the stored deformation.
    Input(s):
        path: folder where the deformation is stored
        image_name: name of the 2nd image
    Output(s):
        dx: map of the vertical deformation, at each pixel position in the 2nd image
        dy: map of the horizontal deformation, at each pixel position in the 2nd image
    '''
    
    compact = cv2.imread(path)
    dx_, sign_channel, dy_ = cv2.split(compact)
    dx_, dy_ = dx_.astype('int16'), dy_.astype('int16')
    
    dx_[(sign_channel == 0) | (sign_channel == 2)] = -dx_[(sign_channel == 0) | (sign_channel == 2)]
    dy_[(sign_channel == 0) | (sign_channel == 1)] = -dy_[(sign_channel == 0) | (sign_channel == 1)]
    
    return dx_, dy_


def applyDeformation(img, dx, dy, coef):
    
    deformation_coef = 1/coef
    
    dx = cv2.resize(dx, (img.shape[1], img.shape[0]))*deformation_coef
    dy = cv2.resize(dy, (img.shape[1], img.shape[0]))*deformation_coef
    
    deformed = elasticTransform(img, dx, dy)
    
    return deformed
    