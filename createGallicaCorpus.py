from utils.utils import getProjectName
import urllib, xmltodict, re, time, os
import glob, requests, shutil, tqdm, json
import numpy as np
import pandas as pd
import argparse

print('\n','Description :')
print('Constitution automatique d\'un corpus via l\'API Gallica.')

print('\n','Options :')
print('--startYear : date de début du corpus (pour requêtage simplifié)')
print('--endYear : date de fin du corpus (pour requêtage simplifié)')
print('--SRU : requête SRU Gallica (pour requêtage avancé)', '\n')

parser = argparse.ArgumentParser()

parser.add_argument('--startYear', action="store", dest="startYear")
parser.add_argument('--endYear', action="store", dest="endYear")
parser.add_argument('--SRU', action="store", dest="SRU")

options = parser.parse_args()

project_name = getProjectName()

with open(os.path.join('settings', project_name, 'settings.json')) as settings:
    settings = json.load(settings)

if ((options.SRU) != None):
    print('Requêtage avancé.')
    if ((options.startYear != None) or (options.endYear != None)):
        print('Avertissement: Vous avez fourni une requête SRU complète. Par conséquent, les paramètres startYear et endYear seront ignorés.')
else:
    print('Requêtage simplifié.')

    print('Cartes de', settings['corpus']['city_name'], end=' ')
    query = '(dc.title%20all%20%22' + settings['corpus']['city_name'] + '%22%20%20and%20(dc.type%20all%20%22carte%22)'

    if options.startYear != None:
        print('à partir de', options.startYear, end=' ')
        query += '%20and%20(gallicapublication_date%3E=%22' + str(options.startYear) + '%22)'

    if options.endYear != None:
        print('jusqu\'en', options.endYear, end='')
        query += '%20and%20(gallicapublication_date%3C=%22' + str(options.endYear) + '%22)'

    print('')
    query += '%20and%20((bibliotheque%20adj%20%22Biblioth%C3%A8que%20nationale%20de%20France%22))%20and%20(provenance%20adj%20%22bnf.fr%22)&suggest=10&keywords='
    query += settings['corpus']['city_name'] + ')'
    
    
print('\n Si vous le souhaitez, entrez un ou plusieurs mots-clés "à éviter". Sinon, laissez vide et appuyez sur ENTER.', '\n')
forbidden = re.findall('([A-ÿ]+)', str(input()))

# Initialize search
base = 'https://gallica.bnf.fr/SRU?operation=searchRetrieve&version=1.2&query='
prefix = 'https://gallica.bnf.fr/iiif/ark:/'
query = str(query.encode())[2:-1].replace('\\x', '%')
startRecord = 1
totalRecords = 999
suffix = ''

# Information to scrap from the API result page
metadata = ['dc:coverage', 'dc:date', 'dc:description', 'dc:identifier', 'dc:language', 'dc:publisher',
            'dc:relation', 'dc:subject', 'dc:source', 'dc:format', 'dc:title']

# Initialize dataframe
dataframe = {}
for indicator in metadata:
    dataframe[indicator[3:]] = []


# May crash due to low speed internet connexion, just re-run this section in this case
# The dots are just for you to follow the progression of the extraction

attempts, success = 0, False
while (attempts < 5) and not(success):
    attempts += 1

    try:
        # Iterate over batches of 50 results
        while startRecord <= totalRecords:

            # Access the Gallica API result page and read it
            url = "".join([base, query, '&startRecord=', str(startRecord), '&maximumRecords=50'])
            s = urllib.request.urlopen(url)
            contents = s.read()
            dico = xmltodict.parse(contents)

            # For each result, scrap a number of information
            for record in dico['srw:searchRetrieveResponse']['srw:records']['srw:record']:
                for indicator in metadata:
                    try:
                        dataframe[indicator[3:]].append(record['srw:recordData']['oai_dc:dc'][indicator])
                    except:
                        dataframe[indicator[3:]].append('')

            # Iterate
            if startRecord == 1:
                totalRecords = int(dico['srw:searchRetrieveResponse']['srw:numberOfRecords'])
            startRecord += 50
            
        success = True
        
    except:
        print("La requête n'a pas pu être menée à bien. Nouvel essai dans 10 secondes.")
        time.sleep(10)


df = pd.DataFrame(dataframe)
print('Nombre de résultats issus de la requête SRU:', len(df))


def extractScale(string):
    """ Extract the scale of the map from a string containing the information in a format close from one of
    the following: 'Echelle 1:10000' or 'échelle: 1: 1.000' or even '1:10 000'. Returns the lower term of fraction.
    Input:
        string(str)
    Output:
        scale(int or float) """
    
    scale = np.nan
    if ':' in string:
        integers = []
        
        for integer in re.findall('([0-9 \.]+)', string):
            digit = integer.replace(' ', '').replace('.', '')
            if len(digit) > 0:
                integers.append(int(digit))
        if (len(integers) == 2) and (integers[0] == 1):
            scale = integers[1]

    return scale


scale = []

for ind, item in df['description'].iteritems():
    
    entries = str(item).replace(',', '#').replace('[', '#').replace(']', '#').split('#')
    scales = []
    
    for entry in entries:
        scale_entry = extractScale(entry)
        if not(np.isnan(scale_entry)):
            scales.append(scale_entry)
    if len(scales) > 0:
        scale.append(scales[0])
    else:
        scale.append(np.nan)
        
df['scale'] = scale


def checkScale(scale, max_scale: int = 1000, min_scale: int = 100000):
    
    # If the scale is neither too big nor too small (if this can be checked)
    if np.isnan(scale):
        return True

    else:
        scale_too_big, scale_too_small = True, True

        if scale >= max_scale:
            scale_too_big = False
        if scale <= min_scale:
            scale_too_small = False

        return (not(scale_too_big) and not(scale_too_small))
        

ark = []

for ind, item in df['identifier'].iteritems():
    if str(type(item)) == "<class 'str'>":
        ark.append(item[34:])
    else:
        for i in item:
            if prefix in i:
                ark.append(i[34:])

df['ark'] = ark


# For language: simplify data, keeping only the language short form
# For geolocation: extract from coverage
# For coverage: remove geolocation infos
# For n_items: extract from format

languages, geolocations, coverages, n_items, date, publisher_city = [], [], [], [], [], []

for ind, row in df[['language', 'coverage', 'format', 'date', 'publisher']].iterrows():

    if row['language'] != '':
        languages.append(row['language'][0])
    else:
        languages.append('')

    coverage, geolocation = row['coverage'], ''
    if str(type(row['coverage'])) == "<class 'list'>":
        if str(type(row['coverage'][-1])) == "<class 'str'>":
            if '°' in row['coverage'][-1]:
                coverage = row['coverage'][0]
                geolocation = row['coverage'][-1]

    coverages.append(coverage)
    geolocations.append(geolocation)

    try:
        n_items.append(int(re.findall('([0-9])+', row['format'][-1])[-1]))
    except:
        n_items.append(int(row['format'][0][0]))

    try:
        if len(str(row['date'])) > 4:
            date.append(int(str(row['date'])[:4]))
        else:
            date.append(int(row['date']))
    except:
        date.append(np.nan)

    res = re.findall(r'\((.*?)\)', str(row['publisher']))

    if len(res) > 0:
        publisher_city.append(res[0])
    else:
        publisher_city.append(np.nan)

df['language'], df['geolocation'], df['coverage'] = languages, geolocations, coverages
df['n_items'], df['date'] = n_items, date
df['publisher_city'] = publisher_city
print("Nombre total d'items trouvés:", np.sum(n_items))


latitude, longitude = [], []
for ind, coord in df['geolocation'].iteritems():

    long, lat = [], []
    vector = coord.replace(' - ', '#').replace(' / ', '#').split('#')

    if len(vector) < 2 and vector[0] == '': 
        latitude.append(np.nan)
        longitude.append(np.nan)
    else:
        for entry in vector:
            is_latitude, is_NE = None, None

            if 'N' in entry:
                is_latitude, NE = True, True
            elif 'S' in entry:
                is_latitude, NE = True, False
            elif 'E' in entry:
                is_latitude, NE = False, True
            elif 'W' in entry:
                is_latitude, NE = False, False
            else:
                latitude.append(np.nan)
                longitude.append(np.nan)
                break

            value = np.nan
            if '°' in entry:
                sub = entry[2:].split('°')
                value = float(sub[0])
                sub = sub[1:]
                if len(sub[0]) > 0:
                    if "'" in entry:
                        sub = sub[0].split("'")
                        value += float(sub[0])/60
                        sub = sub[1:]
                        if len(sub[0]) > 0:
                            if '"' in entry:
                                sub = sub[0].split('"')
                                value += float(sub[0])/3600
                    elif '"' in entry:
                        sub = sub[0].split('"')
                        value += float(sub[0])/60
            elif "'" in entry:
                sub = entry[2:].split("'")
                value = float(sub[0])/60
                sub = sub[1:]
                if len(sub[0]) > 0:
                    if '"' in entry:
                        sub = sub[0].split('"')
                        value += float(sub[0])/3600
            elif '"' in entry:
                sub = entry[2:].split('"')
                value = float(sub[0])/3600

            if is_NE and not(value.isnan()):
                value = -value

            if is_latitude:
                lat.append(value)
            else:
                long.append(value)

        latitude.append(sorted(lat))
        longitude.append(sorted(long))

df['latitude'], df['longitude'] = latitude, longitude  


keep_ind, throw_ind = [], []

for ind, row in df[['subject', 'description', 'scale', 'coverage', 'title']].iterrows():
    
    keep = True
    subject, coverage = str(row['subject']).lower(), str(row['coverage']).lower()
    
    for word in forbidden:
        if (word.lower() in subject) or (word.lower() in coverage):
            keep = False
        
    if keep:
        if checkScale(row['scale'], max_scale = settings['corpus']['max_scale'], min_scale = settings['corpus']['min_scale']):
            keep_ind.append(ind)
        else:
            throw_ind.append(ind)
    else:
        throw_ind.append(ind)


df = df.iloc[keep_ind]

print('Arks conservées après suppression des entrées ne correspondant pas aux critères d\'exclusion par mots-clés et par échelle:', len(keep_ind))
print('Arks écartées:', len(throw_ind))

arks, leaflets = [], []
for ind, row in df[['ark', 'n_items']].iterrows():
    for i in range(row['n_items']):
        arks.append(row['ark'])
        leaflets.append(i+1)
df_ = pd.DataFrame({'ark': arks, 'leaflet': leaflets})
df = pd.merge(df, df_, on='ark', how='outer').drop(columns = 'n_items')

print("Nombre total d'items conservés:", len(df))

df = df.reset_index(drop=True)
df.to_excel(os.path.join('data', project_name, 'data.xlsx'))

region = 'full'
size = 'full'
rotation = '0'
quality = 'native'
imgformat = 'jpg'

path = os.path.join('data', project_name, 'maps')
present_files = glob.glob(os.path.join(path, '*.jpg'))

for ind, row in tqdm.tqdm(df[['identifier', 'ark', 'leaflet']].iterrows()):
    
    leaflet = 'f' + str(row['leaflet'])
    url = "".join([prefix, '12148/', row['ark'], '/', leaflet, '/', 
                   region, '/', size, '/', rotation, '/', quality, '.', imgformat])
            
    filename = os.path.join(path, row['ark'] + leaflet + '.' + imgformat)
            
    if not(filename in present_files):
        
        response = requests.get(url, stream=True)
        
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
            del response


print('__________________________________________________________________________ \n')

