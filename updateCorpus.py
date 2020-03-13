from utils.utils import getProjectName, getImageName
import pandas as pd
import os, glob


def updateCorpus():
	project_name = getProjectName()

	df = pd.read_excel(os.path.join('data', project_name, 'data.xlsx')).drop(columns = ['Unnamed: 0'])

	files = glob.glob(os.path.join('data', project_name, 'maps', '*.*'))

	names, leaflets = [], []

	for file in files:
		image_name = getImageName(file)
		names.append(image_name[:-2])
		leaflets.append(image_name[-1:])

	df_names = pd.DataFrame({'ark': names, 'leaflet': leaflets})
	df_names['leaflet'] = df_names['leaflet'].astype('int64')

	new_df = pd.merge(df, df_names, on = ['ark', 'leaflet'], how = 'inner')
	new_df = new_df.drop_duplicates().reset_index(drop=True)
	new_df.to_excel(os.path.join('data', project_name, 'data.xlsx'))

	df_names['path'] = files
	df_files = pd.merge(df_names, df, on = ['ark', 'leaflet'], how = 'inner')[['ark', 'path', 'leaflet']]

	keep_files = df_files['path'].values.tolist()

	for file in files:
	    if not(file in keep_files):
	        os.remove(file)

	print('Le corpus a été mis à jour.')


updateCorpus()
