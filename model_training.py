import os
import cv2
from sklearn.grid_search import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from LBPH_feature_extractor import LocalBinaryPatterns
import pickle
import numpy as np
import pandas as pd

def main():
	# load images and training labels
	path_to_images = 'img/'

	filenames = os.listdir(path_to_images)

	labels = [filename.split("_")[0] 
			  for filename in filenames]

	le = LabelEncoder()
	le.fit(labels)
	y = le.transform(labels)

	images = [cv2.imread(os.path.join(path_to_images, filename), 0)
	          for filename in filenames]


	# set up search grid
	search_grid = ParameterGrid({
		'n_points': [8,12,16,20,24],
		'radius': [6,8,10,12],
		'n_bins': [4,6,8,10],
		'k' : [3,4,5]
		})

	results = []

	for params in search_grid:
		# set up feature extractor
		extractor = LocalBinaryPatterns(
			params['n_points'],
			params['radius'],
			params['n_bins']
			)
		# extract LBPH features
		features = np.array([extractor.describe(image)
			                 for image in images])
		# fit knn model
		model = KNeighborsClassifier(n_neighbors= params['k'])
		model.fit(features, y)
		score = model.score(features, y)
		result = params.values()
		result.append(score)
		results.append(result)

	results = pd.DataFrame(results)
	results.columns = ['k', 'n_bins', 'n_points', 'radius', 'accuracy_score']

	print results
	best_result = results.loc[results['accuracy_score'] ==  np.max(results.accuracy_score)].iloc[0]

	extractor = LocalBinaryPatterns(
		best_result.n_points,
		best_result.radius,
		best_result.n_bins
		)

	features = np.array([extractor.describe(image) for image in images])

	model = KNeighborsClassifier(n_neighbors= best_result.k)
	model.fit(features, y)

	with open("trained_model.pkl", "wb") as f:
		pickle.dump((features, le, extractor, model), f)

if __name__ == '__main__':
	main()

