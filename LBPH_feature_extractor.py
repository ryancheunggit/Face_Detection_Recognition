# authoer: ryanzjlib@gmail.com

from skimage import feature
import numpy as np

class LocalBinaryPatterns:
	# constructor
	def __init__(self, n_points, radius, n_bins):
		self.n_points = n_points
		self.radius = radius
		self.n_bins = n_bins

	def describe(self, image):
		# extract local binary pattern
		lbp = feature.local_binary_pattern(
			image,
			self.n_points,
			self.radius,
			method = "uniform")

		# calculate the histogram 
		hist, _ = np.histogram(
			lbp.ravel(),
			normed = True,
			bins = np.arange(0, self.n_bins),
			range = (0, self.n_bins)
			)

		# normalize the histogram
		hist = hist.astype("float")
		hist /= np.linalg.norm(hist)

		return hist

