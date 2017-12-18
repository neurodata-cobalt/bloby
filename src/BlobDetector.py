from tifffile import imread
import numpy as np
from sklearn.mixture import GaussianMixture
from skimage import measure
from skimage.morphology import binary_erosion
import scipy.stats

class BlobDetector(object):

    def __init__(self, tif_img_path, n_components=3):
        self.img = imread(tif_img_path)
        self.n_components = n_components

    def _gmm_cluster(self, img, data_points, n_components):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full').fit(data_points)
        cluster_labels = gmm.predict(data_points)
        cluster_centers = np.empty((n_components, len(data_points[0])))

        for i in range(n_components):
            density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i], allow_singular=True).logpdf(data_points)
            cluster_centers[i, :] = data_points[np.argmax(density)]

        cluster_int = [p[0] for p in cluster_centers]
        cluster_int.sort()

        max_intensity = cluster_int[::-1][0]
        medium_intensity = cluster_int[::-1][1]

        avg_intensity = (float(max_intensity) + float(medium_intensity))/2.0

        shape_z, shape_y, shape_x = img.shape

        new_img = np.ndarray((shape_z, shape_y, shape_x))
        np.copyto(new_img, img)

        new_img[img >= avg_intensity] = 255
        new_img[img < avg_intensity] = 0

        return new_img

    def get_blob_centroids(self):
        uniq = np.unique(self.img, return_counts=True)

        data_points = [p for p in zip(*uniq)]
        gm_img = self._gmm_cluster(self.img, data_points, self.n_components)

        eroded_img = binary_erosion(gm_img)

        if self.n_components == 2:
            labeled_img = measure.label(gm_img, background=0)
        else:
            labeled_img = measure.label(eroded_img, background=0)

        centroids = [[round(x.centroid[0]), round(x.centroid[1]), round(x.centroid[2])] for x in measure.regionprops(labeled_img)]
        return centroids
