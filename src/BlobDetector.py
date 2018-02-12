"""This class is the core detector for this package"""

from tifffile import imread
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from skimage import measure
from skimage.morphology import binary_erosion
import scipy.stats
from tqdm import tqdm

__docformat__ = 'reStructuredText'

class BlobDetector(object):


    """
    BlobDetector class can be instantiated with the following args

    - **parameters**, **types**, **return** and **return types**::
    :param tif_img_path: full path of the input TIF stack
    :param data_source: either 'laVision' or 'COLM' - the imaging source of the input image
    :type tif_img_path: string
    :type data_source: string
    """

    def __init__(self, tif_img_path, data_source):
        self.img = imread(tif_img_path)
        self.n_components = 2 if data_source == 'COLM' else 3

    def _gmm_cluster(self, img, data_points, n_components):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full').fit(data_points)

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

        self.threshold = avg_intensity
        self.gmm = gmm

        return new_img

    def get_blob_centroids(self, min_diameter=None, max_diameter=None):
        """
        Gets the blob centroids based on GMM thresholding, erosion and connected components
        """

        uniq = np.unique(self.img, return_counts=True)

        data_points = [p for p in zip(*uniq)]
        gm_img = self._gmm_cluster(self.img, data_points, self.n_components)

        eroded_img = binary_erosion(gm_img)

        if self.n_components == 2:
            labeled_img = measure.label(gm_img, background=0)
        else:
            labeled_img = measure.label(eroded_img, background=0)

        self.labeled_img = labeled_img

        region_props = [x for x in measure.regionprops(labeled_img)]
        if min_diameter:
            region_props = [x for x in region_props if x.major_axis_length >= min_diameter]

        if max_diameter:
            region_props = [x for x in region_props if x.major_axis_length <= max_diameter]

        centroids = [[round(x.centroid[0]), round(x.centroid[1]), round(x.centroid[2])] for x in region_props]
        return centroids

    def get_avg_intensity_by_region(self, reg_atlas_path):
        """
        Given registered atlas image path, gives the average intensity of the regions
        """

        reg_img = imread(reg_atlas_path).astype(np.uint16)
        raw_img = self.img.astype(np.uint16)

        region_numbers = np.unique(reg_img, return_counts=True)[0]

        region_intensities = {}

        rgn_pbar = tqdm(region_numbers)


        for rgn in rgn_pbar:
            rgn_pbar.set_description('Summing intensities of region {}'.format(rgn))

            voxels = np.where(reg_img == rgn)
            voxels = map(list, zip(*voxels))
            region_intensities[str(rgn)] = float(np.sum([raw_img[v[0], v[1], v[2]] for v in voxels]))

        return region_intensities
