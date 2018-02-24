"""This class is the core detector for this package"""

from tifffile import imread, imsave
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from skimage import measure, transform, morphology
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

    def __init__(self, tif_img_path, n_components=4):
        self.img = imread(tif_img_path)
        self.n_components = n_components

    def _gmm_cluster(self, img, data_points, n_components):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', verbose=2).fit(img.reshape(-1, 1)[::4])

        cluster_labels = gmm.predict(img.reshape(-1, 1))

        cluster_labels = cluster_labels.reshape(img.shape)
        c_id = np.argmax(gmm.means_)

        shape_z, shape_y, shape_x = img.shape
        new_img = np.ndarray((shape_z, shape_y, shape_x))
        np.copyto(new_img, self.img)
        new_img[cluster_labels == c_id] = 255
        new_img[cluster_labels != c_id] = 0

        self.thresholded_img = new_img

        return new_img

    def _get_physical_length(self, rprop):
        z, y, x = rprop.image.shape

        if z <= 2 or y <= 2 or x <= 2:
            return 0.0

        rescaled_roi = transform.rescale(rprop.image, scale=[0.5, 0.5, 5], multichannel=False)
        rescaled_roi[rescaled_roi != 0] = 255
        label_img = measure.label(rescaled_roi, background=0)

        rescaled_rprops = measure.regionprops(label_img)
        return rescaled_rprops[0].major_axis_length

    def _get_extended_region_props(self, region_props):
        extended_region_props = []
        for rprop in region_props:
            extended_region_props.append({
                'centroids': [round(rprop.centroid[0]), round(rprop.centroid[1]), round(rprop.centroid[2])],
                'major_axis_length': self._get_physical_length(rprop),
                'mean_intensity': rprop.mean_intensity,
                'volume_in_vox': rprop.area
            })
        return extended_region_props

    def get_blob_centroids(self):
        """
        Gets the blob centroids based on GMM thresholding, erosion and connected components
        """

        uniq = np.unique(self.img, return_counts=True)

        data_points = [p for p in zip(*uniq)]
        gm_img = self._gmm_cluster(self.img, data_points, self.n_components)

        eroded_img = morphology.erosion(gm_img)
        eroded_img = eroded_img.astype(np.uint8) * 255

        self.processed_img = eroded_img

        labeled_img = measure.label(eroded_img, background=0)

        extended_region_props = self._get_extended_region_props(measure.regionprops(labeled_img, self.img))

        centroids = [rprop['centroids'] for rprop in extended_region_props if rprop['major_axis_length'] > 0]
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
