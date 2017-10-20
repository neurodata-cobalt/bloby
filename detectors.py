from scipy import (
    ndimage,
    signal
)
from image_processing import (
    gradient_x,
    gradient_y,
    gradient_z,
    BlobCandidate,
    voxel_region_iter,
    raster_3d_generator,
    block_principal_minors,
    img_hessian_3d,
    format_H,
    regional_blobness,
    regional_flatness,

)
from util import bound_check
import numpy as np
import progressbar

DOG_STACK_SIZE = 5

def DoG(img, gamma = 2, dark = 1, sigma = 2, print_level = 0):
    # if the image has white blobs then invert them to being dark blobs
    if not dark:
        img = 1 - img

    # Differential difference
    a = 0.01
    DoG_stack = []
    sigma_range = np.linspace(sigma, sigma+10, DOG_STACK_SIZE)
    if print_level:
        bar = progressbar.ProgressBar()
        sigma_range = bar(sigma_range)
    for sigma in sigma_range:
        scale_constant = np.power(sigma, gamma - 1)
        # TODO: Do we need a inhouse gaussian filter to control filter size?
        G_1 = ndimage.filters.gaussian_filter(img, sigma+a)
        G_2 = ndimage.filters.gaussian_filter(img, sigma)
        DoG = scale_constant * (G_1 - G_2)/a
        DoG_stack.append((sigma,DoG))
    return DoG_stack

def blob_descriptors(DoG_img, intensity_img, sigma, U):
    blob_candidates_T = []
    H = img_hessian_3d(DoG_img)
    H_filter = np.zeros(DoG_img.shape)
    z_range, y_range, x_range = intensity_img.shape
    for i_c, j_c, k_c in U:
        # For each blob candidate, calculate the regional hession
        region_iter = voxel_region_iter(i_c, j_c, k_c)
        regional_H = np.copy(H[i_c,j_c,k_c,:])
        # TODO: Check is it average intenity of original or DoG?
        intensity = intensity_img[i_c, j_c, k_c]
        for i_r, j_r, k_r in region_iter:
            if bound_check(i_r, 0, z_range-1) and bound_check(j_r, 0, y_range-1) and bound_check(k_r, 0, x_range-1):
                intensity += intensity_img[i_r, j_r, k_r]
                regional_H += H[i_r, j_r, k_r,:]
        regional_H = format_H(regional_H)
        blobness = regional_blobness(regional_H)
        flatness = regional_flatness(regional_H)
        avg_int = intensity/7
        blob_candidates_T.append(
            BlobCandidate(
                (i_c, j_c, k_c ),
                sigma,
                blobness,
                flatness,
                avg_int
            )
        )
    return blob_candidates_T

def find_negative_curvative_points(img):
    img_fx, img_fy, img_fz = gradient_x(img), gradient_y(img), gradient_z(img)
    img_fxy, img_fxz, img_fyz = gradient_y(img_fx), gradient_z(img_fx), gradient_z(img_fy)
    img_fxx, img_fyy, img_fzz = gradient_x(img_fx), gradient_y(img_fy), gradient_z(img_fz)
    D_1, D_2, D_3, H_N = np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape)
    D_1[img_fxx < 0] = 1

    D_2[(img_fxx * img_fyy - img_fxy * img_fxy) > 0] = 1
    D_3[(
            img_fxx * (img_fyy*img_fzz - img_fyz*img_fyz) \
            - img_fxy * (img_fxy*img_fzz - img_fyz*img_fxz) \
            + img_fxz * (img_fxy*img_fyz - img_fyy * img_fxz)
        ) < 0
    ] = 1

    H_N[np.logical_and(D_1 == D_2, D_2 == D_3)] = 1
    return np.argwhere(H_N == 1)
