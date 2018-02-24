"""Utility functions"""

import csv
import tifffile as tiff
import numpy as np
import os

def get_list_from_csv(csv_file_path, parse_float=True, skip_header=False):
    """Given a CSV file, converts it to list"""
    def _parse_float_array(arr):
        return [float(item) for item in arr]

    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    parsed_list = csv_list

    if parse_float:
        parsed_list = [_parse_float_array(item) for item in csv_list]

    return parsed_list[1:] if skip_header else parsed_list

def plot_csv_on_rgb_tif(centroids, reference_img_path, tif_output_path, color=[10000, 0, 0]):
    """Given a CSV file, plots the co-ordinates in the CSV on a RGB TIF stack"""
    def _parse_int_array(arr):
        return [int(item) for item in arr]

    def _draw_square(image, coord, size=2):
        coord = _parse_int_array(coord)
        shape_z, shape_y, shape_x, _ = image.shape
        z_range = range(max(0, coord[0]-size), min(shape_z, coord[0]+size))
        y_range = range(max(0, coord[1]-size), min(shape_y, coord[1]+size))
        x_range = range(max(0, coord[2]-size), min(shape_x, coord[2]+size))

        for z in z_range:
            for y in y_range:
                for x in x_range:
                    image[z, y, x, :] = color

        return image

    img = tiff.imread(reference_img_path)


    if img.ndim == 3:
        shape_z, shape_y, shape_x = img.shape
        new_img = np.zeros((shape_z, shape_y, shape_x, 3))
        new_img[:, :, :, 0] = img
        new_img[:, :, :, 1] = img
        new_img[:, :, :, 2] = img
    elif img.ndim == 4:
        shape_z, shape_y, shape_x, _ = img.shape
        new_img = img

    for i, c in enumerate(centroids):
        new_img = _draw_square(new_img, c)

    tiff.imsave(tif_output_path, new_img.astype(np.uint16))

def plot_csv_on_tif(centroids, reference_img_path, tif_output_path):
    """Given a CSV file, plots the co-ordinates in the CSV on a TIF stack"""
    def _parse_int_array(arr):
        return [int(item) for item in arr]

    def _draw_square(image, coord, size=2):
        coord = _parse_int_array(coord)
        shape_z, shape_y, shape_x = image.shape
        z_range = range(max(0, coord[0]-size), min(shape_z, coord[0]+size))
        y_range = range(max(0, coord[1]-size), min(shape_y, coord[1]+size))
        x_range = range(max(0, coord[2]-size), min(shape_x, coord[2]+size))

        for z in z_range:
            for y in y_range:
                for x in x_range:
                    image[z, y, x] = 255

        return image

    ref_image = tiff.imread(reference_img_path)
    shape_z, shape_y, shape_x = ref_image.shape

    annotated_image = np.zeros((shape_z, shape_y, shape_x))

    for i, c in enumerate(centroids):
        annotated_image = _draw_square(annotated_image, c)

    tiff.imsave(tif_output_path, annotated_image.astype(np.uint8))

def write_list_to_csv(arr, csv_output_path):
    """Given a list, writes it to a CSV file"""
    with open(csv_output_path, 'w') as csv_file:
        for item in arr:
            csv_file.write(','.join([str(x) for x in item]) + '\n')

def create_intern_config():
    """Creates intern.cfg file from BOSS_TOKEN environment variable"""
    INTERN_CFG = 'intern.cfg'
    if os.path.exists(INTERN_CFG):
        os.remove(INTERN_CFG)

    boss_token = os.environ['BOSS_TOKEN']
    with open('intern_cfg_tmpl', 'r') as cfg_file:
        cfg_str = cfg_file.read()

    with open('intern.cfg', 'w') as cfg_file:
        cfg_file.write(cfg_str.format(boss_token))
