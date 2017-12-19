import csv
import tifffile as tiff
import numpy as np

def get_list_from_csv(csv_file_path, parse_float=True, skip_header=False):
    def _parse_float_array(arr):
        return [float(item) for item in arr]

    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    parsed_list = csv_list

    if parse_float:
        parsed_list = [_parse_float_array(item) for item in csv_list]

    return parsed_list[1:] if skip_header else parsed_list

def plot_csv_on_tif(centroids, reference_img_path, tif_output_path):
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
    with open(csv_output_path, 'w') as csv_file:
        for item in arr:
            csv_file.write(','.join([str(x) for x in item]) + '\n')