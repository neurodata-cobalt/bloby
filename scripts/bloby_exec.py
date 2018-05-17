from ndmulticore import parallel
import bloby.util as util
from bloby.IngestTifStack import ConfigParams, IngestTifStack
import numpy as np
import configparser
import sys
from tifffile import imsave, imread

config_file = sys.argv[1] if len(sys.argv) >= 2 else 'neurodata.cfg'

config = configparser.ConfigParser()
config.read(sys.argv[1])

boss_resource_config = 'neurodata.cfg'
module_name = 'bloby.BlobDetector'
function_name = 'multicore_handler'
output_file = config['Bloby']['output_file']

parallel.start_process(module_name, function_name, output_file, boss_resource_config)
print('Centroids saved to {}'.format(output_file))

centroids = util.get_list_from_csv(output_file)
save_path = config['Bloby']['save_path']

z_shape = int(config['Parallel']['z_range'].split(',')[1])
y_shape = int(config['Parallel']['y_range'].split(',')[1])
x_shape = int(config['Parallel']['x_range'].split(',')[1])

img = np.zeros((z_shape, y_shape, x_shape))
imsave(save_path, img.astype(np.uint16))
util.plot_csv_on_tif(centroids, save_path, save_path)

ingest_conf = {
    'collection': config['Bloby']['collection'],
    'experiment': config['Bloby']['experiment'],
    'channel': config['Bloby']['channel'],
    'type': 'image',
    'new_channel': True,
    'source_channel': None,
    'config': config_file,
    'tif_stack': save_path,
    'x_range': [],
    'y_range': [],
    'z_range': []
}
params = ConfigParams(ingest_conf)
#group_name = config['Bloby']['group_name']
ingest = IngestTifStack(params)
upload_link = ingest.start_upload()
print('Results uploaded to {}\n\n'.format(upload_link))
