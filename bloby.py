from src.BlobDetector import BlobDetector
from src import util
from src.BlobMetrics import BlobMetrics

print('**************************DETECT BLOBS**************************')

input_tif_path = input('Enter input TIF path: ')
output_csv_path = input('Enter path to save CSV file: ')
n_components = int(input('Enter number of components for GMM [Default: 2]: ') or 2)

detector = BlobDetector(input_tif_path, n_components=n_components)
centroids = detector.get_blob_centroids()
util.write_list_to_csv(centroids, output_csv_path) #writing the detection output to CSV in (z,y,x) format

print('Centroids saved in (z,y,x) format to {}'.format(output_csv_path))


print('**************************QUANTITATIVE EVALUATION**************************')

ground_truth_path = input('Enter ground truth CSV path: ')
ground_truth = util.get_list_from_csv(ground_truth_path)

metrics = BlobMetrics(ground_truth, centroids, euclidean_distance_threshold=12)
print('Precision: {}\nRecall: {}'.format(metrics.precision(), metrics.recall()))

metrics.plot_predictions_per_ground_truth(fname='pr_gt.png')
metrics.plot_ground_truths_per_prediction(fname='gt_pr.png')

print('**************************BOSS UPLOAD**************************')

output_tif_path = input('Enter path to save output TIF: ')
util.plot_csv_on_tif(centroids, input_tif_path, output_tif_path)

from src.IngestTifStack import ConfigParams

collection = input('Enter BOSS collection name: ')
exp_name = input('Enter BOSS experiment name: ')
channel_name = input('Enter new channel name: ')
config_file = input('Enter path to your intern.cfg file [.]: ') or '.'
config_file += 'intern.cfg' if config_file.endswith('/') else '/intern.cfg'

group_name = input('Enter group name if you want to grant permissions for the new channel: ')

ingest_conf = {
    'collection': collection,
    'experiment': exp_name,
    'channel': channel_name,
    'tif_stack': output_tif_path,
    'type': 'annotation',
    'new_channel': True,
    'source_channel': 'raw_data',
    'config': config_file
}
params = ConfigParams(ingest_conf)

from src.IngestTifStack import IngestTifStack

ingest = IngestTifStack(params)
upload_link = ingest.start_upload(group_name=group_name)
print('Results uploaded to {}'.format(upload_link))
