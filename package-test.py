from src.BlobDetector import BlobDetector
from src.BlobMetrics import BlobMetrics
from src.IngestTifStack import IngestTifStack, ConfigParams
from src import util

detector = BlobDetector('data/s3617/tifs/cell_detection_0.tiff', n_components=2)
centroids = detector.get_blob_centroids()

util.plot_csv_on_tif(centroids, 'data/s3617/tifs/cell_detection_0.tiff', 'data/s3617/prediction_tifs/cell_detection_0.tiff')

ground_truth = util.get_list_from_csv('data/s3617/annotations/cell_detection_0.csv')

metrics = BlobMetrics(ground_truth, centroids, 12)
print(metrics.precision(), metrics.recall())
#metrics.plot_predictions_per_ground_truth()
#metrics.plot_ground_truths_per_prediction()

ingest_conf = {
    'collection': 'cell_detection',
    'experiment': 'cell_detection_0',
    'channel': 'test_annotation_ignore',
    'tif_stack': 'data/s3617/prediction_tifs/cell_detection_0.tiff',
    'type': 'annotation',
    'new_channel': True,
    'source_channel': 'raw_data',
    'config': 'intern.cfg'
}

ingest = IngestTifStack(ConfigParams(ingest_conf))
ingest.start_upload(group_name='ndd17_claritrons')
