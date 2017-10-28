# Blob Detection package

Pipeline for detecting cells in clarity images.

Our first iteration is to implement [Efficient Small Blob Detection Based on Local Convexity, Intensity and Shape Information](http://ieeexplore.ieee.org/abstract/document/7359134/).
And evaluate it on synthetic and manually annotated data.

The package performs the following (tentative list):
1. Reads in tif image
2. Normalizes it
3. Computes the DoG at 10 different scales
4. Computes the Fast hessian based on the 3 principal minors
5. Discovers possible blob candidates by finding all points with a negative-definite hessian (aka concave regions) along with its 6-connected component
6. Calculates blob descriptors namely the "blobness", "flatness", and average intensity of blob candidate regions
7. TODO: Post-prune the blob candidates using a clustering algorithm to find the true blobs

Backlog of Dev tasks:
1. Add unit testing
2. Add checkpoint support
 * Like how tensorflow has checkpoints after computaionally intensive steps, we should incorporate checkpoints after certain steps of our pipeline. Proposed checkpoints: after DoG, after finding blob candidates, between finding the blob descriptors for different scales.
3. Add more preprocessing
4. Add more logging and informative outputs
5. Add support for configuration files
 * Like how Microsoft's CNTK has configuration files that hold all the parameters for the models, we should have a configuration file option for users to specify all the parameters they want to use.
6. Configure Travis CI Pipeline
7. Push the centroids marked data as an annotated channel to boss
