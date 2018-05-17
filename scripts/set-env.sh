rm -rf ndmulticore process_folder progress.log final.csv
git clone https://github.com/neurodata-cobalt/nd-multicore ndmulticore
export PYTHONPATH=`pwd`:`pwd`/ndmulticore
