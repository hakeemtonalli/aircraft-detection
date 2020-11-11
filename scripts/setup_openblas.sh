# set 1 thread for OpenMP to suppress USE_OPENMP=1 error when using cpu
echo "export OMP_NUM_THREADS=1" >> ~/.bashrc 
echo "export OPENBLAS_NUM_THREADS=1" >> ~/.bashrc 
