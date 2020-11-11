
echo "Beginning batch process"

# run with rational quadratic Kernel
python run.py --kernel="RQ_Kernel"

# run with rbf kernel
python run.py --kernel="RBF_Kernel"

# run with linear kernel
python run.py --kernel="Linear_Kernel"

# run with quadratic kernel
python run.py --kernel="Quadratic_Kernel"

# run with cubic kernel
python run.py --kernel="Cubic_Kernel"
