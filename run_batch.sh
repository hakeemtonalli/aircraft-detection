# run with rational quadratic Kernel
echo "Beginning batch process"

start = `date + %s`
python run.py --kernel="RQ_Kernel"
end = `date + %s`
echo $((end - start))

# run with rbf kernel
start = `date + %s`
python run.py --kernel="RBF_Kernel"
end = `date + %s`
echo $((end - start))

# run with linear kernel
start = `date + %s`
python run.py --kernel="Linear_Kernel"
end = `date + %s`
echo $((end - start))

# run with quadratic kernel
start = `date + %s`
python run.py --kernel="Quadratic_Kernel"
end = `date + %s`
echo $((end - start))

# run with cubic kernel
start = `date + %s`
python run.py --kernel="Cubic_Kernel"
end = `date + %s`
echo $((end - start))