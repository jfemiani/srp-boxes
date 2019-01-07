export PYTHONPATH=$PWD

# vector and width
#pushd data/experiments/obb/vector_and_width
#python -m srp.model.train | tee -a output.txt
#popd

# two vectors
pushd data/experiments/obb/two_vectors
python -m srp.model.train | tee -a output.txt
popd


# Late with Concatenation
pushd data/experiments/obb/four_points
python -m srp.model.train | tee -a output.txt
popd



echo "Goodbye!"
