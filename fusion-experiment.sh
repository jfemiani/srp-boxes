export PYTHONPATH=$PWD

# Early Fusion
pushd data/experiments/fusion/early
python -m srp.model.train | tee -a output.txt
popd

# Late with Adding
pushd data/experiments/fusion/late_add
python -m srp.model.train | tee -a output.txt
popd


# Late with Concatenation
pushd data/experiments/fusion/late_cat
python -m srp.model.train | tee -a output.txt
popd



echo "Goodbye!"
