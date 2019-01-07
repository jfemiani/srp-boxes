export PYTHONPATH=$PWD

# L1
#pushd data/experiments/regression_loss/smooth_L1
#python -m srp.model.train | tee -a output.txt
#popd

# L2
pushd data/experiments/regression_loss/L2
python -m srp.model.train | tee -a output.txt
popd

echo "Goodbye!"
