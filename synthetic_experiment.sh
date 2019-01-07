export PYTHONPATH=$PWD

# L1
pushd data/experiments/synthetic/pretrain
python -m srp.model.train | tee -a output.txt
popd

# L2
#pushd data/experiments/synthetic/no_pretrain
#python -m srp.model.train | tee -a output.txt
#popd

echo "Goodbye!"
