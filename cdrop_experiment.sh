export PYTHONPATH=$PWD

# vector and width
#pushd data/experiments/cdrop/cdrop
#python -m srp.model.train | tee -a output.txt
#popd

#no-dropout
pushd data/experiments/cdrop/no_cdrop
python -m srp.model.train | tee -a output.txt
popd


echo "Goodbye!"
