export PYTHONPATH=$PWD

# xent_loss
pushd data/experiments/classification_loss/xent_loss
python -m srp.model.train | tee -a output.txt
popd

# hinge_loss
pushd data/experiments/classification_loss/hinge_loss
python -m srp.model.train | tee -a output.txt
popd

echo "Goodbye!"
