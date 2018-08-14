# Changes



## Release Process
1. Bump the version number (edit `srp/__version__.py`)
1. `make test`
1. `tox`
1. Update this file.
1. `make docs`
1. `git tag vX.X.X`

## Next Release
**Upcoming Version 0.0.1**  (expected 2018-08-27)

* [ ] Include a link to the thesis. 
* [x] Use `python -m srp.data.generate_csv` to generate sample metadata based on the config. 
* [x] Use `python -m srp.data.generate_patches` to generate positive and negative patches based on the config. 
* [x] Use `python -m srp.visualize.show_patches` to launch a GUI that shows the extracted patches. 
* [x] Use `python -m srp.data.train_val_split` to split samples into several folds
* [x] Use `python -m srp.data.generate_variations` to generate variations on the data.
* [ ] Use `python -m srp.model.train` to do training based on the current data and config settings. 
* [ ] Use `python -m srp.report` to generate reports based on the current model and config settings. 
  * [ ]  Generate a confusion matrix that can be pasted into LaTeX
  * [ ]  Report _Accuracy, Precision, Recall, F<sub>1</sub>_ for box detection (is a box in the window). 
  * [ ]  Report _mAP_ for box detecion; show the _AP_ plots. 
  * [ ]  If multiple folds exist, report stats as $\mu \pm \sigma$
  * [ ]  If multiple folds exist, show plots with error-bars. 
* [ ] Use `python -m srp.experiments` to generate a set of config files to run different experiments
* [ ] Setup a docker image (CPU and GPU)
* [ ] Include publicly accessible (Rather then SRP) dataset (Issue #1)
* [ ] Create a pytorch implementation of the network architecture(s)
* [x] Configure Sphinx documentation
* [x] Read configuration settings from local files using TOML
* [x] Fetch data from the MU Vision server as needed. 


