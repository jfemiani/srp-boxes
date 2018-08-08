
.PHONY:data docs requirements debug test

debug: requirements
	pytest --doctest-modules -x --pdb

requirements: requirements.txt
	pip install --quiet -r requirements.txt

test: requirements
	pytest --doctest-modules 

docs: requirements
	cd docs && make html

data:
	$(MAKE) -C data data
