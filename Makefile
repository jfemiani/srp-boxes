


debug: requirements
	pytest --doctest-modules -x --pdb

requirements: requirements.txt
	pip install --quiet -r requirements.txt

test:
	pytest --doctest-modules 

docs:
	cd docs && make html


