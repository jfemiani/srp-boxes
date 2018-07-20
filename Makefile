


debug: requirements
	pytest --doctest-modules -x --pdb

requirements:
	pip install --quiet -r requirements.txt

test:
	pytest --doctest-modules 

docs:
	cd docs && make html


