

debug:
	pytest --doctest-modules -x --gdb

test:
	pytest --doctest-modules 

docs:
	cd docs && make html


