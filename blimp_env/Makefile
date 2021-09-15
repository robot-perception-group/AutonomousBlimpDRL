.PHONY: typehint
typehint: 
	mypy --ignore-missing-imports blimp_env/

.PHONY: test
test:
	pytest tests/

.PHONY: lint
lint:
	pylint blimp_env/

.PHONY: checklist
checklist: lint typehint test  

.PHONY: black
black:
	black -l 79 *.py

.PHONY: clean
clean:
	find . -type f -name "*.pyc" | xargs rm -fr
	find .-type d -name __pycache__ | xargs rm -fr