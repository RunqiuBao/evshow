all:
	@echo '## Make commands ##'
	@echo
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

lint:
	ruff check ./

setup:
	python3 -m pip install .

setup-advanced:
	python3 -m pip install .[advanced]

test:
	python3 -m pip install .[test]
	python3 -m pytest -s
