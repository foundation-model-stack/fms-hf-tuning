# Run all
.PHONY: all
all: fmt lint test

# Run unit tests
.PHONY: test
test:
	tox -e py

# Format python code
.PHONY: fmt
fmt:
	tox -e fmt

# Run pylint to check code
.PHONY: lint
lint:
	tox -e lint
