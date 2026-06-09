PYTHON ?= $(shell \
	for candidate in \
		"$${VIRTUAL_ENV}/bin/python" \
		".venv/bin/python" \
		"$${HOME}"/Documents/Claude/Projects/*/.venv/bin/python \
		"python3" \
		"python"; do \
		if [ -x "$$candidate" ] && "$$candidate" -c "import torch" >/dev/null 2>&1; then \
			printf "%s" "$$candidate"; \
			exit 0; \
		fi; \
	done; \
	printf "%s" "python"; \
)

.PHONY: test
test:
	PYTHONPATH=tools "$(PYTHON)" -m pytest tests -q
