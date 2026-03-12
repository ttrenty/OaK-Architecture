# Default goal if you just run `make`
.DEFAULT_GOAL := help

# Catch-all rule: any target becomes `pixi run <target>`
%:
	exec pixi run $@

# Optional: allow passing extra args
# Example:
# make train ARGS="--epochs 10"
ARGS ?=

# Example explicit help target
help:
	@echo "Usage:"
	@echo "  make <task>           -> pixi run <task>"
	@echo "  make <task> ARGS=...  -> pixi run <task> -- ..."
	exec pixi run