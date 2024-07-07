PHONY: hello env lint unittest

RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
BLUE=\033[0;34m
MAGENTA=\033[0;35m
CYAN=\033[0;36m
RESET=\033[0m

hello:
	@echo "${MAGENTA}Hello, $$(whoami)!${RESET}"
	@echo "${GREEN}Current Time:${RESET}\t\t${YELLOW}$$(date)${RESET}"
	@echo "${GREEN}Working Directory:${RESET}\t${YELLOW}$$(pwd)${RESET}"
	@echo "${GREEN}Shell:${RESET}\t\t\t${YELLOW}$$(echo $$SHELL)${RESET}"
	@echo "${GREEN}Terminal:${RESET}\t\t${YELLOW}$$(echo $$TERM)${RESET}"

env:
	@echo "To activate the Poetry environment, run:"
	@echo "source $$(poetry env info --path)/bin/activate"

lint:
	@echo "Running linter..."
	@source $$(poetry env info --path)/bin/activate && pre-commit run --all-files
	@echo "Done."

unittest:
	@echo "Running tests..."
	@source $$(poetry env info --path)/bin/activate && poetry run python -m unittest discover -s tests -v
	@echo "Done."
