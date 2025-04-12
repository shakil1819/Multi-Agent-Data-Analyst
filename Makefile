pytest:
	pytest --cov-report term --cov=app ./tests

pre-commit:
	pre-commit run --all-files

clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf .coverage

ruff-check:
	ruff check --fix .

ruff-format:
	ruff format .

ruff: ruff-check ruff-format

run:
	uv run uvicorn main:app --port 8000 --reload

ops: pytest pre-commit clean ruff
	@echo "\033[92mAll operations completed successfully.\033[0m"
	@echo "\033[93mPlease check the output above for any errors or warnings.\033[0m"
	@echo "\033[93mIf you see any issues, please address them before proceeding.\033[0m"
	@echo "\033[92mThank you for using this Makefile!\033[0m"
