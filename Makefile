DIR=./world-generator-service/bin/

%:
	@$(DIR)$* $(filter-out $@,$(MAKECMDGOALS))

lint: 
	$(DIR)black .

setup:
	$(DIR)pip install -r requirements.txt

notebook:
	$(DIR)jupyter notebook

test:
	$(DIR)python -m unittest discover -s tests

.PHONY: %
