.PHONY: clean

environment: clean
	cp main_code\\environment.py figures\\environment.py
	cp main_code\\environment.py test_codes\\environment.py
	cp main_code\\environment.py trace_generation\\environment.py
	cp main_code\\environment.py main_code\\tests\\environment.py

clean:
	rm figures\\environment.py test_codes\\environment.py trace_generation\\environment.py main_code\\tests\\environment.py
