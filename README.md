# Personality analyzer
API that predicts personality (according to Big Five model of personality) of someone based on language, speech, and/or appearance.

## Getting started
* Make sure you can run the scripts in the environment specified in the `environment.yml` file. We recommend creating a virtual environment using Anaconda.
* Make sure you have downloaded `GoogleNews-vectors-negative300.bin` and save it in location `personality_analyzer/word2vec/model`.
* In one terminal, run `python personality_analyzer/text_personality_api/api.py` to start the Flask hosting and listening service.
* Open another terminal, and run `python main.py`, which will prompt you for a sentence and returns the personality scores afterwards.

## References
* [Managing conda environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* [Documenting your project using Sphinx](https://pythonhosted.org/an_example_pypi_project/sphinx.html)