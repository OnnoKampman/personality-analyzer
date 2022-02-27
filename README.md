# Personality analyzer
API that predicts personality (according to Big Five model of personality) of someone based on language, speech, and/or appearance.

Only includes language at the moment. The repository contains model training functions as well as pre-trained models.

## Working environment
* Make sure you can run the scripts in the environment specified in the `environment.yml` file.
  We recommend creating a virtual environment using Anaconda (`conda env create -f environment.yml`).
* Make sure you have downloaded `GoogleNews-vectors-negative300.bin` and save it in location `personality_analyzer/word2vec/model/`.
  You can download it by running `wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"`.

## Language API
* In one terminal, run `python text_api/run_rest_server.py` to start the Flask hosting and listening service.
* Open another terminal, and run `python text_api/main.py`, which will prompt you for a sentence and returns the personality scores afterwards.
