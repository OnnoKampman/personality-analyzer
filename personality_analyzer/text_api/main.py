import json
import logging
import requests

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":

    """
    Demo script for analyzing personality based on language usage. 
    It assumes you've run the `run_rest_server.py` script.
    """

    query = input('Enter sentence to find personality:')  # returns a string
    # query = "I have a lot of friends and I like partying a lot! Let's have some fun!"
    print(f"Returning personality for sentence '{query:s}'.")

    # dictToSend = json.dumps({
    #     "data": {
    #         "text": query
    #     }
    # })
    dictToSend = json.dumps({"data": query})

    # Make a POST request.
    res = requests.post('http://localhost:5010/personality', data=dictToSend)
    print('Response from server:', res.text)  # could change to res.json() instead
