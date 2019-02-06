"""
Demo script for analyzing personality based on language usage. It assumes you've run the api.py script.
"""

import requests
import json
import logging
logging.basicConfig(level=logging.INFO)


query = input('Enter sentence to find personality:')  # returns a string
# query = "I have a lot of friends and I like partying a lot! Let's have some fun!"
print("Returning personality for sentence '%s'" % query)

# dictToSend = json.dumps({
#     "data": {
#         "text": query
#     }
# })

dictToSend = json.dumps({"data": query})

# Make a POST request.
res = requests.post('http://localhost:5010/personality', data=dictToSend)
print('Response from server:', res.text)  # could change to res.json() instead
