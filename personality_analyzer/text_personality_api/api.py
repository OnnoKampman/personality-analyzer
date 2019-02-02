#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import sys
import os
import logging
import time
import datetime
from tensorflow.contrib import learn
import csv
from flask import Flask, url_for, request, abort, jsonify, json
from gensim.models.keyedvectors import KeyedVectors
import re

# Parameters
# ==================================================

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

app = Flask(__name__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#path to word2vec binary file
tf.flags.DEFINE_string("word2vec", dir_path + "/../word2vec/models/GoogleNews-vectors-negative300.bin", "word2vec binary file path") 

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

modelNames = [dir_path + '/model/extr/model-2300', dir_path + '/model/agr/model-6900', dir_path + '/model/cons/model-5750', dir_path + '/model/emot/model-5750', dir_path + '/model/openn/model-5175']

for i in range(len(modelNames)):
    print("Model " + str(i) + " = " + modelNames[i])

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)  # before this was FLAGS._parse_flags()

x_raw = "I have a lot of friends and I like partying a lot! Let's have some fun!"

f_word2vec = KeyedVectors.load_word2vec_format(FLAGS.word2vec, binary=True) 

vocab_path = dir_path + '/model/vocab'
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
init_vocab_len = len(vocab_processor.vocabulary_)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


bigFive = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Emotional Stability', 'Openness to Experience']

def getResult(x_test, x_vectors):

    personalityResult = {}
    counter = 0

    for checkpoint_file in modelNames:
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                scores = graph.get_operation_by_name("output/scores").outputs[0]

                scores = sess.run(scores, {input_x: x_test, dropout_keep_prob: 1.0})
                
                personalityResult[bigFive[counter]] = float(str(softmax(scores)[0][1]))
                #print(softmax(scores)[0][1])
                counter += 1

    return jsonify(personalityResult)

@app.route('/personality', methods=['POST'])
def api_personality():
    print(json.loads(request.data)["data"])
    print(type(json.loads(json.loads(request.data)["data"])))

    if not request.data or not 'data' in json.loads(request.data):
        abort(400)

    if not 'text' in json.loads(json.loads(request.data)["data"]):
        abort(400)

    x_raw = json.loads(json.loads(request.data)["data"])["text"]
    #x_raw = clean_str(json.loads(request.data)["text"])
    #x_raw = request.data
    print(x_raw)
    # Map data into vocabulary
    tokens = x_raw.split(' ')
    x_vectors = np.random.uniform(-0.25,0.25,(len(tokens), 300))
    for i in range(len(tokens)):
        if tokens[i] in f_word2vec:
            x_vectors[i] = f_word2vec[tokens[i]]
    
    #vocab_processor.vocabulary_.freeze(False)
    x_test = np.array(list(vocab_processor.transform([x_raw])))

    #vocab_processor.vocabulary_.freeze(True)
    res = getResult(x_test, x_vectors)
    print(res)
    return res;

if __name__ == '__main__':
    app.run(port=5010)
        
