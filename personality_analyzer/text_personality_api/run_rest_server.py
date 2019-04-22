#! /usr/bin/env python

import sys
import os
import re
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn  # TODO: deprecated, should change to tf.data or tensorflow/transform
from flask import Flask, url_for, request, abort, jsonify, json, render_template
from gensim.models.keyedvectors import KeyedVectors
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))
print("\nDirectory PATH:")
print(dir_path)

app = Flask(__name__)

# Path to word2vec binary file.
# TODO: check if word2vec file is present, otherwise ask if user wants to download it
tf.flags.DEFINE_string("word2vec",
                       dir_path + "/../../models/word2vec/GoogleNews-vectors-negative300.bin",
                       "word2vec binary file path")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model configs.
big_five = ['Extraversion',
            'Agreeableness',
            'Conscientiousness',
            'Emotional Stability',
            'Openness to Experience']
model_names = [dir_path + '/../../models/text_model/extr/model-2300',
               dir_path + '/../../models/text_model/agr/model-6900',
               dir_path + '/../../models/text_model/cons/model-5750',
               dir_path + '/../../models/text_model/emot/model-5750',
               dir_path + '/../../models/text_model/openn/model-5175']

for i in range(len(model_names)):
    print("Model " + str(i) + " = " + model_names[i])

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)  # before this was FLAGS._parse_flags()

f_word2vec = KeyedVectors.load_word2vec_format(FLAGS.word2vec, binary=True) 

vocab_path = dir_path + '/../../models/text_model/vocab'
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
init_vocab_len = len(vocab_processor.vocabulary_)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    TODO: can probably be replaced with native TF function
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def clean_str(string):
    """
    Tokenization/string cleaning for all data sets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :return str; cleaned string.
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


def predict_personality(x_test):
    """
    Prediction routine.
    :param x_test: array of sentence in form of word embeddings.
    :return:
    """
    personality_result = {}
    counter = 0

    for checkpoint_file in model_names:
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
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
                scores = sess.run(scores, feed_dict={input_x: x_test,
                                                     dropout_keep_prob: 1.0})
                personality_result[big_five[counter]] = float(str(softmax(scores)[0][1]))

                # Move to next personality trait.
                counter += 1

    return jsonify(personality_result)


# @app.route('/tests/endpoint', methods=['POST'])
# def my_test_endpoint():
#     input_json = request.get_json(force=True)
#     # force=True, above, is necessary if another developer
#     # forgot to set the MIME type to 'application/json'

#     print 'data from client:', input_json
#     dictToReturn = {'answer':42}
#     return jsonify(dictToReturn)


@app.route('/personality', methods=['POST'])
def api_personality():
    """
    Requires a request of data format:
    
    dict = json.dumps({"data": query})
    res = requests.post('http://localhost:5010/personality', data=dict)
    
    where request is the original string.
    """
    print("\n> Sentence to query:")
    print(json.loads(request.data)["data"])
    
    # print("\n> Type of query:")
    # print(type(json.loads(json.loads(request.data)["data"])))

    if not request.data or 'data' not in json.loads(request.data):
        abort(400)

#    if not 'text' in json.loads(json.loads(request.data)["data"]):
#        abort(400)

    x_raw = json.loads(request.data)["data"]  # string
    # x_raw = json.loads(json.loads(request.data)["data"])["text"]  # previous one
    # x_raw = clean_str(json.loads(request.data)["text"])
    # x_raw = request.data
    print(x_raw)
    
    # Map data into vocabulary.
    tokens = x_raw.split(' ')
    x_vectors = np.random.uniform(-0.25, 0.25, (len(tokens), 300))
    for i in range(len(tokens)):
        if tokens[i] in f_word2vec:
            x_vectors[i] = f_word2vec[tokens[i]]
    
    # vocab_processor.vocabulary_.freeze(False)
    x_test = np.array(list(vocab_processor.transform([x_raw])))

    # vocab_processor.vocabulary_.freeze(True)
    res = predict_personality(x_test)
    print(res)
    
    return res


if __name__ == '__main__':

    app.run(port=5010)  # can set debug=False if needed
