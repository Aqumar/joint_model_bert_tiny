# -*- coding: utf-8 -*-
"""
Created on tuesday 13-Apr-2021
@author: rishabbh-sahu
"""

import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from collections import namedtuple
from readers.reader import Reader
from collections import Counter

from text_preprocessing import vectorizer,preprocessing
from text_preprocessing.vectorizer import BERT_PREPROCESSING
from model import JOINT_TEXT_MODEL

from sklearn import metrics
from seqeval.metrics import classification_report, f1_score
from itertools import chain


print("TensorFlow Version:",tf.__version__)
print("Hub version: ",hub.__version__)
print('GPU is in use:',tf.config.list_physical_devices('GPU'))

configuration_file_path = 'config.yaml'
config = {}
config.update(Reader.read_yaml_from_file(configuration_file_path))

data_path = config['data_path']

print('read data ...')
train_text_arr, train_tags_arr, train_intents = Reader.read(data_path+'train/')
val_text_arr, val_tags_arr, val_intents = Reader.read(data_path+'valid/')
data_text_arr, data_tags_arr, data_intents = Reader.read(data_path+'test/')

train_text_arr = preprocessing.remove_next_line(train_text_arr)
train_tags_arr = preprocessing.remove_next_line(train_tags_arr)
train_intents = preprocessing.remove_next_line(train_intents)
print('train_text_arr', len(train_text_arr))

val_text_arr = preprocessing.remove_next_line(val_text_arr)
val_tags_arr = preprocessing.remove_next_line(val_tags_arr)
val_intents = preprocessing.remove_next_line(val_intents)
print('val_text_arr', len(val_text_arr))

data_text_arr = preprocessing.remove_next_line(data_text_arr)
data_tags_arr = preprocessing.remove_next_line(data_tags_arr)
data_intents = preprocessing.remove_next_line(data_intents)
print('Test data size :',len(data_text_arr))

class_dist = Counter(train_intents)
print('Intents & Distributions:',class_dist)

print('encode sequence labels ...')
sequence_label_encoder = vectorizer.label_encoder(train_intents)
train_sequence_labels = vectorizer.label_encoder_transform(train_intents,sequence_label_encoder)
val_sequence_labels = vectorizer.label_encoder_transform(val_intents,sequence_label_encoder)
intents_num = len(sequence_label_encoder.classes_)
print('Total number of sequence labels are', intents_num)

print('encode sequence tags ...')
tags_data = ['<PAD>'] + [item for sublist in [s.split() for s in train_tags_arr] for item in sublist] \
                       + [item for sublist in [s.split() for s in val_tags_arr] for item in sublist]
slot_encoder = vectorizer.label_encoder(tags_data)
slots_num = len(slot_encoder.classes_)
print('Total number of slots are :', slots_num)

# initializing the model
model = JOINT_TEXT_MODEL(slots_num=slots_num,intents_num=intents_num,model_path=config['model_path'],learning_rate=config['LEARNING_RATE'])

# initializing the model tokenizer to be used for creating sub-tokens
model_tokenizer = BERT_PREPROCESSING(model_layer=model.model_layer,max_seq_length=config['MAX_SEQ_LEN'])

print('creating input arrays for the model inputs..')
train = model_tokenizer.create_input_array(train_text_arr)
val = model_tokenizer.create_input_array(val_text_arr)

train_tags = np.array([model_tokenizer.get_tag_labels(text,tag_labels,slot_encoder) \
                       for (text,tag_labels) in zip(train_text_arr,train_tags_arr)])
val_tags = np.array([model_tokenizer.get_tag_labels(text,tag_labels,slot_encoder) \
                     for (text,tag_labels) in zip(val_text_arr,val_tags_arr)])

model.fit(train,[train_tags,train_sequence_labels],validation_data=(val,[val_tags,val_sequence_labels]),
          epochs=config['EPOCHS'],batch_size=config['BATCH_SIZE'])

# Model evaluation
#query = 'could you please play songs from james blunt'

def flatten(y):
    return list(chain.from_iterable(y))

query = 'Early morning , right ? I want to be rested for the big party .'

with open(os.path.join('/home/aqumar/bert_joint_model/intent_and_slot_classification/data/snips/train/seq.in'), encoding='utf-8') as f:
    text_arr = f.read().splitlines()
    # text_arr = f.readlines()
    text_arr = [sub.replace('\n', '') for sub in text_arr]

with open(os.path.join('/home/aqumar/bert_joint_model/intent_and_slot_classification/data/snips/train/seq.out'), encoding='utf-8') as f:
    tags_arr = f.read().splitlines()
    # text_arr = f.readlines()
    tags_arr = [sub.replace('\n', '') for sub in tags_arr]

with open(os.path.join('/home/aqumar/bert_joint_model/intent_and_slot_classification/data/snips/train/label'), encoding='utf-8') as f:
    data_intents = f.read().splitlines()
    data_intents = [sub.replace('\n', '') for sub in data_intents]


#gold_tags = [x.split() for x in tags_arr]

gold_tags = [model_tokenizer.get_prediction_tag_labels(text,tag_labels) \
                       for (text,tag_labels) in zip(text_arr,tags_arr)]

prediction_tags = []
predicted_intents = []
counter = 0
for query in text_arr:

    test_tokens, test_query = model_tokenizer.get_tokenized_query(query)
    test_inputs=model_tokenizer.create_input_array([query])
    slots,intent=model.predict(test_inputs)
    #print('Test query intent prediction:', sequence_label_encoder.inverse_transform([np.argmax(intent)]))
    intent_ = sequence_label_encoder.inverse_transform([np.argmax(intent)])
    predicted_intents.append(intent_)
    # Use the highest logit values for tag prediction
    slots=np.argmax(slots, axis=-1)

    before_pad = list(test_inputs['input_mask'][0]).count(1)
    #list_without_pad=[item for sublist in slots for item in sublist if item > 0]
    list_without_pad = slots[0][0:before_pad]
    # Removing CLS and SEP tokens from the prediction
    pred_tags=slot_encoder.inverse_transform(list_without_pad[1:-1])

    if len(gold_tags[counter]) != len(pred_tags):
        #pass

        print('test query - ', test_query)
        print('test tokens - ', test_tokens)

        print("test inputs::", test_inputs)

        print("list_without_pad::", list_without_pad)

        print('Test query intent prediction:', sequence_label_encoder.inverse_transform([np.argmax(intent)]))

        print("slots::", slots)
        print("original tag::", gold_tags[counter])
        print('Test query entities prediction :', pred_tags)

        print('token level entity predictions :', [(word, tag) for word, tag in zip(model_tokenizer.tokenizer.tokenize(query), pred_tags)])

    prediction_tags.append(list(pred_tags))

    counter = counter + 1

acc = metrics.accuracy_score(data_intents, predicted_intents)
print("Intent accuracy::", acc)

token_f1_score = metrics.f1_score(flatten(gold_tags), flatten(prediction_tags), average='micro')
print("total f1 score::", token_f1_score)


#print("gold_tags::", gold_tags)
#print("prediction_tags::", prediction_tags)
tag_f1_score = f1_score(gold_tags, prediction_tags, average='micro')
print("tag f1 score::", tag_f1_score)

report = classification_report(gold_tags, prediction_tags, digits=4)
print("classification report::", report)

print("Saving model and its config here - ", {os.path.join(config['saved_model_dir_path'],config['model_name'],config['model_version'])})
model.save(os.path.join(config['saved_model_dir_path'],config['model_name'],config['model_version']))
