from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import os
import datetime
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, \
    Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import pickle
import keras.backend as K
import numpy as np
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('model_weight_file', './large_files/chatbot/model_weight_file.h5', 'model weight file from train session')

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU


# make sure we do softmax over the time axis
# expected shape is N x T x D
# note: the latest version of Keras allows you to pass in axis arg
def softmax_over_time(x):
    assert (K.ndim(x) > 2)
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e / s


# config
BATCH_SIZE = 64
EPOCHS = 2
LATENT_DIM = 256
LATENT_DIM_DECODER = 256  # idea: make it different to ensure things all fit together properly!
NUM_SAMPLES = 10000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# Where we will store the data
input_texts = []  # sentence in original language
target_texts = []  # sentence in target language
target_texts_inputs = []  # sentence in target language offset by 1

# load in the data
# download the data at: http://www.manythings.org/anki/
t = 0
for line in open('./large_files/english.txt'):
    # only keep a limited number of samples
    t += 1
    if t > NUM_SAMPLES:
        break

    # input and target are separated by tab
    if '\t' not in line:
        continue

    # split up the input and translation
    input_text, translation = line.rstrip().split('\t')

    # make the target input and output
    # recall we'll be using teacher forcing
    target_text = translation + ' <eos>'
    target_text_input = '<sos> ' + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_text_input)

print("num samples:", len(input_texts))

NUM_SAMPLES = t

# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
with open('./large_files/chatbot/tokenizer.pickle','wb') as f:
    pickle.dump(tokenizer_inputs,f)
print(tokenizer_inputs)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping for input language
word2idx_inputs = tokenizer_inputs.word_index

with open('./large_files/chatbot/inputwords.json', 'w') as f:
    json.dump(word2idx_inputs, f)
print('Found %s unique input tokens.' % len(word2idx_inputs))

# determine maximum length input sequence
max_len_input = max(len(s) for s in input_sequences)

# tokenize the outputs
# don't filter out special characters
# otherwise <sos> and <eos> won't appear
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)  # inefficient, oh well
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# get the word to index mapping for output language
word2idx_outputs = tokenizer_outputs.word_index
with open('./large_files/chatbot/outputwords.json', 'w') as f:
    json.dump(word2idx_outputs, f)
print('Found %s unique output tokens.' % len(word2idx_outputs))

# store number of output words for later
# remember to add 1 since indexing starts at 1
num_words_output = len(word2idx_outputs) + 1

# determine maximum length output sequence
max_len_target = max(len(s) for s in target_sequences)

# pad the sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print("encoder_data.shape:", encoder_inputs.shape)
print("encoder_data[0]:", encoder_inputs[0])

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print("decoder_data[0]:", decoder_inputs[0])
print("decoder_data.shape:", decoder_inputs.shape)

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('./large_files/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
    # is just a space-separated text file in the format:
    # word vec[0] vec[1] vec[2] ...
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
    if i < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector
with open('./large_files/chatbot/embedding_matrix.pickle','wb') as f:
    pickle.dump(embedding_matrix,f)
# create embedding layer
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=max_len_input,
    name='encoder_embedding'
    # trainable=True
)

# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
decoder_targets_one_hot = np.zeros(
    (
        len(input_texts),
        max_len_target,
        num_words_output
    ),
    dtype='float32'
)

# assign the values
for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1

##### build the model #####

# Set up the encoder - simple!
encoder_inputs_placeholder = Input(shape=(max_len_input,), name='encoder_input')
x = embedding_layer(encoder_inputs_placeholder)
encoder = Bidirectional(LSTM(
    LATENT_DIM,
    return_sequences=True,
    name='encoder_layer'
    # dropout=0.5 # dropout not available on gpu
))
encoder_outputs = encoder(x)

# Set up the decoder - not so simple
decoder_inputs_placeholder = Input(shape=(max_len_target,), name='decoder_input')

# this word embedding will not use pre-trained vectors
# although you could
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM, name='decoder_embedding')
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

######### Attention #########
# Attention layers need to be global because
# they will be repeated Ty times at the decoder
attn_repeat_layer = RepeatVector(max_len_input, name='attn_repeat_layer')  # repeat s
attn_concat_layer = Concatenate(axis=-1, name='attn_concat_layer')  # concat h1...ht and s
attn_dense1 = Dense(10, activation='tanh', name='attn_dense1')  # relation between h and s
attn_dense2 = Dense(1, activation=softmax_over_time, name='attn_dense2')  # attention weight
attn_dot = Dot(axes=1, name='attn_dot')  # to perform the weighted sum of alpha[t] * h[t]


def one_step_attention(h, st_1):
    # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
    # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)

    # copy s(t-1) Tx times
    # now shape = (Tx, LATENT_DIM_DECODER)
    st_1 = attn_repeat_layer(st_1)

    # Concatenate all h(t)'s with s(t-1)
    # Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)
    x = attn_concat_layer([h, st_1])

    # Neural net first layer
    x = attn_dense1(x)

    # Neural net second layer with special softmax over time
    alphas = attn_dense2(x)

    # "Dot" the alphas and the h's
    # Remember a.dot(b) = sum over a[t] * b[t]
    context = attn_dot([alphas, h])

    return context


# define the rest of the decoder (after attention)
decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state=True, name='decoder_lstm')
decoder_dense = Dense(num_words_output, activation='softmax', name='decoder_dense')

initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')
context_last_word_concat_layer = Concatenate(axis=2, name='context_last_word_concat_layer')

# Unlike previous seq2seq, we cannot get the output
# all in one step
# Instead we need to do Ty steps
# And in each of those steps, we need to consider
# all Tx h's

# s, c will be re-assigned in each iteration of the loop
s = initial_s
c = initial_c

# collect outputs in a list at first
outputs = []

t_array = range(max_len_target+1)

for t in range(max_len_target):  # Ty times
    # get the context using attention
    context = one_step_attention(encoder_outputs, s)
    # we need a different layer for each time step, teacher forcing
    selector = Lambda(lambda x: x[:, t_array[t]:t_array[t+1]])
    xt = selector(decoder_inputs_x)

    # combine
    decoder_lstm_input = context_last_word_concat_layer([context, xt])

    # pass the combined [context, last word] into the LSTM
    # along with [s, c]
    # get the new [s, c] and output
    o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

    # final dense layer to get next word prediction
    decoder_outputs = decoder_dense(o)
    outputs.append(decoder_outputs)


# 'outputs' is now a list of length Ty
# each element is of shape (batch size, output vocab size)
# therefore if we simply stack all the outputs into 1 tensor
# it would be of shape T x N x D
# we would like it to be of shape N x T x D

def stack_and_transpose(x):
    # x is a list of length T, each element is a batch_size x output_vocab_size tensor
    x = K.stack(x)  # is now T x batch_size x output_vocab_size tensor
    x = K.permute_dimensions(x, pattern=(1, 0, 2))  # is now batch_size x T x output_vocab_size
    return x


# make it a layer
stacker = Lambda(stack_and_transpose)
outputs = stacker(outputs)

# create the model
model = Model(
    inputs=[
        encoder_inputs_placeholder,
        decoder_inputs_placeholder,
        initial_s,
        initial_c,
    ],
    outputs=outputs
)

# compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# train the model
z = np.zeros((NUM_SAMPLES, LATENT_DIM_DECODER))  # initial [s, c]
r = model.fit(
    [encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0
)

train_end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("train_start_time: ", train_start_time)
print("train_end_time: ", train_end_time)

if FLAGS.model_weight_file:
    model.save_weights(FLAGS.model_weight_file)
else:
    print('please specify file to save the model weight or ignore if no need')

