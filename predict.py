from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, \
    Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.sequence import pad_sequences
import json
import pickle
import keras.backend as K
import numpy as np

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM

from absl import flags

FLAGS = flags.FLAGS
import sys
FLAGS(sys.argv)
flags.DEFINE_string('model_weight_file', './large_files/chatbot/mychatbot_weights.h5', 'load model weight file')
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


with open('./large_files/chatbot/tokenizer.pickle', 'rb') as f:
    tokenizer_inputs = pickle.load(f)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping for input language

with open('./large_files/chatbot/inputwords.json', 'r') as f:
    word2idx_inputs = json.load(f)

# print('Found %s unique input tokens.' % len(word2idx_inputs))

# determine maximum length input sequence
max_len_input = 30


with open('./large_files/chatbot/outputwords.json', 'r') as f:
    word2idx_outputs = json.load(f)
# print('Found %s unique output tokens.' % len(word2idx_outputs))

# store number of output words for later
# remember to add 1 since indexing starts at 1
num_words_output = len(word2idx_outputs) + 1

# determine maximum length output sequence
max_len_target = 59

# pad the sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)

with open('./large_files/chatbot/embedding_matrix.pickle','rb') as f:
    embedding_matrix = pickle.load(f)
# create embedding layer
embedding_layer = Embedding(
    957,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=max_len_input,
    name='encoder_embedding'
    # trainable=True
)



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


# The encoder will be stand-alone
# From this we will get our initial decoder hidden state
# i.e. h(1), ..., h(Tx)
encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)
if FLAGS.model_weight_file:
    encoder_model.load_weights(FLAGS.model_weight_file,by_name=True)
else:
    print('no model weights file found, pleas assign one at the beginning')
# next we define a T=1 decoder model
encoder_outputs_as_input = Input(shape=(max_len_input, LATENT_DIM * 2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# no need to loop over attention steps this time because there is only one step
context = one_step_attention(encoder_outputs_as_input, initial_s)

# combine context with last word
decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])

# lstm and final dense
o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)




# create the model object
decoder_model = Model(
    inputs=[
        decoder_inputs_single,
        encoder_outputs_as_input,
        initial_s,
        initial_c
    ],
    outputs=[decoder_outputs, s, c]
)
if FLAGS.model_weight_file:
    decoder_model.load_weights(FLAGS.model_weight_file, by_name=True)
else:
    print('no model weights file found, pleas assign one at the beginning')

# map indexes back into real words
# so we can view the results
idx2word_eng = {v: k for k, v in word2idx_inputs.items()}
idx2word_trans = {v: k for k, v in word2idx_outputs.items()}


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    enc_out = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first character of target sequence with the start character.
    # NOTE: tokenizer lower-cases all words
    target_seq[0, 0] = word2idx_outputs['<sos>']

    # if we get this we break
    eos = word2idx_outputs['<eos>']

    # [s, c] will be updated in each loop iteration
    s = np.zeros((1, LATENT_DIM_DECODER))
    c = np.zeros((1, LATENT_DIM_DECODER))

    # Create the translation
    output_sentence = []
    for _ in range(max_len_target):
        o, s, c = decoder_model.predict([target_seq, enc_out, s, c])

        # Get next word
        idx = np.argmax(o.flatten())

        # End sentence of EOS
        if eos == idx:
            break

        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)

        # Update the decoder input
        # which is just the word just generated
        target_seq[0, 0] = idx

    return ' '.join(output_sentence)

print("the topics include AI, botprofile, computer, emotion, food, gossip, greetings,"
      "health, history, humor, literature, money, movies, politics,psychology,science,sports,trivia")

import webbrowser
while True:

    print('you say:')
    ans = input()
    if 'weather' in ans:
        webbrowser.open(
            "https://www.google.com/search?q=what+is+the+weather+today&rlz=1C5CHFA_enUS814US814&oq=what+is+the+weather+today&aqs=chrome..69i57j0l5.15932j0j7&sourceid=chrome&ie=UTF-8")
    else:
        input_sequences = tokenizer_inputs.texts_to_sequences([ans])
        encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)

        input_seq = encoder_inputs[0:1]
        translation = decode_sequence(input_seq)
        print('chat say: ', translation)

    if ans == 'quit':
        print('chat say: bye')
        break



