# Generative_Chatbot
A conversational agent also referred to as chatbot which is a computer program that tries to generate human like responses during a conversation. In this project, our goal is to build a closed domain generative-based chatbot. We have chosen sequence to sequence model as our baseline model and have improved it by adding an attention layer. The sequence to sequence model has two components - encoder and decoder, which perfectly matches the input and output of our corpus. The encoder encodes the question from the user into the language that can be processed by the computer and the decoder can decode it into the plain text that can be understood by human beings. While the decoder focuses on all time steps of the input, the attention layer actually pays different attention to each time step.  In this project, we use the bahdanau attention.
