import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

def preprocess_lines(lines):
    '''
    Preprocesses the given list of sentence pairs.

    args:
      lines: list of tuples, each containing (source_sentence, target_sentence).

    returns:
      Preprocessed list of sentence pairs.
    '''
    prep_lines = [
        [preprocess(i, sp_tokens=False),
         preprocess(j, sp_tokens=True)]
        for i, j in lines
    ]

    return prep_lines

def create_language_indices(prep_lines):
    '''
    Creates language indices for input and target languages.

    args:
      prep_lines: list of preprocessed sentence pairs.

    returns:
      inp_lang: LanguageIndex object for the input language.
      tgt_lang: LanguageIndex object for the target language.
    '''
    inp_lang = LanguageIndex(en for en, ma in prep_lines)
    tgt_lang = LanguageIndex(ma for en, ma in prep_lines)

    return inp_lang, tgt_lang

def detokenize(tokens, idx2word):
    text = ""
    for t in tokens:
        if 'tensorflow' in str(type(tokens)):
            text += idx2word[t.numpy()] + ' '
        else:
            text += idx2word[t] + ' '
    text = text.replace(' <pad>', '')
    text = text.replace('<start>', '')
    text = text.replace('<end>', '')
    return text.strip()

# Function to plot model metrics
def plot_metrics(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

# Function to plot the attention weights
def plot_attention_weights(attention, sentence, idx):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attention[idx], annot=True, cmap='viridis', ax=ax)
    # ax.matshow(attention[idx], cmap='viridis', )

    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))

    ax.set_xticklabels(sentence, rotation=90)
    ax.set_yticklabels(sentence)

    ax.set_xlabel('Attention for each word in input')
    ax.set_ylabel('Attention by each word in input')
    plt.show()

# Function to preprocess the input text
def preprocess_input_text(text):
    # Tokenize and convert the input text to sequences
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequence to ensure it has the same length as the training data
    padded_sequence = pad_sequences(sequences, maxlen=max_len)
    return padded_sequence


# Predict on new sentences:
def predict_seq2seq(encoder, decoder, src_tokens, tgt_vocab, num_steps):
    enc_X = tf.expand_dims(src_tokens, axis=0)
    mask = tf.expand_dims(enc_X != 0, 1)

    enc_outputs, enc_state = encoder(enc_X, training=False)
    dec_state = enc_state
    
    dec_X = tf.expand_dims(tf.constant([tgt_vocab.word2idx['<start>']]), axis=0)
    output_seq = []
    attention_weights = []
    for _ in range(num_steps):
        Y, dec_state, att_wgts = decoder(
            dec_X, enc_outputs, dec_state, mask,training=False)
        dec_X = tf.argmax(Y, axis=2)
        pred = tf.squeeze(dec_X, axis=0)
        if pred[0].numpy() == tgt_vocab.word2idx['<end>']:
            break
        output_seq.append(pred[0].numpy())
        attention_weights.append(tf.squeeze(att_wgts, 0))
        
    attention_weights = tf.squeeze(tf.stack(attention_weights, axis=0), 1)
    return detokenize(output_seq, tgt_vocab.idx2word), attention_weights


