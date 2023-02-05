import tensorflow as tf
import tensorflow_datasets as tfds
import os

from multihead_attention import MultiHeadAttentionLayer
from positional_encoding import PositionalEncoding

model = tf.keras.models.load_model(
    "model.h5",
    custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
    },
    compile=False,
)
path_to_zip = tf.keras.utils.get_file(
    "data.zip",
    origin="https://bci-datasets-qjp32e.s3.us-west-2.amazonaws.com/data.zip",
    extract=True,
)

path_to_brain_dataset = os.path.join(
    os.path.dirname(path_to_zip), "data"
)


path_to_data = os.path.join(path_to_brain_dataset, "data.txt")
MAX_LENGTH = 369


def load_events():
  inputs, outputs = [], []

  with open(path_to_data, "r") as file:
          lines = file.readlines()

  for line in lines:
    parts = line.replace("\n", "").split(" ###+### ")
    outputs.append(parts[0])
    inputs.append(parts[1])

  return inputs, outputs

key_inputs, accelerometer = load_events()

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    accelerometer + key_inputs, target_vocab_size=2580
)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]


# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

def evaluate(input_sequence):

    input_sequence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(input_sequence) + END_TOKEN, axis=0
    )

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[input_sequence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(input_sequence):
    prediction = evaluate(input_sequence)
    output_sequence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )
    return output_sequence