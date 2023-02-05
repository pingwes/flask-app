import tensorflow as tf


model = tf.keras.models.load_model(
    "model.h5",
    custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
    },
    compile=False,
)

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