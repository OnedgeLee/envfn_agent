import tensorflow as tf
import tensorflow.contrib.slim as slim

hidden_units = [32, 32, 32, 32, 32]

def predict_mlp_model(
    inputs,
    is_training=True,
    predict_size=1,
    l2_weight=0.1,
    batch_norm_decay=0.99
):

    with slim.arg_scope(
        [slim.fully_connected],
        activation_fn=tf.nn.elu,
        normalizer_fn=slim.batch_norm,
        normalizer_params={'is_training': is_training,'decay': batch_norm_decay},
        weights_regularizer=slim.l2_regularizer(l2_weight)
        # weights_regularizer=None
    ):
        net = inputs
        i = 0
        for units in hidden_units:
            i += 1
            net = slim.fully_connected(net, units, scope="fc_" + str(i))            

        with slim.arg_scope(
            [slim.fully_connected],
            activation_fn=None,
            normalizer_fn=None
        ):
            predict = slim.fully_connected(net, predict_size, scope="fc_pred")

    return predict
