import tensorflow as tf

actor_hidden_units = [200, 100]
critic_s_hidden_units = [200]
critic_a_hidden_units = []
critic_q_hidden_units = [100]


def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter


def build_actor(s, a_dim, is_training, reuse=None, getter=None):
    with tf.variable_scope('Actor', reuse=reuse, custom_getter=getter):
        net = s
        layer_num = 0
        for units in actor_hidden_units:
            layer_num += 1

            net = tf.layers.dense(
                net, units, 
                activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal(), 
                name='fc_' + str(layer_num))
            tf.summary.histogram('fc_out_' + str(layer_num), net)

            net = tf.layers.batch_normalization(
                net, 
                training=is_training,
                name='bn_' + str(layer_num))
            tf.summary.histogram('bn_out_' + str(layer_num), net)

            tf.summary.histogram(
                'fc_w_' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Actor/fc_'+str(layer_num)))[0])
            tf.summary.histogram(
                'fc_b_' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Actor/fc_'+str(layer_num)))[1])
            tf.summary.histogram(
                'bn_w' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Actor/bn_'+str(layer_num)))[0])
            tf.summary.histogram(
                'bn_b' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Actor/bn_'+str(layer_num)))[1])
        
        a = tf.layers.dense(
            net, a_dim, 
            activation=tf.nn.sigmoid, kernel_initializer=tf.initializers.he_normal(),
            name='a')

        tf.summary.histogram(
            'a_w', 
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/a')[0])
        tf.summary.histogram(
            'a_b', 
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor/a')[1])
        tf.summary.histogram('a_out', a)

    return a


def build_critic(s, a, is_training, reuse=None, getter=None):
    with tf.variable_scope('Critic', reuse=reuse, custom_getter=getter):
        layer_num = 0
        net_s = s
        for units in critic_s_hidden_units:
            layer_num += 1

            net_s = tf.layers.dense(
                net_s, units, 
                activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal(), 
                name='fc_s_' + str(layer_num))
            tf.summary.histogram('fc_s_out_' + str(layer_num), net_s)

            net_s = tf.layers.batch_normalization(
                net_s, 
                training=is_training,
                name='bn_s_' + str(layer_num))
            tf.summary.histogram('bn_s_out_' + str(layer_num), net_s)

            tf.summary.histogram(
                'fc_s_w_' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/fc_s_'+str(layer_num)))[0])
            tf.summary.histogram(
                'fc_s_b_' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/fc_s_'+str(layer_num)))[1])
            tf.summary.histogram(
                'bn_s_w' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/bn_s_'+str(layer_num)))[0])
            tf.summary.histogram(
                'bn_s_b' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/bn_s_'+str(layer_num)))[1])

        net_s = tf.layers.dense(
            net_s, critic_q_hidden_units[0], 
            activation=None, kernel_initializer=tf.initializers.he_normal(), 
            name='fc_q_s')

        tf.summary.histogram(
            'fc_q_s_w', 
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/fc_q_s')[0])
        tf.summary.histogram(
            'fc_q_s_b', 
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/fc_q_s')[1])
        tf.summary.histogram('fc_q_s_out', net_s)


        layer_num = 0
        net_a = a
        for units in critic_a_hidden_units:
            layer_num += 1

            net_a = tf.layers.dense(
                net_a, units, 
                activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal(), 
                name='fc_a_' + str(layer_num))
            tf.summary.histogram('fc_a_out_' + str(layer_num), net_a)

            net_a = tf.layers.batch_normalization(
                net_a, 
                training=is_training,
                name='bn_a_' + str(layer_num))
            tf.summary.histogram('bn_a_out_' + str(layer_num), net_a)

            tf.summary.histogram(
                'fc_a_w_' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/fc_a_'+str(layer_num)))[0])
            tf.summary.histogram(
                'fc_a_b_' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/fc_a_'+str(layer_num)))[1])
            tf.summary.histogram(
                'bn_s_w' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/bn_a_'+str(layer_num)))[0])
            tf.summary.histogram(
                'bn_s_b' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/bn_a_'+str(layer_num)))[1])

        net_a = tf.layers.dense(
            net_a, critic_q_hidden_units[0], 
            activation=None, kernel_initializer=tf.initializers.he_normal(), 
            name='fc_q_a')

        tf.summary.histogram(
            'fc_q_a_w', 
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/fc_q_a')[0])
        tf.summary.histogram(
            'fc_q_a_b', 
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/fc_q_a')[1])
        tf.summary.histogram('fc_q_a_out', net_a)

        net_q = tf.nn.relu(net_s + net_a)

        layer_num = 1
        net_q = tf.nn.relu(net_s + net_a)
        for units in critic_q_hidden_units[1:-1]:
            layer_num += 1

            net_q = tf.layers.dense(
                net_q, units, 
                activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal(), 
                name='fc_q_' + str(layer_num))
            tf.summary.histogram('fc_q_out_' + str(layer_num), net_q)

            net_q = tf.layers.batch_normalization(
                net_q, 
                training=is_training,
                name='bn_q_' + str(layer_num))
            tf.summary.histogram('bn_q_out_' + str(layer_num), net_q)

            tf.summary.histogram(
                'fc_q_w_' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/fc_q_'+str(layer_num)))[0])
            tf.summary.histogram(
                'fc_a_b_' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/fc_q_'+str(layer_num)))[1])
            tf.summary.histogram(
                'bn_s_w' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/bn_q_'+str(layer_num)))[0])
            tf.summary.histogram(
                'bn_s_b' + str(layer_num), 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ('Critic/bn_q_'+str(layer_num)))[1])
        
        q = tf.layers.dense(
            net_q, 1, 
            activation=None, kernel_initializer=tf.initializers.he_normal(),
            name='q')

        tf.summary.histogram(
            'q_w', 
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/q')[0])
        tf.summary.histogram(
            'q_b', 
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic/q')[1])
        tf.summary.histogram('q_out', q)
        
    return q