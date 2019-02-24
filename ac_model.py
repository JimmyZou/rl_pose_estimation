import tensorflow as tf


def get_target_updates(_vars, target_vars, tau):
    print('--------------------------------------------------------------------------')
    print('setting up target updates ...')
    soft_updates = []
    assert len(_vars) == len(target_vars)
    for _var, target_var in zip(_vars, target_vars):
        print('  {} <- {} ({})'.format(target_var.name, _var.name, _var.shape))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * _var))
    print('--------------------------------------------------------------------------')
    return tf.group(*soft_updates)


class ActorRoot(object):
    def __init__(self, tau=0.01,
                 obs_dims=(60, 60, 30),
                 scope='root_actor',
                 cnn_layer=(8, 16, 32, 64, 32, 16, 8, 4),
                 fc_layer=(200, 50, 6)):
        # obs_width: W, H, D; change to N, D, H, W, C
        self.obs_dims = (2 * obs_dims[2] + 1, 2 * obs_dims[1] + 1, 2 * obs_dims[0] + 1)
        self.cnn_layer = cnn_layer
        self.fc_layer = fc_layer
        self.ac_dim = fc_layer[-1]
        self.scope = scope
        self.target_scope = 'target_' + scope
        self.tau = tau
        self.sess = None

        # build model
        self.obs, self.ac, self.actor_vars = self.build_model(self.scope)
        self.target_obs, self.target_ac, self.target_actor_vars = self.build_model(self.target_scope)
        self.update_target_ops = get_target_updates(self.actor_vars, self.target_actor_vars, self.tau)

        # variable for optimization
        self.q_gradient_input = tf.placeholder(shape=(None, self.ac_dim), dtype=tf.float32, name='q_gradient')
        self.state_input = tf.placeholder(shape=(None,) + self.obs_dims + (1,), dtype=tf.float32, name='state')
        self.optimizer = self.set_optimizer()

    def build_model(self, scope):
        print('building model %s' % scope)
        with tf.variable_scope(scope):
            obs = tf.placeholder(shape=(None,) + self.obs_dims + (1,), dtype=tf.float32, name='root_obs')

            # tf.contrib.layers.conv3d
            last_out = tf.identity(obs)
            for idx, i in enumerate(self.cnn_layer):
                last_out = tf.contrib.layers.conv3d(inputs=last_out,
                                                    num_outputs=i,
                                                    kernel_size=5,
                                                    activation_fn=tf.nn.relu,
                                                    stride=1,
                                                    padding='SAME',
                                                    data_format='NDHWC',
                                                    scope='3dcnn%i' % idx)
                if idx % 2 == 1:
                    last_out = tf.contrib.layers.max_pool3d(inputs=last_out,
                                                            kernel_size=[2, 2, 2],
                                                            stride=2,
                                                            padding='VALID',
                                                            data_format='NDHWC',
                                                            scope='maxpooling%i' % idx)
            fc_out = tf.contrib.layers.flatten(last_out, scope='flatten')
            for idx, i in enumerate(self.fc_layer):
                fc_out = tf.contrib.layers.fully_connected(inputs=fc_out,
                                                           num_outputs=i,
                                                           activation_fn=tf.nn.relu,
                                                           scope='fc%i' % idx)
            ac = tf.identity(fc_out)
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        return obs, ac, scope_vars

    def load_sess(self, sess):
        self.sess = sess

    def get_action(self, obs):
        return self.sess.run(self.ac, feed_dict={self.obs: obs})

    def get_target_action(self, obs):
        return self.sess.run(self.target_ac, feed_dict={self.target_obs: obs})

    def set_optimizer(self, lr):
        actor_grads = tf.gradients(self.ac, self.actor_vars, -self.q_gradient_input)
        optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(actor_grads, self.actor_vars))
        return optimizer

    def train(self, sess, q_gradient, state):
        sess.run(self.optimizer, feed_dict={self.q_gradient_input: q_gradient, self.state_input: state})


class CriticRoot(object):
    def __init__(self, tau=0.01,
                 obs_dims=(60, 60, 30),
                 scope='root_critic',
                 cnn_layer=(8, 16, 32, 64, 32, 16, 8, 4),
                 fc_layer=(200, 50, 1)):
        # obs_width: W, H, D; change to N, D, H, W, C
        self.obs_dims = (2 * obs_dims[2] + 1, 2 * obs_dims[1] + 1, 2 * obs_dims[0] + 1)
        self.cnn_layer = cnn_layer
        self.fc_layer = fc_layer
        self.ac_dim = fc_layer[-1]
        self.scope = scope
        self.target_scope = 'target_' + scope
        self.tau = tau
        self.sess = None

    def build_model(self, scope):
        print('building model %s' % scope)
        with tf.variable_scope(scope):
            obs = tf.placeholder(shape=(None,) + self.obs_dims + (1,), dtype=tf.float32, name='root_obs')

            # tf.contrib.layers.conv3d
            last_out = tf.identity(obs)
            for idx, i in enumerate(self.cnn_layer):
                last_out = tf.contrib.layers.conv3d(inputs=last_out,
                                                    num_outputs=i,
                                                    kernel_size=5,
                                                    activation_fn=tf.nn.relu,
                                                    stride=1,
                                                    padding='SAME',
                                                    data_format='NDHWC',
                                                    scope='3dcnn%i' % idx)
                if idx % 2 == 1:
                    last_out = tf.contrib.layers.max_pool3d(inputs=last_out,
                                                            kernel_size=[2, 2, 2],
                                                            stride=2,
                                                            padding='VALID',
                                                            data_format='NDHWC',
                                                            scope='maxpooling%i' % idx)
            fc_out = tf.contrib.layers.flatten(last_out, scope='flatten')
            for idx, i in enumerate(self.fc_layer):
                fc_out = tf.contrib.layers.fully_connected(inputs=fc_out,
                                                           num_outputs=i,
                                                           activation_fn=tf.nn.relu,
                                                           scope='fc%i' % idx)
            ac = tf.identity(fc_out)
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        return obs, ac, scope_vars




def in_test():
    import numpy as np
    actor = ActorRoot(tau=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(actor.update_target_ops)

        # obs = np.zeros((1,) + actor.obs_dims + (1,))
        # obs[0, 0, 0, 0, 0] = 1
        # tmp = actor.get_action(sess, obs)
        # tmp1 = actor.get_target_action(sess, obs)
        # print(tmp, tmp1)


if __name__ == '__main__':
    in_test()


