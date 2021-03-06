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


class Actor(object):
    def __init__(self, scope,
                 tau=0.001,
                 lr=1e-4,
                 obs_dims=(40, 40, 20),
                 cnn_layer=(8, 16, 32, 64, 128),
                 fc_layer=(128, 32)):
        # obs_width: W, H, D; change to N, D, H, W, C
        self.obs_dims = (2 * obs_dims[2] + 1, 2 * obs_dims[1] + 1, 2 * obs_dims[0] + 1)
        self.cnn_layer = cnn_layer
        self.fc_layer = fc_layer
        self.ac_dim = 6
        self.scope = scope
        self.target_scope = 'target_' + scope
        self.tau = tau
        self.lr = lr
        self.sess = None

        # build model
        self.obs, self.ac, self.actor_vars = self.build_model(self.scope)
        self.target_obs, self.target_ac, self.target_actor_vars = self.build_model(self.target_scope)
        self.update_target_ops = get_target_updates(self.actor_vars, self.target_actor_vars, self.tau)

        # variable for optimization
        self.q_gradient_input = tf.placeholder(shape=(None, self.ac_dim), dtype=tf.float32, name='q_gradient')
        self.optimizer, self.actor_loss = self.set_optimizer()

    def build_model(self, scope):
        print('building model %s' % scope)
        with tf.variable_scope(scope):
            obs = tf.placeholder(shape=(None,) + self.obs_dims + (1,), dtype=tf.float32, name='root_obs')

            last_out = tf.identity(obs)
            for idx, i in enumerate(self.cnn_layer):
                last_out = tf.contrib.layers.conv3d(inputs=last_out,
                                                    num_outputs=i,
                                                    kernel_size=3,
                                                    activation_fn=tf.nn.elu,
                                                    stride=1,
                                                    padding='SAME',
                                                    data_format='NDHWC',
                                                    scope='3dcnn%i' % idx)
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
                                                           activation_fn=tf.nn.elu,
                                                           scope='fc%i' % idx)
            # the last layer
            ac = tf.contrib.layers.fully_connected(inputs=fc_out, num_outputs=6,
                                                   activation_fn=None, scope='last_fc')
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        return obs, ac, scope_vars

    def set_optimizer(self):
        actor_loss = - tf.reduce_mean(self.ac * self.q_gradient_input, axis=0)
        actor_grads = tf.gradients(actor_loss, self.actor_vars)
        optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(actor_grads, self.actor_vars))
        return optimizer, actor_loss

    def load_sess(self, sess):
        self.sess = sess

    def get_action(self, obs):
        # get action using current policy
        return self.sess.run(self.ac, feed_dict={self.obs: obs})

    def get_target_action(self, obs):
        # get action using target policy
        return self.sess.run(self.target_ac, feed_dict={self.target_obs: obs})

    def train(self, q_gradient, obs):
        # optimization
        self.sess.run([self.optimizer, self.actor_loss], feed_dict={self.q_gradient_input: q_gradient, self.obs: obs})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_target_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.target_scope)


class Critic(object):
    def __init__(self,
                 scope,
                 tau=0.001,
                 lr=1e-4,
                 obs_dims=(40, 40, 20),
                 cnn_layer=(8, 16, 32, 64, 128),
                 fc_layer=(64, 20, 20, 10)):
        # obs_width: W, H, D; change to N, D, H, W, C
        self.obs_dims = (2 * obs_dims[2] + 1, 2 * obs_dims[1] + 1, 2 * obs_dims[0] + 1)
        self.fc_layer = fc_layer
        self.cnn_layer = cnn_layer
        self.ac_dim = 6
        self.scope = scope
        self.target_scope = 'target_' + scope
        self.tau = tau
        self.lr = lr
        self.sess = None

        # build model
        self.obs, self.ac, self.q, self.critic_vars = self.build_model(self.scope)
        self.target_obs, self.target_ac, self.target_q, self.target_critic_vars = self.build_model(self.target_scope)
        self.update_target_ops = get_target_updates(self.critic_vars, self.target_critic_vars, self.tau)

        # gradient of q wrt a
        self.q_gradient = tf.gradients(self.q, self.ac)
        self.target_q_gradient = tf.gradients(self.target_q, self.target_ac)

        # optimizer
        self.r = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='root_r')
        self.gamma = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='root_gamma')
        self.optimizer, self.q_loss = self.set_optimizer()

    def build_model(self, scope):
        print('building model %s' % scope)
        with tf.variable_scope(scope):
            ac = tf.placeholder(shape=(None, 6), dtype=tf.float32, name='root_q_ac')
            obs = tf.placeholder(shape=(None,) + self.obs_dims + (1,), dtype=tf.float32, name='root_q_obs')

            last_out = tf.identity(obs)
            for idx, i in enumerate(self.cnn_layer):
                last_out = tf.contrib.layers.conv3d(inputs=last_out,
                                                    num_outputs=i,
                                                    kernel_size=3,
                                                    activation_fn=tf.nn.elu,
                                                    stride=1,
                                                    padding='SAME',
                                                    data_format='NDHWC',
                                                    scope='3dcnn%i' % idx)
                last_out = tf.contrib.layers.max_pool3d(inputs=last_out,
                                                        kernel_size=[2, 2, 2],
                                                        stride=2,
                                                        padding='VALID',
                                                        data_format='NDHWC',
                                                        scope='maxpooling%i' % idx)
            fc_out = tf.contrib.layers.flatten(last_out, scope='flatten')
            for idx, i in enumerate(self.fc_layer):
                if idx == 2:
                    fc_out = tf.concat([fc_out, ac], axis=1)
                fc_out = tf.contrib.layers.fully_connected(inputs=fc_out,
                                                           num_outputs=i,
                                                           activation_fn=tf.nn.elu,
                                                           scope='fc%i' % idx)
            # last layer
            q_value = tf.contrib.layers.fully_connected(inputs=fc_out, num_outputs=1,
                                                        activation_fn=None, scope='last_fc')
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        return obs, ac, q_value, scope_vars

    def set_optimizer(self):
        q_loss = tf.reduce_mean(tf.square(self.r + self.gamma * self.target_q - self.q), axis=0)
        critic_grads = tf.gradients(q_loss, self.critic_vars)
        optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(critic_grads, self.critic_vars))
        return optimizer, q_loss

    def load_sess(self, sess):
        self.sess = sess

    def get_q(self, obs, ac):
        return self.sess.run(self.q, feed_dict={self.obs: obs, self.ac: ac})

    def get_target_q(self, obs, ac):
        return self.sess.run(self.target_q, feed_dict={self.target_obs: obs, self.target_ac: ac})

    def get_q_gradient(self, obs, ac):
        return self.sess.run(self.q_gradient, feed_dict={self.obs: obs, self.ac: ac})

    def get_target_q_gradient(self, obs, ac):
        return self.sess.run(self.target_q_gradient, feed_dict={self.target_obs: obs, self.target_ac: ac})

    def train(self, obs, ac, next_obs, next_ac, r, gamma):
        self.sess.run([self.optimizer, self.q_loss], feed_dict={self.obs: obs, self.ac: ac, self.r: r,
                                                                self.gamma: gamma, self.target_obs: next_obs,
                                                                self.target_ac: next_ac})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_target_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.target_scope)



def in_test():
    import numpy as np
    # actor = Actor('actor_root', tau=1)
    # with tf.Session() as sess:
    #     actor.load_sess(sess)
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(actor.update_target_ops)
    #
    #     obs = np.zeros((1,) + actor.obs_dims + (1,))
    #     obs[0, 0, 0, 0, 0] = 1
    #     q_gradient_input = np.array([[1, 1, 1, 1, 1, 1]])
    #     actor.train(q_gradient_input, obs)

    critic = Critic('critic_root', tau=1)
    with tf.Session() as sess:
        critic.load_sess(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(critic.update_target_ops)

        obs = np.zeros((2,) + critic.obs_dims + (1,))
        obs[0, 0, 0, 0, 0] = 1
        next_obs = obs + 0.1
        ac = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]])
        next_ac = ac + 0.1
        critic.train(obs, ac, next_obs, next_ac, np.array([[0.1]]), np.array([[0.9]]))

    pass


if __name__ == '__main__':
    in_test()


