import tensorflow as tf


class Pretrain(object):
    def __init__(self, scope, obs_dims, cnn_layer, fc_layer, ac_dim):
        self.scope = scope
        self.obs_dims = obs_dims
        self.cnn_layer = cnn_layer
        self.fc_layer = fc_layer
        self.ac_dim = ac_dim
        self.obs, self.ac, self.dropout_prob = self.build_model()

    def build_model(self):
        print('building model %s' % self.scope)
        with tf.variable_scope(self.scope):
            obs = tf.placeholder(shape=(None,) + self.obs_dims, dtype=tf.float32, name='state')
            dropout_prob = tf.placeholder(shape=(), dtype=tf.float32, name='dropout_prob')

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
                last_out = tf.contrib.layers.max_pool3d(inputs=last_out,
                                                        kernel_size=[2, 2, 2],
                                                        stride=2,
                                                        padding='SAME',
                                                        data_format='NDHWC',
                                                        scope='maxpooling%i' % idx)
            fc_out = tf.contrib.layers.flatten(last_out, scope='flatten')
            for idx, i in enumerate(self.fc_layer):
                fc_out = tf.contrib.layers.dropout(
                    tf.contrib.layers.fully_connected(inputs=fc_out,
                                                      num_outputs=i,
                                                      activation_fn=tf.nn.relu,
                                                      scope='fc%i' % idx), keep_prob=dropout_prob)
            # the last layer
            ac = tf.contrib.layers.fully_connected(inputs=fc_out, num_outputs=self.ac_dim,
                                                   activation_fn=None, scope='last_fc')

        return obs, ac, dropout_prob

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


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
    def __init__(self, scope, obs_dims, ac_dim, cnn_layer, fc_layer, beta=1.0, tau=0.001, lr=1e-4):
        self.obs_dims = obs_dims
        self.cnn_layer = cnn_layer
        self.fc_layer = fc_layer
        self.ac_dim = ac_dim
        self.scope = scope
        self.target_scope = 'target_' + scope
        self.tau = tau
        self.lr = lr
        self.beta = beta
        self.sess = None

        # build model
        self.step_size = tf.placeholder(shape=(), dtype=tf.float32, name='step_size')
        self.dropout_prob = tf.placeholder(shape=(), dtype=tf.float32, name='dropout_prob')
        self.obs, self.ac, self.actor_vars = self.build_model(self.scope)
        self.target_obs, self.target_ac, self.target_actor_vars = self.build_model(self.target_scope)
        self.update_target_ops = get_target_updates(self.actor_vars, self.target_actor_vars, self.tau)

        # variable for optimization
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.q_gradient_input = tf.placeholder(shape=(None, self.ac_dim), dtype=tf.float32, name='q_gradient')
        self.optimizer, self.actor_loss = self.set_optimizer()

    def build_model(self, scope):
        print('building model %s' % scope)
        with tf.variable_scope(scope):
            obs = tf.placeholder(shape=(None,) + self.obs_dims, dtype=tf.float32, name='state')

            last_out = tf.identity(obs)
            for idx, i in enumerate(self.cnn_layer):
                last_out = tf.contrib.layers.conv3d(inputs=last_out,
                                                    num_outputs=i,
                                                    kernel_size=5,
                                                    activation_fn=tf.nn.elu,
                                                    stride=1,
                                                    padding='SAME',
                                                    data_format='NDHWC',
                                                    scope='3dcnn%i' % idx)
                last_out = tf.contrib.layers.max_pool3d(inputs=last_out,
                                                        kernel_size=[2, 2, 2],
                                                        stride=2,
                                                        padding='SAME',
                                                        data_format='NDHWC',
                                                        scope='maxpooling%i' % idx)
            fc_out = tf.contrib.layers.flatten(last_out, scope='flatten')
            for idx, i in enumerate(self.fc_layer):
                fc_out = tf.contrib.layers.dropout(
                    tf.contrib.layers.fully_connected(inputs=fc_out,
                                                      num_outputs=i,
                                                      activation_fn=tf.nn.elu,
                                                      scope='fc%i' % idx), keep_prob=self.dropout_prob)
            # the last layer
            ac = self.step_size * tf.contrib.layers.fully_connected(inputs=fc_out, num_outputs=self.ac_dim,
                                                                    activation_fn=None, scope='last_fc')
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        return obs, ac, scope_vars

    def set_optimizer(self):
        actor_loss = - tf.reduce_mean(self.ac * self.q_gradient_input, axis=0) \
                     + self.beta * tf.reduce_mean(tf.square(self.ac), axis=0)
        actor_grads = tf.gradients(actor_loss, self.actor_vars)
        optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(actor_grads, self.actor_vars),
                                                                    global_step=self.global_step)
        return optimizer, actor_loss

    def load_sess(self, sess):
        self.sess = sess

    def get_action(self, obs, step_size, dropout_prob):
        # get action using current policy
        return self.sess.run(self.ac, feed_dict={self.obs: obs,
                                                 self.step_size: step_size,
                                                 self.dropout_prob: dropout_prob})

    def get_target_action(self, obs, step_size, dropout_prob):
        # get action using target policy
        return self.sess.run(self.target_ac, feed_dict={self.target_obs: obs,
                                                        self.step_size: step_size,
                                                        self.dropout_prob: dropout_prob})

    def train(self, q_gradient, obs, step_size, dropout_prob):
        # optimization
        return self.sess.run([self.optimizer, self.actor_loss, self.global_step, self.ac],
                             feed_dict={self.q_gradient_input: q_gradient, self.obs: obs,
                                        self.step_size: step_size, self.dropout_prob: dropout_prob})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_target_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.target_scope)


class Critic(object):
    def __init__(self, scope, obs_dims, ac_dim, cnn_layer, fc_layer, tau=0.001, lr=1e-4):
        self.obs_dims = obs_dims
        self.fc_layer = fc_layer
        self.cnn_layer = cnn_layer
        self.ac_dim = ac_dim
        self.scope = scope
        self.target_scope = 'target_' + scope
        self.tau = tau
        self.lr = lr
        self.sess = None

        # build model
        self.dropout_prob = tf.placeholder(shape=(), dtype=tf.float32, name='dropout_prob')
        self.obs, self.ac, self.q, self.critic_vars = self.build_model(self.scope)
        self.target_obs, self.target_ac, self.target_q, self.target_critic_vars = self.build_model(self.target_scope)
        self.update_target_ops = get_target_updates(self.critic_vars, self.target_critic_vars, self.tau)

        # gradient of q wrt a
        self.q_gradient = tf.gradients(self.q, self.ac)

        # optimizer
        self.r = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='reward')
        self.gamma = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='gamma')
        self.optimizer, self.q_loss = self.set_optimizer()

    def build_model(self, scope):
        print('building model %s' % scope)
        with tf.variable_scope(scope):
            ac = tf.placeholder(shape=(None, self.ac_dim), dtype=tf.float32, name='q_ac')
            obs = tf.placeholder(shape=(None,) + self.obs_dims, dtype=tf.float32, name='q_obs')

            last_out = tf.identity(obs)
            for idx, i in enumerate(self.cnn_layer):
                last_out = tf.contrib.layers.conv3d(inputs=last_out,
                                                    num_outputs=i,
                                                    kernel_size=5,
                                                    activation_fn=tf.nn.elu,
                                                    stride=1,
                                                    padding='SAME',
                                                    data_format='NDHWC',
                                                    scope='3dcnn%i' % idx)
                last_out = tf.contrib.layers.max_pool3d(inputs=last_out,
                                                        kernel_size=[2, 2, 2],
                                                        stride=2,
                                                        padding='SAME',
                                                        data_format='NDHWC',
                                                        scope='maxpooling%i' % idx)
            fc_out = tf.contrib.layers.flatten(last_out, scope='flatten')
            for idx, i in enumerate(self.fc_layer):
                if idx == 1:
                    fc_out = tf.concat([fc_out, ac], axis=1)
                fc_out = tf.contrib.layers.dropout(
                    tf.contrib.layers.fully_connected(inputs=fc_out,
                                                      num_outputs=i,
                                                      activation_fn=tf.nn.elu,
                                                      scope='fc%i' % idx), keep_prob=self.dropout_prob)
            # last layer
            q_value = tf.contrib.layers.fully_connected(inputs=fc_out, num_outputs=1,
                                                        activation_fn=None, scope='last_fc')
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        return obs, ac, q_value, scope_vars

    def set_optimizer(self):
        weight_decay = tf.add_n([0.01 * tf.nn.l2_loss(var) for var in self.critic_vars])
        q_loss = tf.reduce_mean(tf.square(self.r + self.gamma * self.target_q - self.q), axis=0) + weight_decay
        critic_grads = tf.gradients(q_loss, self.critic_vars)
        optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(critic_grads, self.critic_vars))
        return optimizer, q_loss

    def load_sess(self, sess):
        self.sess = sess

    def get_q(self, obs, ac, dropout_prob):
        return self.sess.run(self.q, feed_dict={self.obs: obs, self.ac: ac, self.dropout_prob: dropout_prob})

    def get_target_q(self, obs, ac, dropout_prob):
        return self.sess.run(self.target_q, feed_dict={self.target_obs: obs,
                                                       self.target_ac: ac,
                                                       self.dropout_prob: dropout_prob})

    def get_q_gradient(self, obs, ac, dropout_prob):
        return self.sess.run(self.q_gradient, feed_dict={self.obs: obs, self.ac: ac, self.dropout_prob: dropout_prob})

    def train(self, obs, ac, next_obs, next_ac, r, gamma, dropout_prob):
        return self.sess.run([self.optimizer, self.q_loss],
                             feed_dict={self.obs: obs, self.ac: ac, self.r: r,
                                        self.gamma: gamma, self.target_obs: next_obs,
                                        self.target_ac: next_ac,  self.dropout_prob: dropout_prob})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_target_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.target_scope)


def in_test():
    pass


if __name__ == '__main__':
    in_test()
