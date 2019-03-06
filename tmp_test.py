# import numpy as np
# import scipy.io as sio
# from matplotlib.animation import FuncAnimation
# import utils
#
# # Initialize the kinematic chain configuration
# config = utils.bone_config()
#
# # Load some sample data
# original_xyz = sio.loadmat('C:/Users/szou2/Desktop/H3.6m/'
#                            'Data/Human/Train/train_xyz/S1_directions_1_xyz.mat')['joint_xyz']
#
# njoints = original_xyz.shape[1]
#
# # Computing the bone lengths
# skip = config.skip
# bone_skip = skip[0:-1]
#
# bone = np.zeros([njoints, 3])
# for i in range(njoints):
#     if i in bone_skip:
#         continue
#     else:
#         bone[i, 0] = round(np.linalg.norm(original_xyz[0, i, :] - original_xyz[0, i - 1, :]), 2)
#
# # Set the bone lengths
# config.bone = bone
#
# # Compute the lie parameters characterizing the pose
# lie_params = utils.inverse_kinematics(original_xyz, config)
#
# # Compute the 3D joint locations after forward kinematics for verifying that everything is fine
# fk_xyz = utils.forward_kinematics(lie_params, config)
#
# plot_3D = utils.plot_human(original_xyz, fk_xyz, config)
# anim = FuncAnimation(plot_3D.fig, plot_3D.update, frames=plot_3D.nframes, interval=50)
# # anim.to_html5_video()
# anim.save('basic_animation.html')


#
# import ray
#
# ray.init(num_cpus=15, num_gpus=3)
# samples = [train_root.remote(config) for _ in range(3)]
# ray.get(samples)

import tensorflow as tf
import numpy as np

a = tf.placeholder(shape=(None, 10), dtype=tf.float32)
b = tf.placeholder(shape=(None, 10), dtype=tf.float32)

c = tf.reduce_mean(a + b, axis=0)
g = tf.gradients(c, a)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())

    _a = np.zeros([5, 10]) + 0.1
    _b = np.zeros([5, 10]) + 0.1

    print(sess.run(c, feed_dict={a: _a, b: _b}))
    print(sess.run(g, feed_dict={a: _a, b: _b}))
