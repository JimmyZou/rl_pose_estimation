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

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
plt.figure()
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])


pca = PCA(n_components=2)
pca.fit(X)

pca_score = pca.explained_variance_ratio_
V = pca.components_

rotated_X = np.dot(X, pca.components_.T)
# rotated_X = pca.fit_transform(X)
plt.subplot(122)
plt.scatter(rotated_X[:, 0], rotated_X[:, 1])
plt.show()

print(np.dot(rotated_X, pca.components_))
print(pca.components_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

