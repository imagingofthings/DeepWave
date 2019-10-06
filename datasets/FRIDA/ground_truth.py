"""
Author: Robin SCHEIBLER [LCAV]
Original file name: edm_to_positions.py

Modified by Sepand KASHANI to export microphone positions and ground truth.
    dev_xyz : :py:class:`~numpy.ndarray`
        (3, N_antenna) microphone coordinates
    src_label : :py:class:`~numpy.ndarray`
        (N_src,) speaker names
    src_xyz : dict(str, :py:class:`~numpy.ndarray`)
        'marker': (3, N_src) marker coordinates
        'twitter': (3, N_src) twitter centroids
        'woofer': (3, N_src) woofer centroids

Note
----
The coordinate system is such that the center of the pyramic lies at the origin.
"""

import json

import numpy as np

import marker
import deepwave.tools.instrument as instrument


# Open the experimental protocol
with open('./FRIDA/recordings/20160908/protocol.json') as fd:
    protocol = json.load(fd)

# Get the labels and distances (flattened upper triangular of distance matrix, row-wise)
labels = protocol['calibration']['labels']
flat_distances = protocol['calibration']['distances']
m = len(labels)

# fill in the EDM
EDM = np.zeros((m, m))
flat_counter = 0
for i in range(0, m - 1):
    for j in range(i + 1, m):
        EDM[i, j] = flat_distances[flat_counter] ** 2
        EDM[j, i] = EDM[i, j]
        flat_counter += 1

# Create the marker objects
markers = marker.MarkerSet(EDM=EDM, labels=labels)

# Here we know all speakers should be in a plane
markers.flatten(labels[:-2])

# Let the pyramic ref point be the center
markers.center('pyramic')

# And align x-axis onto speaker 7
markers.align(labels[0], 'z')

# The speakers have some rotations around z-axis
# i.e. the baffles point to different directions
rotations = protocol['calibration']['rotation']


def rotz(v, deg):
    """
    rotation around z axis by some degrees
    """
    c1, s1 = np.cos(deg / 180. * np.pi), np.sin(deg / 180. * np.pi)
    rot = np.array([[c1, -s1, 0.], [s1, c1, 0.], [0., 0., 1.]])
    return np.dot(rot, v)


# Now there a few correction vectors to apply between the measurement points
# and the center of the baffles
vec_twit = np.array([-0.01, 0., -0.05])  # the correction vector
corr_twitter = {}
for lbl, rot in zip(labels[:-2], rotations[:-2]):
    corr_twitter[lbl] = rotz(vec_twit, rot[2])

vec_woof = np.array([-0.02, 0., -0.155])
corr_woofer = {}
for lbl, rot in zip(labels[:-2], rotations[:-2]):
    corr_woofer[lbl] = rotz(vec_woof, rot[2])

# Now make two sets of markers for twitters and woofers
twitters = markers.copy()
woofers = markers.copy()

# Apply the correction vectors
twitters.correct(corr_twitter)
woofers.correct(corr_woofer)

### Modifications by Sepand KASHANI
src_label = markers.labels[:-2]  # Drop 'pyramic', 'compactsix'
src_xyz = np.stack([markers.X, twitters.X, woofers.X], axis=0)[:, :, :-2]  # (3, 3, N_src)
dev_xyz = instrument.pyramic_geometry()  # Already in right coordinate system w.r.t. measurements.
# Center coordinate system at Pyramic CoG.
dev_centroid = np.mean(dev_xyz, axis=1)
dev_xyz -= dev_centroid.reshape(3, 1)
src_xyz -= dev_centroid.reshape(1, 3, 1)
src_xyz = dict(zip(['marker', 'twitter', 'woofer'], src_xyz))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d

    # Plot all the markers in the same figure to check all the locations are correct
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')

    twitters.plot(axes=axes, c='b', marker='s')
    woofers.plot(axes=axes, c='r', marker='<')
    markers.plot(axes=axes, c='k', marker='.')

    print(f'DoA of Speaker 5 to FPGA: {np.rad2deg(twitters.doa("pyramic", "5"))} degrees')

    plt.show()
