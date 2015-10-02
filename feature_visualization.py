import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import h5py
from sklearn import cluster, covariance, manifold


edge_model = covariance.GraphLassoCV()
with h5py.File("ECoG_data.h5", "r+") as f:
    X = f["sub1"]["train_data"][:]
X /= X.std(axis=0)
edge_model.fit(X)
_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()
node_position_model = manifold.LocallyLinearEmbedding(
                      n_components=2, eigen_solver='dense', n_neighbors=6)
embedding = node_position_model.fit_transform(X.T).T
plt.figure(1, facecolor='w', figsize=(20, 20))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.1)
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2,
            c=labels,
            cmap=plt.cm.hot)
start_idx, end_idx = np.where(non_zero)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(1)
ax.add_collection(lc)
plt.show()
