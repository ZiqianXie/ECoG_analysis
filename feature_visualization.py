import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import h5py
from sklearn import covariance, manifold
from utils import num2info
from collections import Counter
import pickle
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt


def embedding(sid):
    edge_model = covariance.GraphLassoCV()
    with h5py.File("ECoG_data.h5", "r+") as f:
        X = f["sub{}/train_data".format(sid)][:]
    X /= X.std(axis=0)
    edge_model.fit(X)
    node_position_model = manifold.LocallyLinearEmbedding(
                          n_components=2, eigen_solver='dense', n_neighbors=6)
    embedding = node_position_model.fit_transform(X.T).T
    return embedding, edge_model


def vis(embedding, edge_model, idx):
    cnt = Counter(x[0] for x in map(num2info, idx))
    labels = map(lambda x: cnt[x], range(embedding.shape[1]))
    plt.figure(1, facecolor='w', figsize=(20, 20))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.1)
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2 *
                np.array((map(lambda x: x+1, labels))),
                c=labels,
                cmap=plt.cm.OrRd)
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

e1 = embedding(1)
e2 = embedding(2)
rd = pickle.load(open("omp_result", "rb"))
plt.figure(figsize=(30, 30))
plt.ioff()
vis(e1[0], e1[1], rd["r1"][1][0][0])
plt.savefig("omp_sub1_finger_1", bbox_inches='tight')
plt.figure(figsize=(30, 30))
vis(e1[0], e1[1], rd["r1"][1][1][0])
plt.savefig("omp_sub1_finger_2", bbox_inches='tight')
plt.figure(figsize=(30, 30))
vis(e1[0], e1[1], rd["r1"][1][2][0])
plt.savefig("omp_sub1_finger_4", bbox_inches='tight')
plt.figure(figsize=(30, 30))
vis(e1[0], e1[1], rd["r1"][1][3][0])
plt.savefig("omp_sub1_finger_5", bbox_inches='tight')
plt.figure(figsize=(30, 30))
vis(e2[0], e2[1], rd["r2"][1][0][0])
plt.savefig("omp_sub2_finger_1", bbox_inches='tight')
plt.figure(figsize=(30, 30))
vis(e2[0], e2[1], rd["r2"][1][1][0])
plt.savefig("omp_sub2_finger_4", bbox_inches='tight')
plt.figure(figsize=(30, 30))
vis(e2[0], e2[1], rd["r2"][1][2][0])
plt.savefig("omp_sub2_finger_5", bbox_inches='tight')
with h5py.File("selected.h5", "r") as f:
    for method in ["l1", "scad", "mcp"]:
        for finger in [1, 2, 4, 5]:
            plt.figure(figsize=(30, 30))
            vis(e1[0], e1[1], f["sub1/finger{}/{}".format(finger, method)][:] -
                1)
            plt.savefig("{}_sub1_finger_{}".format(method, finger),
                        bbox_inches='tight')
        for finger in [1, 4, 5]:
            plt.figure(figsize=(30, 30))
            vis(e2[0], e2[1], f["sub2/finger{}/{}".format(finger, method)])
            plt.savefig("{}_sub2_finger_{}".format(method, finger),
                        bbox_inches='tight')
