"""Experiment to quantify the correlation between the streamline
distance (MAM or MDF) against the Euclidean distance on the
corresponding dissimilarity representation embedding of the
streamlines.
"""

import numpy as np
import nibabel as nib
from euclidean_embeddings import dissimilarity
from functools import partial
from dipy.tracking.distances import (bundles_distances_mdf,
                                     bundles_distances_mam)
from dipy.tracking.streamline import set_number_of_points
from euclidean_embeddings.subsampling import compute_subset
import os


def experiment(filename_idx, embedding, k, distance_function, nb_points, distance_threshold):
    n = 300  # number of streamlines to query for neighbors
    max_neighbors = 200
    max_streamlines = 100000
    savefig = True
    extension_format = '.jpg'
    results_dir = 'results_%d/' % filename_idx
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    filenames = ['sub-100206_var-FNAL_tract.trk', 'sub-500222_var-EPR_tract.tck']
    filename = filenames[filename_idx]
    print("Loading %s" % filename)
    streamlines2 = nib.streamlines.load(filename).streamlines
    print("Subsampling %s at random from the whole tractogram, to reduce computations")
    streamlines = streamlines2[np.random.permutation(len(streamlines2))[:max_streamlines]]
    if distance_function == bundles_distances_mdf or embedding == 'FLAT' or embedding == 'FLATFLIP':
        print("Resampling streamlines to %s points because of MDF or FLAT embedding" % nb_points)
        streamlines = np.array(set_number_of_points(streamlines, nb_points=nb_points))
        distance_name = 'MDF%d' % nb_points
    elif distance_function == bundles_distances_mam:
        distance_name = 'MAM'
        streamlines = np.array(streamlines, dtype=np.object)
    else:
        raise NotImplementedError

    # landmark_policy = 'random'
    landmark_policy = 'sff'
    if embedding == 'DR':
        print("Computing %s prototypes with %s policy" % (k, landmark_policy))
        prototype_idx = compute_subset(dataset=streamlines,
                                       distance=distance_function,
                                       num_landmarks=k,
                                       landmark_policy=landmark_policy)
        embedding_name = embedding + '%03d' % k
    elif embedding == 'FLAT' or embedding == 'FLATFLIP':
        embedding_name = embedding
        # assert(distance_function == bundles_distances_mdf)
    else:
        raise NotImplementedError

    original_distance = []
    euclidean_distance = []
    streamline1_idx = []
    streamline2_idx = []
    print("Randomly subsampling %s streamlines for nearest-neighbors queries" % n)
    s1_idx = np.random.permutation(len(streamlines))[:n]
    print("Computing %s on streamlines vs Euclidean distance on %s" % (distance_name, embedding_name))
    for i, idx in enumerate(s1_idx):
        print(i)
        s1 = streamlines[idx]
        distances = distance_function([s1], streamlines)[0]
        tmp = np.where(distances < distance_threshold)[0]
        tmp = tmp[tmp != 0.0]  # remove s1_idx from the result
        if len(tmp) > max_neighbors:
            print("Trimming %s neighbors to %s" % (len(tmp), max_neighbors))
            tmp = np.random.permutation(tmp)[:max_neighbors]

        streamline1_idx.append([idx] * len(tmp))
        streamline2_idx.append(tmp)
        original_distance.append(distances[tmp])
        if embedding == 'DR':
            v_s1 = distance_function([s1], streamlines[prototype_idx])
            v_neighbors = distance_function(streamlines[tmp],
                                            streamlines[prototype_idx])
        elif embedding == 'FLAT' or embedding == 'FLATFLIP':
            v_s1 = s1.flatten()
            v_neighbors = streamlines[tmp].reshape(tmp.shape[0], -1)
        else:
            raise NotImplementedError

        if embedding == 'FLATFLIP':
            direct_distances = np.linalg.norm(v_s1 - v_neighbors, axis=1)
            flipped_distances = np.linalg.norm(v_s1.reshape(-1, 3)[::-1].flatten() - v_neighbors, axis=1)
            euclidean_distance.append(np.minimum(direct_distances, flipped_distances))
        else:
            euclidean_distance.append(np.linalg.norm(v_s1 - v_neighbors, axis=1))

    streamline1_idx = np.concatenate(streamline1_idx)
    streamline2_idx = np.concatenate(streamline2_idx)
    original_distance = np.concatenate(original_distance)
    euclidean_distance = np.concatenate(euclidean_distance)
    global_correlation = np.corrcoef(original_distance, euclidean_distance)[0, 1]
    print("Global correlation: %s" % global_correlation)

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    plt.plot(original_distance, euclidean_distance, 'o')
    plt.xlabel('$'+distance_name+'$')
    plt.ylabel('Euclidean distance')
    plt.title(r'$%s$ vs Euclid(%s): $\rho$=%f' % (distance_name,
                                                  embedding_name,
                                                  global_correlation))
    filename_fig = results_dir + '%s_vs_%s_%d_%d' % (distance_name,
                                                     embedding_name,
                                                     original_distance.min(),
                                                     distance_threshold)
    if savefig:
        tmp = filename_fig + extension_format
        print('Saving figure to %s' % tmp)
        plt.savefig(tmp)
    
    print("Local correlation:")
    n_steps = 10
    distance_threshold_min = np.linspace(0, original_distance.max(), n_steps)
    correlations = np.zeros(n_steps - 1)
    for i, (dtmin, dtmax) in enumerate(zip(distance_threshold_min[:-1], distance_threshold_min[1:])):
        tmp = np.logical_and(original_distance > dtmin, original_distance <= dtmax)
        # tmp = original_distance <= dtmax
        od = original_distance[tmp]
        ed = euclidean_distance[tmp]
        correlations[i] = np.corrcoef(od, ed)[0, 1]
        print("%s) %s - %s : %s, corr=%s" % (i, dtmin, dtmax, tmp.sum(), correlations[i]))

    plt.figure()
    # plt.hist(correlations, bins=distance_threshold_min)
    plt.bar(distance_threshold_min[:-1], correlations, width=np.diff(distance_threshold_min).mean())
    plt.xlabel(distance_name)
    plt.ylabel('correlation')
    plt.title(r'$\rho$($%s$, Euclid(%s)) in different intervals' % (distance_name,
                                                                    embedding_name))
    plt.xlim([distance_threshold_min.min(), distance_threshold_min.max()])
    plt.ylim([min(0, correlations.min()), 1.0])
    plt.plot([distance_threshold_min.min(),
              distance_threshold_min.max()], [global_correlation,
                                              global_correlation], 'r-',
             label=r'global $\rho$')
    plt.plot([distance_threshold_min.min(),
              distance_threshold_min.max()], [correlations.mean(),
                                              correlations.mean()], 'g-',
             label=r'avg $\rho$')
    plt.legend()
    if savefig:
        tmp = filename_fig + '_correlations' + extension_format
        print('Saving figure to %s' % tmp)
        plt.savefig(tmp)


if __name__ == '__main__':
    np.random.seed(42)
    filename_idx = 0
    embedding = 'DR'  # 'FLATFLIP'  # 'FLIP'
    k = 40
    distance_function = bundles_distances_mdf
    # distance_function = bundles_distances_mam
    nb_points = 20
    distance_threshold = 200.0
    experiment(filename_idx, embedding, k, distance_function, nb_points, distance_threshold)
