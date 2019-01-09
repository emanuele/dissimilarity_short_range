import numpy as np
from dipy.tracking.distances import (bundles_distances_mam,
                                     bundles_distances_mdf)

if __name__ == '__main__':
    np.random.seed(42)
    filename_idxs = [0, 1]
    embeddings = ['DR', 'FLIP']
    ks = [5, 20, 40, 100]
    nbs_points = [20, 64]
    distance_thresholds = [20.0, 200.0]
    distance_functions = [bundles_distances_mam, bundles_distances_mdf]

    for filename_idx in filename_idxs:
        for embedding in embeddings:
            for k in ks:
                for nb_points in nbs_points:
                    for distance_threshold in distance_thresholds:
                        for distance_function in distance_functions:
                            print("EXPERIMENT BEGINS")
                            experiment(filename_idx, embedding, k, distance_function, nb_points, distance_threshold)
                            print("EXPERIMENT ENDS")
                            print("")
                            print("")
                            print("")
                            print("")
                            
                
