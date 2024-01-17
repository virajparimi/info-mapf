import numpy as np

def locations_within_distance(locations, target_locations, k):
    locations_array = np.array(locations)
    target_locations_array = np.array(target_locations)

    print(locations_array[:,  np.newaxis, :])
    print(locations_array[:,  np.newaxis, :].shape)
    print(target_locations_array.shape)
    print(target_locations_array)

    distances = np.linalg.norm(locations_array[:, np.newaxis, :] - target_locations_array, axis=2)

    print(distances)

    # Find indices where the distance is less than or equal to k
    indices = np.any(distances <= k, axis=1)

    print(indices)

    result_set = set(tuple(locations[i]) for i in range(len(locations)) if indices[i])

    return result_set

# Example usage:
locations = [(1, 2), (3, 4), (5, 6)]
target_locations = [(3, 4), (8, 8)]
k = 2  # Euclidean distance threshold

result_set = locations_within_distance(locations, target_locations, k)
print(result_set)

