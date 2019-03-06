import numpy as np
import matplotlib.pyplot as plt


def pldist(point, start, end):
    # point to line segment distance
    dist = np.abs(np.cross(point - start, end - start)) / np.linalg.norm(end - start)
    return dist


def _rdp_rec(poly, epsilon):
    cnt = poly.shape[0]
    if cnt == 2:
        return [0, 1]

    max_dist = 0
    max_idx = -1

    for i in range(1, cnt - 1):
        dist = pldist(poly[i], poly[0], poly[-1])
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left_ids = _rdp_rec(poly[:(max_idx + 1), :], epsilon)
        right_ids = _rdp_rec(poly[max_idx:, :], epsilon)

        right_ids = np.arange(max_idx, cnt)[right_ids]
        final_ids = np.hstack((left_ids[:-1], right_ids))
    else:
        final_ids = np.array([0, cnt - 1])

    return final_ids


def rdp_rec(poly, epsilon):
    """
    recursive rdp algorithm
    poly: np.ndarray -> (N, 2)
    epsilon: maximum distance from a point to nearby line segments
    """
    poly_ids = _rdp_rec(poly, epsilon)

    return poly[poly_ids, :]


def _rdp_iter(poly, epsilon):
    cnt = poly.shape[0]
    # start, end point
    sep = []
    sep.append([0, cnt - 1])
    indices = np.ones((cnt,), dtype=bool)

    while sep:
        start, end = sep.pop()
        max_dist = 0
        max_idx = -1

        for i in range(start + 1, end):
            dist = pldist(poly[i], poly[start], poly[end])
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            sep.append([start, max_idx])
            sep.append([max_idx, end])
        else:
            for i in range(start + 1, end):
                indices[i] = False

    return np.where(indices)[0]


def rdp_iter(poly, epsilon):
    """
    iterative rdp algorithm
    poly: np.ndarray -> (N, 2)
    epsilon: maximum distance from a point to nearby line segments
    """
    poly_ids = _rdp_iter(poly, epsilon)

    return poly[poly_ids, :]


if __name__ == '__main__':
    """
    Ramer–Douglas–Peucker algorithm for line simplification
    I check my code by comparing with 'https://github.com/fhirschmann/rdp'
    """

    x = np.linspace(0, 100, 1000)
    y = x**3

    poly = np.concatenate((x[..., None], y[..., None]), axis=1)
    poly_rec = rdp_rec(poly, 5)
    poly_iter = rdp_iter(poly, 5)

    plt.figure()
    plt.plot(poly[:, 0], poly[:, 1], color='red', label='poly %d' % poly.shape[0])
    plt.plot(poly_rec[:, 0], poly_rec[:, 1], color='blue', label='poly recursively %d' % poly_rec.shape[0])
    plt.plot(poly_iter[:, 0], poly_iter[:, 1], color='yellow', label='poly iteratively %d' % poly_iter.shape[0])
    plt.legend()
    plt.show()

