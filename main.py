import zlib
import numpy as np
import torch
import itertools

if __name__ == '__main__':
    aaa = np.arange(1000, dtype=np.int64)
    bbb = np.arange(1000, dtype=np.float64)
    ccc = np.arange(2000, 3000, dtype=np.int64)
    ddd = np.arange(2000, 3000, dtype=np.float64)

    aa = aaa.tobytes()
    bb = bbb.tobytes()
    cc = ccc.tobytes()
    dd = ddd.tobytes()
    print(len(aa), len(bb), len(cc), len(dd))
    a = zlib.compress(aa, level=9)
    b = zlib.compress(bb, level=9)
    c = zlib.compress(cc, level=9)
    d = zlib.compress(dd, level=9)
    print(len(a), len(b), len(c), len(d))
    print(len(a) - len(c), len(b) - len(d))

    a1 = np.array([1, 0, 0, 0, 0], dtype=np.int64)
    a2 = np.array([0, 1, 0, 0, 0], dtype=np.int64)
    a3 = np.array([0, 0, 1, 0, 0], dtype=np.int64)
    a4 = np.array([0, 0, 0, 1, 0], dtype=np.int64)
    a5 = np.array([0, 0, 0, 0, 1], dtype=np.int64)

    print(len(zlib.compress(a1.tobytes(), level=9)))
    print(len(zlib.compress(a2.tobytes(), level=9)))
    print(len(zlib.compress(a3.tobytes(), level=9)))
    print(len(zlib.compress(a4.tobytes(), level=9)))
    print(len(zlib.compress(a5.tobytes(), level=9)))
    print('-----')
    print(len(zlib.compress(np.arange(1, 14).astype(np.int64).tobytes(), level=9)))

    all_sq = list(itertools.product([a1, a2, a3, a4, a5], repeat=6))
    for i in range(len(all_sq)):
        all_sq[i] = np.concatenate(all_sq[i])

    dist_set = set({})
    for i in range(len(all_sq)):
        for j in range(i + 1, len(all_sq)):
            dist_set.add(np.dot(all_sq[i], all_sq[j]))
    print(len(dist_set))
    # sq1 = np.concatenate([a1, a2, a1, a2, a1, a2])
    # sq2 = np.concatenate([a2, a1, a2, a1, a2, a1])
    # sq3 = np.concatenate([a1, a1, a1, a1, a1, a1])
    # sq4 = np.concatenate([a2, a2, a2, a2, a2, a2])



    # print('-----')
    # print(len(zlib.compress(sq1.tobytes(), level=9)))
    # print(len(zlib.compress(sq2.tobytes(), level=9)))
    # print(len(zlib.compress(sq3.tobytes(), level=9)))
    # print(len(zlib.compress(sq4.tobytes(), level=9)))

    # d = np.arange(10).astype(float)
    # b = d.tobytes()
    # print(len(b))
    # c = zlib.compress(b, level=9)
    # print(len(c))