import cv2
import numpy as np

if __name__ == "__main__":
    coors = [
        10433,
        12577,
        10658,
        12680,
        10720,
        12680,
        10843,
        12762,
        11273,
        13130,
        11724,
        13868,
        11826,
        14114,
        11888,
        14319,
        11893,
        14372,
        12727,
        14319,
        12707,
        15466,
        12563,
        15855,
        12441,
        16019,
        12215,
        16285,
        11847,
        16531,
        11150,
        16797,
        10720,
        16899,
        9696,
        16899,
        8958,
        16736,
        8672,
        16613,
        8180,
        16244,
        8016,
        16060,
        7668,
        15425,
        7606,
        15199,
        7606,
        14503,
        7647,
        14257,
        8016,
        13581,
        8426,
        13151,
        8713,
        12905,
        8958,
        12762,
        9532,
        12557,
        10105,
        12536,
    ]

    assert len(coors) % 2 == 0
    xs = [coors[i] for i in range(0, len(coors), 2)]
    ys = [coors[i] for i in range(1, len(coors), 2)]

    xmax = max(xs)
    ymax = max(ys)

    h = ymax + 1
    w = xmax + 1
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(coors).reshape(-1, 2)], 255)
    cv2.imwrite("ssss.png", mask)

    print(list(zip(xs, ys)))

    x1 = min(xs)
    y1 = min(ys)

    x2 = max(xs)
    y2 = max(ys)

    h = y2 - y1 + 1
    w = x2 - x1 + 1

    xs = [v - x1 for v in xs]
    ys = [v - y1 for v in ys]
    print(h, w)
    print(xs)
    print(ys)
    print(list(zip(xs, ys)))
