import numpy as np
import random
import sklearn.datasets

# Dataset iterator
def inf_train_gen(DATASET, BATCH_SIZE, BIAS, VARIANCE):
    if DATASET == '9gaussians':

        dataset = []
        scale = 3
        for i in range(int(100000 / 25)):
            for x in range(-2, 3, 2):
                for y in range(-2, 3, 2):
                    point = np.random.randn(2) * VARIANCE
                    point[0] += scale * x
                    point[1] += scale * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        #dataset /= 2.828  # stdev
        while True:
            for i in range(int(len(dataset) / BATCH_SIZE)):
                yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=1
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 1.5  # stdev plus a little
            yield data

    elif DATASET == '8gaussians':

        scale = 10.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) *np.sqrt(VARIANCE)
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')

            yield dataset

    elif DATASET == '4gaussians':
        scale = 4.
        centers = [
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * np.sqrt(VARIANCE)
                center = random.choice(centers)
                if BIAS:
                    point[0] += center[0] + BIAS
                else:
                    point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            #dataset /= 1.414  # stdev
            yield dataset
