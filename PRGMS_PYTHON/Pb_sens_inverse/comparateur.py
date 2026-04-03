import numpy as np

class Comparateur:
    def __init__(self, seuil):
        self.seuil = seuil

    def compute_bits(self, images):
        # images : liste de N images (H x W)
        bits = [(img >= self.seuil).astype(int) for img in images]
        return np.stack(bits, axis=0)  # (N, H, W)

    def compute_code(self, images):
        bits = self.compute_bits(images)
        N = bits.shape[0]
        weights = 2 ** np.arange(N-1, -1, -1)[:, None, None]
        C = np.sum(bits * weights, axis=0)
        return C

if __name__ == "__main__":
    