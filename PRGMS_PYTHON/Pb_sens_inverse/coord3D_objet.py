import numpy as np
import os
import matplotlib.pyplot as plt

class Coord3D:
    def __init__(self, ME, MR):
        self.ME = ME
        self.MR = MR

    def triangulate_vectorized(self, uE, vE, uR, vR):
        n = uE.size
        A = np.zeros((n, 4, 3))
        b = np.zeros((n, 4))
        A[:, 0, :] = uE[:, None] * self.ME[2, :3] - self.ME[0, :3]
        A[:, 1, :] = vE[:, None] * self.ME[2, :3] - self.ME[1, :3]
        b[:, 0] = self.ME[0, 3] - uE * self.ME[2, 3]
        b[:, 1] = self.ME[1, 3] - vE * self.ME[2, 3]
        A[:, 2, :] = uR[:, None] * self.MR[2, :3] - self.MR[0, :3]
        A[:, 3, :] = vR[:, None] * self.MR[2, :3] - self.MR[1, :3]
        b[:, 2] = self.MR[0, 3] - uR * self.MR[2, 3]
        b[:, 3] = self.MR[1, 3] - vR * self.MR[2, 3]
        AtA = np.transpose(A, (0, 2, 1)) @ A
        Atb = np.transpose(A, (0, 2, 1)) @ b[..., None]
        return np.linalg.solve(AtA, Atb).reshape(n, 3)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath("stockage"))
    folder_stockage = os.path.join(script_dir, "Pronto-Imageur-3D", "stockage") 

    # 1. Chargement
    ME = np.loadtxt(os.path.join(folder_stockage, "ME.txt"))
    MR = np.loadtxt(os.path.join(folder_stockage, "MR.txt"))
    uR_grid = np.loadtxt(os.path.join(folder_stockage, "uRzoom.txt"))
    vR_grid = np.loadtxt(os.path.join(folder_stockage, "vRzoom.txt"))
    posi_g = np.loadtxt(os.path.join(folder_stockage, "PosiGauche.txt"))

    # 2. Reconstruction Planaire (uE = uR)
    mask = (posi_g > 0)
    vE_v = posi_g[mask] * (1280 / 32)
    uE_v = uR_grid[mask] 

    coord = Coord3D(ME, MR)
    XYZ = coord.triangulate_vectorized(uE_v, vE_v, uR_grid[mask], vR_grid[mask])

    # 3. Recentrage sur le plan (Z=0 pour le fond)
    Z_fond = np.percentile(XYZ[:, 2], 10)
    XYZ[:, 2] -= Z_fond

    # 4. AFFICHAGE DOUBLE (3D + PROFIL)
    fig = plt.figure(figsize=(16, 7))

    # --- Sous-graphe 1 : Vue 3D (Style Fig 6-10) ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], c='blue', s=1, alpha=0.7)
    
    ax1.set_title("Rendu 3D (Profils planaires)")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.set_zlim(-20, 100)
    ax1.view_init(elev=20, azim=-55)
    ax1.set_box_aspect([1, 1, 0.4])

    # --- Sous-graphe 2 : Vue de Profil (X, Z) ---
    ax2 = fig.add_subplot(122)
    # On affiche X en abscisse et Z en ordonnée
    # La couleur permet de distinguer les différentes franges
    ax2.scatter(XYZ[:, 0], XYZ[:, 2], c='blue', s=0.5, alpha=0.5)
    
    ax2.set_title("Coupe de Profil (Silhouette du bouclier)")
    ax2.set_xlabel("Xmes en mm")
    ax2.set_ylabel("Zmes en mm")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # On force les axes à avoir la même échelle pour ne pas déformer la sphère
    ax2.set_aspect('equal', adjustable='box') 
    
    # On peut limiter la vue pour zoomer sur la bosse
    ax2.set_ylim(-10, 80)

    plt.tight_layout()
    plt.show()