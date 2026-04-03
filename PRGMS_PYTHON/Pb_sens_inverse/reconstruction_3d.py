import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reconstructor_3d():
    # ==========================================
    # 1. Gestion des chemins (Votre méthode)
    # ==========================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # On remonte de deux crans pour atteindre la racine du projet
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    folder_stockage = os.path.join(parent_dir, "stockage")
    folder_img = os.path.join(parent_dir, "img")

    # ==========================================
    # 2. Chargement des données
    # ==========================================
    try:
        ME = np.loadtxt(os.path.join(folder_stockage, "ME.txt"))
        MR = np.loadtxt(os.path.join(folder_stockage, "MR.txt"))
        
        # Chargement des grilles de coordonnées
        uR_grid = np.loadtxt(os.path.join(folder_stockage, "uRzoom.txt"))
        vR_grid = np.loadtxt(os.path.join(folder_stockage, "vRzoom.txt"))
        
        # Chargement des résultats de la détection de franges
        PosiGlobal = np.loadtxt(os.path.join(folder_stockage, "Posiglobal.txt"))
        N = int(np.loadtxt(os.path.join(folder_stockage, "N.txt")))
        
        # Taille du projecteur (nécessaire pour vE)
        # On peut la déduire de vE_grid si disponible, sinon valeur standard
        vE_grid = np.loadtxt(os.path.join(folder_stockage, "vE.txt"))
        NbHE = vE_grid.shape[1] 
        
    except FileNotFoundError as e:
        print(f"Erreur : Un fichier est manquant dans {folder_stockage}\n{e}")
        return

    # ==========================================
    # 3. Logique de Reconstruction
    # ==========================================
    points_3d = []
    rows, cols = PosiGlobal.shape

    # On parcourt chaque pixel du récepteur
    for i in range(rows):
        for j in range(cols):
            C = PosiGlobal[i, j]
            
            if C > 0:  # Si un bord de frange est présent
                # 1. Récupérer les coordonnées uR, vR au pixel (i,j)
                uR = uR_grid[i, j]
                vR = vR_grid[i, j]
                
                # 2. Calculer la coordonnée vE correspondante sur l'émetteur
                # Selon la méthode du code binaire : vE = (NbHE / 2^N) * C
                vE = (NbHE / (2**N)) * C
                
                # 3. Construction du système G * X = H (Triangulation)
                G = np.array([
                    [MR[0,0] - uR*MR[2,0], MR[0,1] - uR*MR[2,1], MR[0,2] - uR*MR[2,2]],
                    [MR[1,0] - vR*MR[2,0], MR[1,1] - vR*MR[2,1], MR[1,2] - vR*MR[2,2]],
                    [ME[1,0] - vE*ME[2,0], ME[1,1] - vE*ME[2,1], ME[1,2] - vE*ME[2,2]]
                ])
                
                H = np.array([
                    uR*MR[2,3] - MR[0,3],
                    vR*MR[2,3] - MR[1,3],
                    vE*ME[2,3] - ME[1,3]
                ])
                
                # 4. Résolution par moindres carrés
                try:
                    coord= np.linalg.lstsq(G, H, rcond=None)[0]
                    points_3d.append(coord)
                except np.linalg.LinAlgError:
                    continue

    points_3d = np.array(points_3d)

    # ==========================================
    # 4. Affichage
    # ==========================================
    if points_3d.size > 0:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot avec couleur basée sur la profondeur Z
        p = ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], 
                       c=points_3d[:,2], cmap='jet', s=2)
        
        ax.set_title("Reconstitution 3D de l'objet")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_box_aspect([1.5,1.5,1])
        ax.set_zlim(-20,150)
        fig.colorbar(p, label="Profondeur")
        plt.show()
    else:
        print("Aucun point 3D n'a pu être reconstruit. Vérifiez PosiGauche.")

if __name__ == "__main__":
    reconstructor_3d()