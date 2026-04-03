# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 09:59:46 2018
MAJ octobre 2023
@author: Elisabeth Lys
"""
import time
start_time = time.process_time() # début mesure temps d'éxecusion
import numpy as np
# On importe le module numpy qui permet de faire du calcul numérique
#import numpy as np
from numpy import meshgrid, sqrt, linspace, savetxt
# On importe le module matplotlib qui permet de générer des graphiques 2D et 3D
import matplotlib.pyplot as plt


#************************************************************************
#*********** Création de l'objet bouclier dans repère Objet (O,X,Y,Z) ***
#************************************************************************
#Nb pixel Objet (échantillonnage objet)
NbHO = 1280
NbVO = 800
#Rayon sphere mm
R = 360;
#Recul sphere mm
a = 300;

#Coordonnées matricielles des pts M de l'objet
[X,Y] = meshgrid(linspace(-600,600,NbHO),linspace(-375,375,NbVO));

#Affixe de l'objet (mm)
# Initialisation de Z à 0 (Condition : si |x|>300 ou |y|>200, z=0)
Z = np.zeros_like(X)

# Masque pour les points à l'intérieur du rectangle [-300, 300] x [-200, 200]
inside_rect = (np.abs(X) <= 300) & (np.abs(Y) <= 200)

# 1. Calcul pour le plan P1 (Région supérieure)
# Equation: -1/200*y - 1/75*z + 1 = 0 => z = 75 * (1 - y/200)
mask_p1 = inside_rect & (Y >= 0) & (Y > -X - 100) & (Y > X - 100)
Z[mask_p1] = 75 * (1 - Y[mask_p1] / 200)

# 2. Calcul pour le plan P2 (Région droite)
# Equation: -1/300*x - 2/225*z + 1 = 0 => z = (225/2) * (1 - x/300)
mask_p2 = inside_rect & (Y > -X + 100) & (Y < X - 100)
Z[mask_p2] = 112.5 * (1 - X[mask_p2] / 300)

# 3. Calcul pour le plan P3 (Région inférieure)
# Equation: 1/200*y - 1/75*z + 1 = 0 => z = 75 * (1 + y/200)
mask_p3 = inside_rect & (Y < 0) & (Y < -X + 100) & (Y < X + 100)
Z[mask_p3] = 75 * (1 + Y[mask_p3] / 200)

# 4. Calcul pour le plan P4 (Région gauche)
# Equation: 1/300*x - 2/225*z + 1 = 0 => z = (225/2) * (1 + x/300)
mask_p4 = inside_rect & (Y < -X - 100) & (Y > X + 100)
Z[mask_p4] = 112.5 * (1 + X[mask_p4] / 300)
#Enregistrement des coordonnées matricelles objet
savetxt('Pronto-Imageur-3D/stockage/Xt.txt', X, fmt='%-7.6f')   
savetxt('Pronto-Imageur-3D/stockage/Yt.txt', Y, fmt='%-7.6f')
savetxt('Pronto-Imageur-3D/stockage/Zt.txt', Z, fmt='%-7.6f')  


#************************************************************************
#************************ Affichage de l'objet  *************************
#************************************************************************

z_min, z_max = 0, abs(Z).max()
plt.figure();
plt.pcolor(X,Y,Z, cmap='gray', vmin=z_min, vmax=z_max)
plt.title('Zt (mm) - Objet bouclier simulé')
# set the limits of the plot to the limits of the data
plt.axis([X.min(), X.max(), Y.min(), Y.max()])
plt.colorbar()
plt.savefig("Pronto-Imageur-3D/img/Objet_toiture.png")
print(time.process_time() - start_time, "seconds")  # fin mesure temps d'éxecusion