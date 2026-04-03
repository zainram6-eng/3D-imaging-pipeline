# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 08:28:41 2017
MAJ octobre 2023
@author: Elisabeth Lys
"""
# %%
import time
start_time = time.process_time() # début mesure temps d'éxecusion

# On importe le module matplotlib qui permet de générer des graphiques 2D et 3D
import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
from skimage.morphology import disk
# On importe le module numpy qui permet de faire du calcul numérique
import numpy as np
from numpy import loadtxt, empty, zeros, ones, savetxt

#Chargement nombre d'image
N = loadtxt('Pronto-Imageur-3D/stockage/N.txt', np.int32)
#Chargement abscisses et ordonnées récepteur zoom
uRzoomvect = loadtxt('Pronto-Imageur-3D/stockage/uRzoomvect.txt')
vRzoomvect = loadtxt('Pronto-Imageur-3D/stockage/vRzoomvect.txt')

#Chargement abscisses et ordonnées récepteur zoom
uRzoom = loadtxt('Pronto-Imageur-3D/stockage/uRzoom.txt')
vRzoom = loadtxt('Pronto-Imageur-3D/stockage/vRzoom.txt')

#Taille récepteur zoom
NbVRzoom = len(uRzoomvect)
NbHRzoom = len(vRzoomvect)

#On créé la matrice IRzoom
IRzoom = zeros((NbHRzoom,NbVRzoom,N))
Posiglobal = zeros((NbHRzoom,NbVRzoom))
PosiGauche = zeros((NbHRzoom,NbVRzoom))
PosiDroite = zeros((NbHRzoom,NbVRzoom))

# Dans la boucle de chargement des images :
for k in range (N):
    Nom = f'Pronto-Imageur-3D/img/IRZoomt{str(k + 1)}.bmp'
    img = io.imread(Nom)
    gray = img[:,:,0]
    
    # CALCUL D'UN SEUIL AUTOMATIQUE
    thresh_val = filters.threshold_otsu(gray) 
    idx = gray > thresh_val
    
    IRz = zeros(gray.shape)
    IRz[idx] = 1  # Binarisation propre
    IRzoom[:,:,k] = IRz

#Libération mémoire
Nom = None
R = None     

# ----------------------Localisation de la frange C 
    
#On initialise les variales LClogic et LC 
LC_num = empty((NbHRzoom,NbVRzoom))
LClogic = ones((NbHRzoom,NbVRzoom), dtype=bool) 

LC = zeros((NbHRzoom,NbVRzoom))
for C in range (1,2**N+1,2):

    #Numéro base binaire de la frange à localiser
    Cbin = np.binary_repr(C, N)
    
    # On transforme le nombre binaire en une liste dont on peut chercher les éléments un à un
    tabCbin = list(map(int, Cbin)) 
    
    # Localisation de la frange C impaire
    LClogic = IRzoom[:,:,0]==tabCbin[0]
    for l in range (1,N):
        LClogic = LClogic & (IRzoom[:,:,l]==tabCbin[l])
 
    # Matrice de localisation numérique 
    LC = LClogic*1
    
    # Libération de la mémoire
    del LClogic
    # IRzoom[:,:,l] = None
    # Filtrage médian sur les images des franges permet de nettoyer les petites imperfections
    LC = filters.median(np.uint8(LC),disk(1))

    for i in range (NbHRzoom): #NbHRzoom
        p0 = np.nonzero(LC[i,:] == 1)[0]
        if len(p0) > 0: 
            aG = p0[0] # first element -> début de frange
            aD = p0[-1] # last element -> fin de frange
            # on concatène les position 
            # Position G et D
            Posiglobal[i,aG] = C 
            Posiglobal[i,aD] = C + 1 
            #on concatène les position gauche avec les positions G existantes
            PosiGauche[i,aG] = C
            #on concatène les position droite avec les positions D existantes
            PosiDroite[i,aD] = C + 1

#Enregistrement 
savetxt('Pronto-Imageur-3D/stockage/PosiGauche.txt', PosiGauche, fmt='%-7.0f')
savetxt('Pronto-Imageur-3D/stockage/PosiDroite.txt', PosiDroite, fmt='%-7.0f')
savetxt('Pronto-Imageur-3D/stockage/Posiglobal.txt', Posiglobal, fmt='%-7.0f')
savetxt('Pronto-Imageur-3D/stockage/LC_num.txt', LC_num, fmt='%-7.0f')

#Inversion de contraste pour l'affichage
InvPosiglobal=1.-Posiglobal

#Enregistrement image des cotes de franges
couleur_cotes = np.asarray([255,255,255])  #Blanc intensitee maximale
B = zeros((NbHRzoom,NbVRzoom,3))

A = 'Pronto-Imageur-3D/img/Cotes_franges.bmp'
B[:,:,0] = couleur_cotes[0]*InvPosiglobal
B[:,:,1] = couleur_cotes[1]*InvPosiglobal
B[:,:,2] = couleur_cotes[2]*InvPosiglobal
B=B.astype(np.uint8)
io.imsave(A,B)

# Affichage
plt.figure()
plt.imshow(PosiGauche, cmap = plt.get_cmap('gray'))
plt.title('Position Gauche')

plt.figure()
plt.imshow(PosiDroite, cmap = plt.get_cmap('gray'))
plt.title('Position Droite')

plt.figure()
plt.imshow(Posiglobal, cmap = plt.get_cmap('gray'))
plt.title('Position Globale')

#Affichage de l'image enregistrée des positions globales des franges
plt.figure()
plt.imshow(B[:,:,1], cmap = plt.get_cmap('gray'))
plt.title('Image des cotés des franges')
plt.xlabel('vRzoom pixels')
plt.ylabel('uRzomm pixels')

print(time.process_time() - start_time, "seconds")  # fin mesure temps d'éxecusion
# %%
