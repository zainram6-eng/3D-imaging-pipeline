"""
=============================================================================
CALIBRATION DU RECEPTEUR - IMAGEUR 3D PAR CODE BINAIRE
=============================================================================
Auteur  : Code généré pour la calibration de l'imageur 3D (IMT Atlantique)
Méthode : Méthode de Faugeras-Toscani simplifiée (matrice perspective normalisée)
Ref doc : "Imageur 3D à Code Binaire" - SERVAGENT Noël R9 / Révision Elisabeth Lys R13

Description :
    Ce script réalise la calibration géométrique du récepteur (caméra) en :
    1. Permettant la détection automatique des points rouges sur une image damier
       OU l'injection manuelle des coordonnées pixels mesurées
    2. Résolvant le système linéaire (Eq. 30 du document) par pseudo-inverse (moindres carrés)
       pour obtenir la matrice de projection perspective normalisée MRN
    3. Remontant aux paramètres intrinsèques et extrinsèques (Eq. 28 du document)

DONNÉES À FOURNIR PAR L'UTILISATEUR (section "DONNÉES UTILISATEUR") :
    - Coordonnées 3D des points de calibration objet Mi(Xi, Yi, Zi) [mm]
    - Coordonnées 2D pixel des points images mRi(uRi, vRi) [pixels]
      → soit par détection automatique sur image (points rouges sur damier)
      → soit par saisie manuelle
    - Valeur de t3R (cote ZEO de l'origine du repère objet dans le repère récepteur) [mm]
      → à mesurer physiquement sur le banc optique (distance caméra-objet environ)

Caméra utilisée : LOGITECH C920E
    - Résolution : 1920 x 1080 pixels
    - Taille pixel : 2.2 µm
    - Focale nominale : 3.67 mm
    - Facteur d'échelle : kvR = kuR = 454.55 px/mm
=============================================================================
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import lstsq
import os

# ===========================================================================
# SECTION 1 : PARAMÈTRES TECHNIQUES DE LA CAMÉRA (valeurs du document)
# ===========================================================================

# Résolution de la caméra LOGITECH C920E
NbHR = 1920   # Nombre de pixels horizontaux
NbVR = 1080   # Nombre de pixels verticaux

# Paramètres physiques du capteur CCD
TAILLE_PIXEL_UM = 2.2        # Taille d'un pixel en µm
FOCALE_NOM_MM   = 3.67       # Focale nominale en mm
KVR = 1000.0 / TAILLE_PIXEL_UM  # Facteur d'échelle horizontal en px/mm ≈ 454.55
KUR = KVR                        # Identique (pixels carrés)

# Centre optique (par hypothèse : centre du capteur)
U0R_NOMINAL = NbVR / 2.0 - 0.5   # 539.5 px
V0R_NOMINAL = NbHR / 2.0 - 0.5   # 959.5 px

print("=" * 70)
print("CALIBRATION DU RÉCEPTEUR - IMAGEUR 3D PAR CODE BINAIRE")
print("=" * 70)
print(f"Caméra    : LOGITECH C920E  ({NbHR} x {NbVR} px)")
print(f"ku = kv   : {KUR:.2f} px/mm")
print(f"Focale    : {FOCALE_NOM_MM} mm (nominale)")
print(f"u0 nominal: {U0R_NOMINAL} px  |  v0 nominal: {V0R_NOMINAL} px")
print()


# ===========================================================================
# SECTION 2 : DONNÉES UTILISATEUR
# ===========================================================================
# ⚠️  REMPLACEZ CES VALEURS PAR VOS MESURES RÉELLES
# ===========================================================================

# ---------------------------------------------------------------------------
# 2.1  COORDONNÉES 3D DES POINTS DE CALIBRATION OBJET Mi(Xi, Yi, Zi)  [mm]
#      Repère objet centré au milieu de la mire damier
#      Deux plans : Z = 0 mm et Z = 100 mm
#      Format : tableau Nx3  [Xi, Yi, Zi]
# ---------------------------------------------------------------------------
# Mire damier : 995 x 622 mm, 9 points par plan
# Points M1..M9 → Z = 0 mm,  Points M10..M18 → Z = 100 mm

points_3D_objet = np.array([
    # --- Plan Z = 0 mm ---
    [-497.5,  234.0,   0.0],   # M1
    [-497.5,    0.0,   0.0],   # M2
    [-497.5, -234.0,   0.0],   # M3
    [   0.0,  311.0,   0.0],   # M4
    [   0.0,    0.0,   0.0],   # M5
    [   0.0, -311.0,   0.0],   # M6
    [ 497.5,  234.0,   0.0],   # M7
    [ 497.5,    0.0,   0.0],   # M8
    [ 497.5, -234.0,   0.0],   # M9
    # --- Plan Z = 100 mm ---
    [-497.5,  234.0, 100.0],   # M10
    [-497.5,    0.0, 100.0],   # M11
    [-497.5, -234.0, 100.0],   # M12
    [   0.0,  311.0, 100.0],   # M13
    [   0.0,    0.0, 100.0],   # M14
    [   0.0, -311.0, 100.0],   # M15
    [ 497.5,  234.0, 100.0],   # M16
    [ 497.5,    0.0, 100.0],   # M17
    [ 497.5, -234.0, 100.0],   # M18
], dtype=float)

# ---------------------------------------------------------------------------
# 2.2  COORDONNÉES PIXELS DES POINTS IMAGES mRi(uRi, vRi)  [pixels]
#      Mesurées sur les images acquises par la caméra
#      → Remplacez par vos mesures réelles ou utilisez la détection automatique
#      Format : tableau Nx2  [uRi, vRi]
#
#      NOTE : les valeurs ci-dessous sont celles du document (simulation)
#      Pour un test réel, mesurez ces points sur vos images !
# ---------------------------------------------------------------------------
points_2D_image = np.array([
    # --- Plan Z = 0 mm ---
    [165,  209],    # mR1
    [539,  209],    # mR2
    [913,  209],    # mR3
    [120,  959],    # mR4
    [539,  959],    # mR5
    [967,  959],    # mR6
    [257, 1517],    # mR7
    [539, 1517],    # mR8
    [822, 1517],    # mR9
    # --- Plan Z = 100 mm ---
    [127,  195],    # mR10
    [539,  195],    # mR11
    [951,  195],    # mR12
    [ 85, 1010],    # mR13
    [539, 1010],    # mR14
    [1005, 1010],   # mR15
    [236, 1603],    # mR16
    [539, 1603],    # mR17
    [843, 1603],    # mR18
], dtype=float)

# ---------------------------------------------------------------------------
# 2.3  COTE t3R : distance de l'origine du repère objet au centre optique
#                du récepteur, projetée sur l'axe optique  [mm]
#      → À MESURER physiquement sur le banc : distance approximative
#        entre la caméra et l'objet selon l'axe optique de la caméra
#      Dans la simulation du document : t3R ≈ 1210.656 mm
# ---------------------------------------------------------------------------
t3R_mesure = 1210.656   # ← REMPLACEZ par votre mesure réelle [mm]


# ===========================================================================
# SECTION 3 : DÉTECTION AUTOMATIQUE DES POINTS ROUGES (OPTIONNEL)
# ===========================================================================

def detecter_points_rouges(chemin_image, n_points_attendus=9, afficher=True):
    """
    Détecte automatiquement les centres des marqueurs rouges sur une image
    de mire damier et retourne leurs coordonnées pixel (u, v).

    Paramètres
    ----------
    chemin_image        : str   – chemin vers l'image PNG/JPG
    n_points_attendus   : int   – nombre de points rouges attendus
    afficher            : bool  – afficher le résultat avec matplotlib

    Retourne
    --------
    centres : ndarray (N, 2) – coordonnées (u=ligne, v=colonne) des centres
                               triées de haut en bas, puis gauche à droite
    """
    img = cv2.imread("Pronto-Imageur-3D\img\damier.jpg")
    if img is None:
        raise FileNotFoundError(f"Image non trouvée : {chemin_image}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Masque couleur rouge (deux plages HSV pour couvrir rouge saturé)
    masque1 = cv2.inRange(img_hsv, np.array([0, 120, 70]),  np.array([10, 255, 255]))
    masque2 = cv2.inRange(img_hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    masque  = cv2.bitwise_or(masque1, masque2)

    # Nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  kernel, iterations=2)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Détection des contours et calcul des centroïdes
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_filtres = [c for c in contours if cv2.contourArea(c) > 50]

    centres = []
    for c in contours_filtres:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]   # colonne → v
            cy = M["m01"] / M["m00"]   # ligne   → u
            centres.append((cy, cx))   # (u=ligne, v=colonne)

    centres = np.array(sorted(centres, key=lambda p: (round(p[0] / 50) * 50, p[1])))

    if afficher and len(centres) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Image originale")
        axes[0].axis("off")

        axes[1].imshow(img_rgb)
        for i, (u, v) in enumerate(centres):
            axes[1].plot(v, u, 'yo', markersize=10, markeredgecolor='k')
            axes[1].text(v + 8, u - 8, f'mR{i+1}', color='yellow',
                         fontsize=8, fontweight='bold')
        axes[1].set_title(f"Points détectés : {len(centres)}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/detection_points.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[INFO] {len(centres)} points rouges détectés.")
        if len(centres) != n_points_attendus:
            print(f"[AVERT] {n_points_attendus} points attendus, {len(centres)} trouvés.")
            print("        → Vérifiez les seuils HSV ou saisissez les points manuellement.")

    return centres   # shape (N, 2) : [uRi, vRi]


# ===========================================================================
# SECTION 4 : CONSTRUCTION DU SYSTÈME LINÉAIRE (Eq. 30 du document)
# ===========================================================================

def construire_systeme_lineaire(pts_3D, pts_2D):
    """
    Construit la matrice B (2n × 11) et le vecteur c (2n × 1) du système
    B · x = c  où x contient les 11 coefficients normalisés de MNR
    (le 12ème coefficient m34NR est normalisé à 1).cv2

    Pour chaque point i :
        Ligne 2i   : [Xi Yi Zi 1  0  0  0  0  -uRi·Xi  -uRi·Yi  -uRi·Zi] · x = uRi
        Ligne 2i+1 : [ 0  0  0  0 Xi Yi Zi  1  -vRi·Xi  -vRi·Yi  -vRi·Zi] · x = vRi

    Paramètres
    ----------
    pts_3D : ndarray (n, 3) – coordonnées objet [Xi, Yi, Zi]
    pts_2D : ndarray (n, 2) – coordonnées image [uRi, vRi]

    Retourne
    --------
    B : ndarray (2n, 11)
    c : ndarray (2n,)
    """
    n = pts_3D.shape[0]
    B = np.zeros((2 * n, 11))
    c = np.zeros(2 * n)

    for i in range(n):
        Xi, Yi, Zi = pts_3D[i]
        uRi, vRi   = pts_2D[i]

        # Ligne paire → équation sur uR
        B[2*i, 0:4]  = [Xi, Yi, Zi, 1]
        B[2*i, 8:11] = [-uRi * Xi, -uRi * Yi, -uRi * Zi]
        c[2*i]       = uRi

        # Ligne impaire → équation sur vR
        B[2*i+1, 4:8]  = [Xi, Yi, Zi, 1]
        B[2*i+1, 8:11] = [-vRi * Xi, -vRi * Yi, -vRi * Zi]
        c[2*i+1]        = vRi

    return B, c


def resoudre_matrice_perspective(pts_3D, pts_2D):
    """
    Résout B · x = c par pseudo-inverse (moindres carrés, Eq. 22 du document)
    et reconstruit la matrice de projection perspective normalisée MNR (3×4).

    x = B⁺ · c  avec  B⁺ = (Bᵀ B)⁻¹ Bᵀ
    MNR = [[x1..x4], [x5..x8], [x9 x10 x11 1]]

    Paramètres
    ----------
    pts_3D : ndarray (n, 3)
    pts_2D : ndarray (n, 2)

    Retourne
    --------
    MNR : ndarray (3, 4) – matrice de projection perspective normalisée
    residus : float      – norme des résidus (indicateur qualité)
    """
    B, c = construire_systeme_lineaire(pts_3D, pts_2D)

    # Résolution par pseudo-inverse (moindres carrés)
    x, residus, rang, sv = lstsq(B, c)

    # Reconstruction de MNR (3×4), le 12ème coeff vaut 1 (normalisation)
    MNR = np.array([
        [x[0], x[1], x[2],  x[3]],
        [x[4], x[5], x[6],  x[7]],
        [x[8], x[9], x[10], 1.0 ]
    ])

    res_norm = np.linalg.norm(B @ x - c)
    print(f"[INFO] Rang de B : {rang} / 11")
    print(f"[INFO] Norme résidus ||Bx - c|| = {res_norm:.4f} px")

    return MNR, res_norm


# ===========================================================================
# SECTION 5 : EXTRACTION DES PARAMÈTRES INTRINSÈQUES ET EXTRINSÈQUES (Eq. 28)
# ===========================================================================

def extraire_parametres(MNR, t3R):
    """
    Remonte aux paramètres intrinsèques et extrinsèques du récepteur
    à partir de la matrice de projection perspective normalisée MNR.

    La matrice de projection réelle est :  MR = t3R · MNR

    Paramètres intrinsèques (Eq. 28) :
        u0R = m1R · m3R
        v0R = m2R · m3R
        αuR = -‖m1R ∧ m3R‖   (négatif car axe u inversé, cf Eq. 15)
        αvR = +‖m2R ∧ m3R‖

    Paramètres extrinsèques (Eq. 28) :
        r3R = m3R  (vecteur de rotation ligne 3)
        r1R = (m1R - u0R·m3R) / αuR
        r2R = (m2R - v0R·m3R) / αvR
        t1R = (m14R - u0R·m34R) / αuR
        t2R = (m24R - v0R·m34R) / αvR
        t3R = m34R  (= valeur mesurée physiquement)

    Focales reconstruites :
        fR_u = -αuR / kuR  [mm]
        fR_v =  αvR / kvR  [mm]

    Paramètres
    ----------
    MNR : ndarray (3, 4) – matrice normalisée issue de la calibration
    t3R : float          – cote Z de l'origine objet dans le repère récepteur [mm]

    Retourne
    --------
    dict contenant tous les paramètres
    """
    # Matrice perspective réelle (non normalisée)
    MR = t3R * MNR

    # Vecteurs lignes de MR
    m1R = MR[0, 0:3]
    m2R = MR[1, 0:3]
    m3R = MR[2, 0:3]
    m14R = MR[0, 3]
    m24R = MR[1, 3]
    m34R = MR[2, 3]   # = t3R par construction

    # ---- Paramètres intrinsèques ----
    u0R  = np.dot(m1R, m3R)
    v0R  = np.dot(m2R, m3R)
    alphaU_R = -np.linalg.norm(np.cross(m1R, m3R))   # négatif (cf Eq. 15)
    alphaV_R =  np.linalg.norm(np.cross(m2R, m3R))   # positif

    # Focales reconstruites à partir des paramètres intrinsèques
    # αuR = -kuR · fR  →  fR = -αuR / kuR
    fR_u = -alphaU_R / KUR
    fR_v =  alphaV_R / KVR

    # ---- Paramètres extrinsèques ----
    r3R = m3R / np.linalg.norm(m3R)  # normalisé car m3R = r3R (rotation ⇒ norme 1)
    r1R = (m1R - u0R * m3R) / alphaU_R
    r2R = (m2R - v0R * m3R) / alphaV_R

    t1R = (m14R - u0R * m34R) / alphaU_R
    t2R = (m24R - v0R * m34R) / alphaV_R

    # Matrice de rotation RR (3×3)
    RR = np.array([r1R, r2R, r3R])

    # Vérification orthogonalité de RR
    orth_err = np.linalg.norm(RR @ RR.T - np.eye(3))

    return {
        "MNR"       : MNR,
        "MR"        : MR,
        "u0R"       : u0R,
        "v0R"       : v0R,
        "alphaU_R"  : alphaU_R,
        "alphaV_R"  : alphaV_R,
        "fR_u_mm"   : fR_u,
        "fR_v_mm"   : fR_v,
        "r1R"       : r1R,
        "r2R"       : r2R,
        "r3R"       : r3R,
        "t1R"       : t1R,
        "t2R"       : t2R,
        "t3R"       : m34R,
        "RR"        : RR,
        "TR"        : np.array([t1R, t2R, m34R]),
        "orth_err"  : orth_err,
    }


# ===========================================================================
# SECTION 6 : VALIDATION — REPROJECTION DES POINTS DE CALIBRATION
# ===========================================================================

def reprojeter_points(MNR, pts_3D):
    """
    Reprojette les points 3D dans le plan image via MNR.
    Retourne les coordonnées pixel recalculées (uR_calc, vR_calc).
    """
    n = pts_3D.shape[0]
    pts_calc = np.zeros((n, 2))

    for i in range(n):
        X, Y, Z = pts_3D[i]
        h = MNR @ np.array([X, Y, Z, 1.0])   # coordonnées homogènes
        pts_calc[i, 0] = h[0] / h[2]          # uR
        pts_calc[i, 1] = h[1] / h[2]          # vR

    return pts_calc


def evaluer_erreur_reprojection(pts_2D_mesures, pts_2D_calcules, afficher=True):
    """
    Calcule et affiche l'erreur de reprojection point par point.
    """
    erreurs = np.linalg.norm(pts_2D_mesures - pts_2D_calcules, axis=1)
    err_moy = np.mean(erreurs)
    err_max = np.max(erreurs)

    if afficher:
        print("\n--- Erreurs de reprojection ---")
        print(f"{'Point':>6} | {'uR_mes':>8} {'vR_mes':>8} | {'uR_calc':>8} {'vR_calc':>8} | {'Erreur':>8} px")
        print("-" * 65)
        for i, (mes, calc, err) in enumerate(zip(pts_2D_mesures, pts_2D_calcules, erreurs)):
            print(f"  mR{i+1:02d} | {mes[0]:8.1f} {mes[1]:8.1f} | {calc[0]:8.2f} {calc[1]:8.2f} | {err:8.3f}")
        print("-" * 65)
        print(f"  Erreur moyenne : {err_moy:.3f} px")
        print(f"  Erreur maximale: {err_max:.3f} px")

    return erreurs, err_moy, err_max


def afficher_reprojection(pts_2D_mesures, pts_2D_calcules):
    """Visualise les points mesurés vs reprojettés."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, NbHR)
    ax.set_ylim(NbVR, 0)
    ax.set_facecolor('#1a1a2e')
    ax.set_title("Reprojection : Points mesurés vs calculés", fontsize=13, color='white')
    ax.set_xlabel("vR [pixels]", color='white')
    ax.set_ylabel("uR [pixels]", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    fig.patch.set_facecolor('#1a1a2e')

    for i, (mes, calc) in enumerate(zip(pts_2D_mesures, pts_2D_calcules)):
        ax.plot(mes[1],  mes[0],  'go', markersize=9, zorder=5)
        ax.plot(calc[1], calc[0], 'r+', markersize=12, markeredgewidth=2, zorder=6)
        ax.plot([mes[1], calc[1]], [mes[0], calc[0]], 'w-', linewidth=0.8, alpha=0.6)
        ax.annotate(f"mR{i+1}", (mes[1], mes[0]), color='cyan',
                    fontsize=7, xytext=(4, -10), textcoords='offset points')

    green_patch = mpatches.Patch(color='green', label='Points mesurés')
    red_patch   = mpatches.Patch(color='red',   label='Points reprojettés (calculés)')
    ax.legend(handles=[green_patch, red_patch], facecolor='#2a2a3e', labelcolor='white')

    plt.tight_layout()
    plt.savefig("reprojection_calibration.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[INFO] Figure sauvegardée : reprojection_calibration.png")


# ===========================================================================
# SECTION 7 : AFFICHAGE COMPLET DES RÉSULTATS
# ===========================================================================

def afficher_resultats(params):
    """Affiche un rapport complet de calibration."""
    sep = "=" * 70

    print(f"\n{sep}")
    print("  RÉSULTATS DE CALIBRATION DU RÉCEPTEUR")
    print(sep)

    print("\n[1] Matrice de projection perspective normalisée MNR :")
    MNR = params["MNR"]
    for row in MNR:
        print(f"    {row[0]:+14.6e}  {row[1]:+14.6e}  {row[2]:+14.6e}  {row[3]:+14.6e}")

    print("\n[2] Paramètres INTRINSÈQUES :")
    print(f"    u0R     = {params['u0R']:>12.4f} px   (nominal: {U0R_NOMINAL:.1f} px)")
    print(f"    v0R     = {params['v0R']:>12.4f} px   (nominal: {V0R_NOMINAL:.1f} px)")
    print(f"    αuR     = {params['alphaU_R']:>12.4f} px·mm⁻¹")
    print(f"    αvR     = {params['alphaV_R']:>12.4f} px·mm⁻¹")
    print(f"    fR (u)  = {params['fR_u_mm']:>12.4f} mm  (nominal: {FOCALE_NOM_MM} mm)")
    print(f"    fR (v)  = {params['fR_v_mm']:>12.4f} mm")

    print("\n[3] Paramètres EXTRINSÈQUES :")
    print(f"    t1R     = {params['t1R']:>12.4f} mm")
    print(f"    t2R     = {params['t2R']:>12.4f} mm")
    print(f"    t3R     = {params['t3R']:>12.4f} mm  (valeur fournie)")
    print(f"    r1R     = [{params['r1R'][0]:+.6f}, {params['r1R'][1]:+.6f}, {params['r1R'][2]:+.6f}]")
    print(f"    r2R     = [{params['r2R'][0]:+.6f}, {params['r2R'][1]:+.6f}, {params['r2R'][2]:+.6f}]")
    print(f"    r3R     = [{params['r3R'][0]:+.6f}, {params['r3R'][1]:+.6f}, {params['r3R'][2]:+.6f}]")

    print("\n[4] Matrice de rotation RR :")
    for row in params["RR"]:
        print(f"    [{row[0]:+.8f}  {row[1]:+.8f}  {row[2]:+.8f}]")
    print(f"    → Erreur orthogonalité ‖RR·RRᵀ - I‖ = {params['orth_err']:.2e}")

    print(f"\n{sep}")


# ===========================================================================
# SECTION 8 : EXPORT DES RÉSULTATS VERS UN FICHIER TEXTE
# ===========================================================================

def exporter_resultats(params, pts_2D_mes, pts_2D_calc, erreurs,
                        fichier="resultats_calibration.txt"):
    """Sauvegarde les résultats dans un fichier texte."""
    with open(fichier, "w") as f:
        f.write("CALIBRATION RÉCEPTEUR - IMAGEUR 3D PAR CODE BINAIRE\n")
        f.write("=" * 60 + "\n\n")

        f.write("MNR (matrice de projection normalisée) :\n")
        for row in params["MNR"]:
            f.write(f"  {row[0]:+14.8e}  {row[1]:+14.8e}  {row[2]:+14.8e}  {row[3]:+14.8e}\n")

        f.write("\nParamètres intrinsèques :\n")
        f.write(f"  u0R    = {params['u0R']:.6f} px\n")
        f.write(f"  v0R    = {params['v0R']:.6f} px\n")
        f.write(f"  alphaU = {params['alphaU_R']:.6f}\n")
        f.write(f"  alphaV = {params['alphaV_R']:.6f}\n")
        f.write(f"  fR_u   = {params['fR_u_mm']:.6f} mm\n")
        f.write(f"  fR_v   = {params['fR_v_mm']:.6f} mm\n")

        f.write("\nParamètres extrinsèques :\n")
        f.write(f"  t1R  = {params['t1R']:.6f} mm\n")
        f.write(f"  t2R  = {params['t2R']:.6f} mm\n")
        f.write(f"  t3R  = {params['t3R']:.6f} mm\n")
        f.write(f"  r1R  = {params['r1R']}\n")
        f.write(f"  r2R  = {params['r2R']}\n")
        f.write(f"  r3R  = {params['r3R']}\n")
        f.write(f"  RR   =\n")
        for row in params["RR"]:
            f.write(f"    {row}\n")

        f.write("\nErreurs de reprojection :\n")
        for i, (mes, calc, err) in enumerate(zip(pts_2D_mes, pts_2D_calc, erreurs)):
            f.write(f"  mR{i+1:02d}: mes=({mes[0]:.1f},{mes[1]:.1f})  "
                    f"calc=({calc[0]:.2f},{calc[1]:.2f})  err={err:.3f} px\n")
        f.write(f"\nErreur moyenne : {np.mean(erreurs):.3f} px\n")
        f.write(f"Erreur max     : {np.max(erreurs):.3f} px\n")

    print(f"[INFO] Résultats exportés → {fichier}")


# ===========================================================================
# SECTION 9 : PIPELINE PRINCIPAL
# ===========================================================================

def pipeline_calibration(pts_3D, pts_2D, t3R,
                          image_path=None,
                          plan_z0_only=False):
    """
    Pipeline complet de calibration :
      1. (Optionnel) Détection des points rouges sur image
      2. Résolution du système linéaire → MNR
      3. Extraction des paramètres
      4. Validation par reprojection
      5. Affichage et export

    Paramètres
    ----------
    pts_3D      : ndarray (n, 3)  – points objet [mm]
    pts_2D      : ndarray (n, 2)  – points image [px]
    t3R         : float           – cote t3R mesurée [mm]
    image_path  : str ou None     – si fourni, tente la détection automatique
    plan_z0_only: bool            – si True, n'utilise que les 9 premiers points
    """
    print("\n[ÉTAPE 1] Données de calibration")
    print(f"  Nombre de points : {pts_3D.shape[0]}")
    print(f"  t3R utilisé      : {t3R} mm")

    # -------- Détection automatique (si image fournie) ---------------------
    if image_path is not None and os.path.exists(image_path):
        print(f"\n[ÉTAPE 1b] Détection automatique des points rouges sur {image_path}")
        try:
            pts_detectes = detecter_points_rouges(image_path,
                                                   n_points_attendus=9 if plan_z0_only else 18)
            print(f"  Coordonnées détectées :\n{pts_detectes}")
            print("  → Utilisez ces valeurs pour remplacer points_2D_image dans la Section 2.2")
        except Exception as e:
            print(f"  [AVERT] Détection échouée : {e}")

    # -------- Calibration --------------------------------------------------
    print("\n[ÉTAPE 2] Résolution du système linéaire (moindres carrés)")
    MNR, res = resoudre_matrice_perspective(pts_3D, pts_2D)

    print("\n[ÉTAPE 3] Extraction des paramètres intrinsèques et extrinsèques")
    params = extraire_parametres(MNR, t3R)

    # -------- Validation ---------------------------------------------------
    print("\n[ÉTAPE 4] Validation par reprojection")
    pts_calc = reprojeter_points(MNR, pts_3D)
    erreurs, err_moy, err_max = evaluer_erreur_reprojection(pts_2D, pts_calc)
    afficher_reprojection(pts_2D, pts_calc)

    # -------- Affichage rapport --------------------------------------------
    afficher_resultats(params)

    # -------- Export -------------------------------------------------------
    exporter_resultats(params, pts_2D, pts_calc, erreurs)

    return params, MNR


# ===========================================================================
# SECTION 10 : POINT D'ENTRÉE
# ===========================================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  IMAGEUR 3D - CALIBRATION RÉCEPTEUR (Code Binaire)")
    print("  Référence : IMT Atlantique - SERVAGENT/LYS")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Optionnel : fournissez le chemin vers votre image damier pour la
    # détection automatique des points rouges
    # Ex : IMAGE_DAMIER = "damier_Z0.png"
    # -----------------------------------------------------------------------
    IMAGE_DAMIER = None   # ← REMPLACEZ par votre chemin image si disponible

    # -----------------------------------------------------------------------
    # Lancement du pipeline de calibration
    # -----------------------------------------------------------------------
    params_calibration, MNR_final = pipeline_calibration(
        pts_3D      = points_3D_objet,
        pts_2D      = points_2D_image,
        t3R         = t3R_mesure,
        image_path  = IMAGE_DAMIER,
    )

    # -----------------------------------------------------------------------
    # Résumé final
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RÉSUMÉ — DONNÉES À FOURNIR POUR UN TEST RÉEL :")
    print("=" * 70)
    print("""
  1. points_3D_objet  [mm]   → Coordonnées Xi,Yi,Zi des coins du damier
                               dans le repère centré au milieu de la mire.
                               Mesurez avec une règle ou pied à coulisse.

  2. points_2D_image  [px]   → Coordonnées uRi,vRi des points rouges dans
                               l'image CCD.
                               Méthode A : détection auto (passer IMAGE_DAMIER)
                               Méthode B : saisie manuelle dans la Section 2.2
                               (cliquer sur les points dans GIMP/ImageJ)

  3. t3R_mesure       [mm]   → Distance selon l'axe optique entre la caméra
                               et le plan Z=0 de la mire.
                               Mesurez au mètre-ruban ou au laser.

  NOTE : Pour améliorer la précision, utilisez n ≥ 18 points (deux plans Z)
         et distribuez les points dans tout le champ de vue de la caméra.
""")
    print("Calibration terminée. Fichiers générés :")
    print("  - /mnt/user-data/outputs/resultats_calibration.txt")
    print("  - /mnt/user-data/outputs/reprojection_calibration.png")