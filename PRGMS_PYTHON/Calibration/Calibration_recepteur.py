import cv2
import numpy as np
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "stockage"
IMAGE_Z0_PATH = REPO_ROOT / "img" / "damier0.jpg"
IMAGE_Z100_PATH = REPO_ROOT / "img" / "damier100.jpg"


def extraire_centroides(image_path, nb_points=21):
    img = cv2.imread(str(image_path))
    if img is None:
        return np.empty((0, 2), dtype=float)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    mask = cv2.addWeighted(
        cv2.inRange(hsv, lower_red1, upper_red1),
        1.0,
        cv2.inRange(hsv, lower_red2, upper_red2),
        1.0,
        0,
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, _, _, centroids = cv2.connectedComponentsWithStats(mask)
    tous_les_centroides = centroids[1:]

    rayon_securite = 20
    points_filtres = []

    for centroid in tous_les_centroides:
        est_trop_proche = False
        for point_filtre in points_filtres:
            if np.linalg.norm(centroid - point_filtre) < rayon_securite:
                est_trop_proche = True
                break

        if not est_trop_proche:
            points_filtres.append(centroid)

    if not points_filtres:
        return np.empty((0, 2), dtype=float)

    points_filtres = np.array(points_filtres, dtype=float)
    points_filtres = points_filtres[points_filtres[:, 1].argsort()]

    if len(points_filtres) < nb_points:
        print(f"ALERTE : seulement {len(points_filtres)} points trouves apres filtrage.")

    return points_filtres[:nb_points]


def sauvegarder_visualisation_points(image_path, points_2d, output_name):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    annotated = image.copy()

    for index, point in enumerate(points_2d, start=1):
        x_value = int(round(point[0]))
        y_value = int(round(point[1]))
        cv2.circle(annotated, (x_value, y_value), 12, (0, 255, 0), 2)
        cv2.circle(annotated, (x_value, y_value), 3, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            str(index),
            (x_value + 8, y_value - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name
    cv2.imwrite(str(output_path), annotated)
    print(f"Visualisation des points sauvegardee : {output_path}")
    return output_path


def calculer_MRN(points_3d, points_2d):
    n = len(points_3d)
    b_matrix = np.zeros((2 * n, 11))
    c_vector = np.zeros((2 * n, 1))

    for i in range(n):
        x_value, y_value, z_value = points_3d[i]
        u_value, v_value = points_2d[i]

        b_matrix[2 * i] = [
            x_value,
            y_value,
            z_value,
            1,
            0,
            0,
            0,
            0,
            -u_value * x_value,
            -u_value * y_value,
            -u_value * z_value,
        ]
        c_vector[2 * i] = u_value

        b_matrix[2 * i + 1] = [
            0,
            0,
            0,
            0,
            x_value,
            y_value,
            z_value,
            1,
            -v_value * x_value,
            -v_value * y_value,
            -v_value * z_value,
        ]
        c_vector[2 * i + 1] = v_value

    bt_matrix = b_matrix.T
    xR = np.linalg.pinv(bt_matrix @ b_matrix) @ bt_matrix @ c_vector
    return np.append(xR, [1.0]).reshape(3, 4)


def analyser_physique(mrn):
    m3r_norm = mrn[2, :3]
    t3r_calcule = 1.0 / np.linalg.norm(m3r_norm)
    mr = t3r_calcule * mrn

    m1r, m2r, m3r = mr[0, :3], mr[1, :3], mr[2, :3]

    u0r = np.dot(m1r, m3r)
    v0r = np.dot(m2r, m3r)
    alpha_ur = -np.linalg.norm(np.cross(m1r, m3r))
    alpha_vr = np.linalg.norm(np.cross(m2r, m3r))

    r1r = (1 / alpha_ur) * (m1r - u0r * m3r)
    r2r = (1 / alpha_vr) * (m2r - v0r * m3r)
    r3r = m3r
    rr = np.vstack([r1r, r2r, r3r])

    theta = np.degrees(np.arcsin(-rr[2, 0]))
    psi = np.degrees(np.arctan2(rr[2, 1], rr[2, 2]))
    phi = np.degrees(np.arctan2(rr[1, 0], rr[0, 0]))

    np.savetxt(REPO_ROOT / "stockage" / "angles_exp.txt", (phi, theta, psi), fmt="%1.16e")
    np.savetxt(REPO_ROOT / "stockage" / "MR_physique.txt", mr, fmt="%1.16e")

    return mr, t3r_calcule, (u0r, v0r, alpha_ur, alpha_vr), (phi, theta, psi)


M_points_3d = np.array([
    [-0.47, 0.20625, 0], [-0.47, 0.06875, 0], [-0.47, -0.06875, 0], [-0.47, -0.20625, 0],
    [-0.276, 0.20625, 0], [-0.276, 0.06875, 0], [-0.276, -0.06875, 0], [-0.276, -0.20625, 0],
    [-0.092, 0.20625, 0], [-0.092, 0.06875, 0], [-0.092, -0.06875, 0], [-0.092, -0.20625, 0],
    [0.092, 0.20625, 0], [0.092, 0.06875, 0], [0.092, -0.06875, 0], [0.092, -0.20625, 0],
    [0.276, 0.20625, 0], [0.276, 0.06875, 0], [0.276, -0.06875, 0], [0.276, -0.20625, 0],
    [0.368, 0, 0],
    [-0.47, 0.20625, 0.1], [-0.47, 0.06875, 0.1], [-0.47, -0.06875, 0.1], [-0.47, -0.20625, 0.1],
    [-0.276, 0.20625, 0.1], [-0.276, 0.06875, 0.1], [-0.276, -0.06875, 0.1], [-0.276, -0.20625, 0.1],
    [-0.092, 0.20625, 0.1], [-0.092, 0.06875, 0.1], [-0.092, -0.06875, 0.1], [-0.092, -0.20625, 0.1],
    [0.092, 0.20625, 0.1], [0.092, 0.06875, 0.1], [0.092, -0.06875, 0.1], [0.092, -0.20625, 0.1],
    [0.276, 0.20625, 0.1], [0.276, 0.06875, 0.1], [0.276, -0.06875, 0.1], [0.276, -0.20625, 0.1],
    [0.368, 0, 0.1],
])


try:
    pts_2d_0 = extraire_centroides(IMAGE_Z0_PATH, nb_points=21)
    pts_2d_100 = extraire_centroides(IMAGE_Z100_PATH, nb_points=21)

    sauvegarder_visualisation_points(
        IMAGE_Z0_PATH,
        pts_2d_0,
        "calibration_recepteur_points_z0.png",
    )
    sauvegarder_visualisation_points(
        IMAGE_Z100_PATH,
        pts_2d_100,
        "calibration_recepteur_points_z100.png",
    )

    mR_points_2d = np.vstack((pts_2d_0, pts_2d_100))
    MRN_finale = calculer_MRN(M_points_3d, mR_points_2d)
    MR, t3R, intrinseques, angles = analyser_physique(MRN_finale)

    print("-" * 30)
    print("coefficient t3R :", t3R)
    print("MATRICE MR :\n", MR)
    print("-" * 30)
    print(f"Centre optique : u0={intrinseques[0]:.2f}, v0={intrinseques[1]:.2f}")
    print(f"Focales (px)  : alpha_u={intrinseques[2]:.2f}, alpha_v={intrinseques[3]:.2f}")
    print("-" * 30)
    print("ANGLES D'ORIENTATION (Degres) :")
    print(f"Phi (Lacet)    : {angles[0]:.2f} deg")
    print(f"Theta (Tangage): {angles[1]:.2f} deg")
    print(f"Psi (Roulis)   : {angles[2]:.2f} deg")

except Exception as e:
    print("Erreur lors de l'acquisition :", e)
