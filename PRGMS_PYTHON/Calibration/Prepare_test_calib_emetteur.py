from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = REPO_ROOT / "img"
STORAGE_DIR = REPO_ROOT / "stockage"
DOC_PATH = REPO_ROOT / "PRGMS_PYTHON" / "Calibration" / "TEST_CALIB_EMETTEUR.md"

WIDTH = 1280
HEIGHT = 800
GRID_COLS = 8
GRID_ROWS = 8

POINTS_EMITTER = np.array(
    [
        [100, 320],
        [400, 320],
        [700, 320],
        [100, 640],
        [400, 640],
        [700, 640],
        [100, 960],
        [400, 960],
        [700, 960],
        [100, 1280],
        [400, 1280],
        [700, 1280],
    ],
    dtype=int,
)


def build_checkerboard():
    y_coords, x_coords = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing="ij")
    pattern = (
        np.sin(y_coords * GRID_COLS * np.pi / HEIGHT)
        * np.sin(x_coords * GRID_ROWS * np.pi / WIDTH)
        < 0
    ).astype(np.uint8)

    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :, 2] = pattern * 255
    return image


def to_xy(point):
    u_value, v_value = int(point[0]), int(point[1])
    return v_value, u_value


def annotate_image(base_image):
    annotated = base_image.copy()

    for index, point in enumerate(POINTS_EMITTER, start=1):
        x_value, y_value = to_xy(point)
        cv2.circle(annotated, (x_value, y_value), 16, (0, 255, 255), thickness=2)
        cv2.circle(annotated, (x_value, y_value), 3, (255, 255, 255), thickness=-1)
        cv2.putText(
            annotated,
            str(index),
            (x_value + 18, y_value - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            str(index),
            (x_value + 18, y_value - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_points_templates():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(STORAGE_DIR / "mE_emetteur.txt", np.vstack([POINTS_EMITTER, POINTS_EMITTER]), fmt="%d")
    np.savetxt(STORAGE_DIR / "mR_emetteur_experimental_z0.txt", np.zeros((12, 2)), fmt="%d")
    np.savetxt(STORAGE_DIR / "mR_emetteur_experimental_z100.txt", np.zeros((12, 2)), fmt="%d")
    np.savetxt(STORAGE_DIR / "mR_emetteur_experimental.txt", np.zeros((24, 2)), fmt="%d")
    np.savetxt(STORAGE_DIR / "Z_emetteur_experimental.txt", np.array([0] * 12 + [100] * 12), fmt="%d")


def build_markdown():
    return """# Test calibration emetteur

## Fichiers generes

- `img/Mire_damier.bmp` : mire a projeter.
- `img/Mire_damier_points.png` : mire avec numerotation des 12 points a relever.
- `stockage/mE_emetteur.txt` : coordonnees LCD des 24 points `mEj`.
- `stockage/mR_emetteur_experimental_z0.txt` : 12 points camera a renseigner pour le plan `Z = 0 mm`.
- `stockage/mR_emetteur_experimental_z100.txt` : 12 points camera a renseigner pour le plan `Z = 100 mm`.
- `stockage/mR_emetteur_experimental.txt` : version concatenee des 24 points camera utilisee par le script.
- `stockage/Z_emetteur_experimental.txt` : profondeurs associees aux 24 points.

## Procedure

1. Projeter `img/Mire_damier.bmp` sur le projecteur.
2. Afficher la mire en plein ecran avec :
   `python PRGMS_PYTHON\\Calibration\\Projeter_image_plein_ecran.py img\\Mire_damier.bmp`
3. Positionner une surface plane a `Z = 0 mm`, prendre une photo nette.
4. Positionner la meme surface plane a `Z = 100 mm`, prendre une photo nette.
5. Ouvrir `img/Mire_damier_points.png` et relever dans l'ordre les 12 points visibles sur chaque photo.
6. Remplir `stockage/mR_emetteur_experimental_z0.txt` puis `stockage/mR_emetteur_experimental_z100.txt`.
7. Concatener les deux fichiers dans `stockage/mR_emetteur_experimental.txt` dans cet ordre : `Z=0`, puis `Z=100`.
8. Lancer :
   `python PRGMS_PYTHON\\Calibration\\Calib_emetteur.py`

## Ordre des points

L'ordre est impose par la numerotation de `img/Mire_damier_points.png`.
Les 12 premiers points correspondent a `Z = 0 mm`.
Les 12 suivants correspondent aux memes points pour `Z = 100 mm`.

## Remarques

- Les fichiers `mR_*` doivent contenir deux colonnes : `uR vR`.
- Le script de calibration utilise `stockage/MR.txt` pour retroprojeter les points objet.
- Si tu mesures une autre distance que `100 mm`, il faut aussi mettre a jour `stockage/Z_emetteur_experimental.txt`.
"""


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    checkerboard = build_checkerboard()
    cv2.imwrite(str(IMG_DIR / "Mire_damier.bmp"), checkerboard)
    cv2.imwrite(str(IMG_DIR / "Mire_damier_points.png"), annotate_image(checkerboard))

    save_points_templates()
    write_text(DOC_PATH, build_markdown())

    print(f"Mire generee : {IMG_DIR / 'Mire_damier.bmp'}")
    print(f"Mire annotee : {IMG_DIR / 'Mire_damier_points.png'}")
    print(f"Guide : {DOC_PATH}")


if __name__ == "__main__":
    main()
