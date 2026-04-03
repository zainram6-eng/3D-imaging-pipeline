import argparse
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = REPO_ROOT / "stockage"
DEFAULT_RECEIVER_POINTS_PATH = STORAGE_DIR / "mR_emetteur_experimental.txt"
DEFAULT_RECEIVER_POINTS_Z0_PATH = STORAGE_DIR / "mR_emetteur_experimental_z0.txt"
DEFAULT_RECEIVER_POINTS_Z100_PATH = STORAGE_DIR / "mR_emetteur_experimental_z100.txt"
DEFAULT_EMITTER_POINTS_PATH = STORAGE_DIR / "mE_emetteur.txt"
DEFAULT_DEPTHS_PATH = STORAGE_DIR / "Z_emetteur_experimental.txt"

# Points mEj du LCD utilises pour la calibration de l'emetteur (PDF R13, 7.2.1).
EMITTER_IMAGE_POINTS = np.array(
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
    dtype=float,
)

# Points mRj mesures dans l'image recepteur (PDF R13, 7.2.1).
RECEIVER_IMAGE_POINTS = np.array(
    [
        [200, 596],
        [540, 596],
        [882, 596],
        [242, 960],
        [540, 960],
        [838, 960],
        [276, 1242],
        [540, 1242],
        [805, 1242],
        [302, 1466],
        [540, 1466],
        [778, 1466],
        [201, 707],
        [540, 707],
        [879, 707],
        [244, 1062],
        [540, 1062],
        [836, 1062],
        [277, 1335],
        [540, 1335],
        [803, 1335],
        [304, 1554],
        [540, 1554],
        [777, 1554],
    ],
    dtype=float,
)

POINT_DEPTHS = np.array([0.0] * 12 + [100.0] * 12, dtype=float)


def normalize_projection_matrix(matrix):
    if matrix.shape != (3, 4):
        raise ValueError("La matrice de projection doit etre de taille 3x4.")
    if np.isclose(matrix[2, 3], 0.0):
        raise ValueError("Le coefficient m34 doit etre non nul pour normaliser la matrice.")
    return matrix / matrix[2, 3]


def reconstruct_object_points(receiver_matrix_norm, receiver_points, depths):
    object_points = []

    for (u_r, v_r), z_value in zip(receiver_points, depths):
        system_matrix = np.array(
            [
                [
                    u_r * receiver_matrix_norm[2, 0] - receiver_matrix_norm[0, 0],
                    u_r * receiver_matrix_norm[2, 1] - receiver_matrix_norm[0, 1],
                ],
                [
                    v_r * receiver_matrix_norm[2, 0] - receiver_matrix_norm[1, 0],
                    v_r * receiver_matrix_norm[2, 1] - receiver_matrix_norm[1, 1],
                ],
            ],
            dtype=float,
        )

        rhs = np.array(
            [
                (receiver_matrix_norm[0, 2] - u_r * receiver_matrix_norm[2, 2]) * z_value
                + receiver_matrix_norm[0, 3]
                - u_r,
                (receiver_matrix_norm[1, 2] - v_r * receiver_matrix_norm[2, 2]) * z_value
                + receiver_matrix_norm[1, 3]
                - v_r,
            ],
            dtype=float,
        )

        x_value, y_value = np.linalg.solve(system_matrix, rhs)
        object_points.append([x_value, y_value, z_value])

    return np.array(object_points, dtype=float)


def calibrate_emitter_dlt(object_points, emitter_points):
    if len(object_points) != len(emitter_points):
        raise ValueError("Le nombre de points objet et de points image doit etre identique.")

    num_points = len(object_points)
    b_matrix = np.zeros((2 * num_points, 11), dtype=float)
    c_vector = np.zeros((2 * num_points, 1), dtype=float)

    for index, ((x_value, y_value, z_value), (u_e, v_e)) in enumerate(
        zip(object_points, emitter_points)
    ):
        row = 2 * index
        b_matrix[row] = [
            x_value,
            y_value,
            z_value,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -u_e * x_value,
            -u_e * y_value,
            -u_e * z_value,
        ]
        c_vector[row, 0] = u_e

        b_matrix[row + 1] = [
            0.0,
            0.0,
            0.0,
            0.0,
            x_value,
            y_value,
            z_value,
            1.0,
            -v_e * x_value,
            -v_e * y_value,
            -v_e * z_value,
        ]
        c_vector[row + 1, 0] = v_e

    pseudo_inverse = np.linalg.pinv(b_matrix)
    solution = pseudo_inverse @ c_vector
    matrix_norm = np.append(solution.flatten(), 1.0).reshape(3, 4)
    return matrix_norm


def extract_intrinsic_extrinsic_parameters(matrix):
    m1 = matrix[0, :3]
    m2 = matrix[1, :3]
    m3 = matrix[2, :3]
    m34 = float(matrix[2, 3])

    u0 = float(np.dot(m1, m3))
    v0 = float(np.dot(m2, m3))
    alpha_u = float(np.sqrt(max(np.dot(m1, m1) - u0**2, 0.0)))
    alpha_v = float(np.sqrt(max(np.dot(m2, m2) - v0**2, 0.0)))

    r3 = m3
    r1 = (m1 - u0 * r3) / alpha_u
    r2 = (m2 - v0 * r3) / alpha_v

    t1 = float((matrix[0, 3] - u0 * m34) / alpha_u)
    t2 = float((matrix[1, 3] - v0 * m34) / alpha_v)
    t3 = m34

    return {
        "u0": u0,
        "v0": v0,
        "alpha_u": alpha_u,
        "alpha_v": alpha_v,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "t1": t1,
        "t2": t2,
        "t3": t3,
    }


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, array, fmt="%-7.9f")


def load_points_or_default(path, default_array, expected_columns):
    if path is not None and Path(path).exists():
        loaded = np.loadtxt(path, dtype=float)
        loaded = np.atleast_2d(loaded)
        if loaded.shape[1] != expected_columns:
            raise ValueError(
                f"Le fichier {path} doit contenir {expected_columns} colonnes, pas {loaded.shape[1]}."
            )
        return loaded
    return default_array.copy()


def load_depths_or_default(path, default_array):
    if path is not None and Path(path).exists():
        loaded = np.loadtxt(path, dtype=float)
        loaded = np.atleast_1d(loaded).reshape(-1)
        return loaded
    return default_array.copy()


def all_points_are_zero(points):
    return np.allclose(points, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Calibration de l'emetteur a partir du PDF R13.")
    parser.add_argument(
        "--mr-path",
        default=str(STORAGE_DIR / "MR.txt"),
        help="Chemin vers la matrice de projection du recepteur.",
    )
    parser.add_argument(
        "--receiver-points-path",
        default=str(DEFAULT_RECEIVER_POINTS_PATH),
        help="Chemin vers les points mRj mesures experimentalement (24 lignes, colonnes: uR vR).",
    )
    parser.add_argument(
        "--receiver-points-z0-path",
        default=str(DEFAULT_RECEIVER_POINTS_Z0_PATH),
        help="Chemin vers les 12 points mRj pour le plan Z = 0 mm.",
    )
    parser.add_argument(
        "--receiver-points-z100-path",
        default=str(DEFAULT_RECEIVER_POINTS_Z100_PATH),
        help="Chemin vers les 12 points mRj pour le plan Z = 100 mm.",
    )
    parser.add_argument(
        "--emitter-points-path",
        default=str(DEFAULT_EMITTER_POINTS_PATH),
        help="Chemin vers les points mEj du LCD (24 lignes, colonnes: uE vE).",
    )
    parser.add_argument(
        "--depths-path",
        default=str(DEFAULT_DEPTHS_PATH),
        help="Chemin vers les profondeurs Zj associees aux points (24 valeurs).",
    )
    parser.add_argument(
        "--t3-emetteur",
        type=float,
        default=1472.313636613,
        help="Valeur mesuree de t3E pour denormaliser la matrice emetteur.",
    )
    parser.add_argument(
        "--save-norm",
        default=str(STORAGE_DIR / "MEt_norm_calib.txt"),
        help="Chemin de sortie de la matrice normalisee calculee.",
    )
    parser.add_argument(
        "--save-full",
        default=str(STORAGE_DIR / "MEt_calib.txt"),
        help="Chemin de sortie de la matrice denormalisee calculee.",
    )
    parser.add_argument(
        "--save-points",
        default=str(STORAGE_DIR / "points_objet_calib_emetteur.txt"),
        help="Chemin de sortie des points objet reconstruits.",
    )
    parser.add_argument(
        "--write-templates",
        action="store_true",
        help="Ecrit les gabarits de points experimentaux dans stockage/ puis quitte.",
    )
    args = parser.parse_args()

    if args.write_templates:
        save_array(DEFAULT_EMITTER_POINTS_PATH, EMITTER_IMAGE_POINTS)
        save_array(DEFAULT_RECEIVER_POINTS_PATH, RECEIVER_IMAGE_POINTS)
        save_array(DEFAULT_DEPTHS_PATH, POINT_DEPTHS.reshape(-1, 1))
        print(f"Gabarit points emetteur ecrit dans {DEFAULT_EMITTER_POINTS_PATH}")
        print(f"Gabarit points recepteur ecrit dans {DEFAULT_RECEIVER_POINTS_PATH}")
        print(f"Gabarit profondeurs ecrit dans {DEFAULT_DEPTHS_PATH}")
        return

    receiver_matrix = np.loadtxt(args.mr_path)
    receiver_matrix_norm = normalize_projection_matrix(receiver_matrix)
    receiver_points_combined = load_points_or_default(
        args.receiver_points_path,
        RECEIVER_IMAGE_POINTS,
        expected_columns=2,
    )
    receiver_points_z0 = load_points_or_default(
        args.receiver_points_z0_path,
        np.zeros((12, 2), dtype=float),
        expected_columns=2,
    )
    receiver_points_z100 = load_points_or_default(
        args.receiver_points_z100_path,
        np.zeros((12, 2), dtype=float),
        expected_columns=2,
    )
    emitter_points = load_points_or_default(
        args.emitter_points_path,
        EMITTER_IMAGE_POINTS,
        expected_columns=2,
    )
    depths = load_depths_or_default(args.depths_path, POINT_DEPTHS)

    if not all_points_are_zero(receiver_points_z0) and not all_points_are_zero(receiver_points_z100):
        receiver_points = np.vstack([receiver_points_z0, receiver_points_z100])
    elif not all_points_are_zero(receiver_points_combined):
        receiver_points = receiver_points_combined
    else:
        receiver_points = RECEIVER_IMAGE_POINTS.copy()

    if not (len(receiver_points) == len(emitter_points) == len(depths)):
        raise ValueError(
            "Les fichiers de points experimentaux doivent decrire le meme nombre de points."
        )

    object_points = reconstruct_object_points(
        receiver_matrix_norm,
        receiver_points,
        depths,
    )
    emitter_matrix_norm = calibrate_emitter_dlt(object_points, emitter_points)
    emitter_matrix = emitter_matrix_norm * args.t3_emetteur
    parameters = extract_intrinsic_extrinsic_parameters(emitter_matrix)

    save_array(Path(args.save_points), object_points)
    save_array(Path(args.save_norm), emitter_matrix_norm)
    save_array(Path(args.save_full), emitter_matrix)

    np.set_printoptions(suppress=True, precision=9)
    print(f"Points recepteur combines: {Path(args.receiver_points_path)}")
    print(f"Points recepteur Z0: {Path(args.receiver_points_z0_path)}")
    print(f"Points recepteur Z100: {Path(args.receiver_points_z100_path)}")
    print(f"Points emetteur utilises: {Path(args.emitter_points_path)}")
    print(f"Profondeurs utilisees: {Path(args.depths_path)}")
    print("Points objet reconstruits :")
    print(object_points)
    print("\nMatrice emetteur normalisee MENmes :")
    print(emitter_matrix_norm)
    print("\nMatrice emetteur denormalisee MEt :")
    print(emitter_matrix)
    print("\nParametres intrinseques/extrinseques :")
    print(f"u0E = {parameters['u0']:.9f}")
    print(f"v0E = {parameters['v0']:.9f}")
    print(f"alpha_uE = {parameters['alpha_u']:.9f}")
    print(f"alpha_vE = {parameters['alpha_v']:.9f}")
    print(f"t1E = {parameters['t1']:.9f}")
    print(f"t2E = {parameters['t2']:.9f}")
    print(f"t3E = {parameters['t3']:.9f}")
    print(f"r1E = {parameters['r1']}")
    print(f"r2E = {parameters['r2']}")
    print(f"r3E = {parameters['r3']}")


if __name__ == "__main__":
    main()
