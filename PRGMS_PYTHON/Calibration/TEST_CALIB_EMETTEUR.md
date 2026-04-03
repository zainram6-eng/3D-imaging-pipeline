# Test calibration emetteur

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
   `python PRGMS_PYTHON\Calibration\Projeter_image_plein_ecran.py img\Mire_damier.bmp`
3. Positionner une surface plane a `Z = 0 mm`, prendre une photo nette.
4. Positionner la meme surface plane a `Z = 100 mm`, prendre une photo nette.
5. Ouvrir `img/Mire_damier_points.png` et relever dans l'ordre les 12 points visibles sur chaque photo.
6. Remplir `stockage/mR_emetteur_experimental_z0.txt` puis `stockage/mR_emetteur_experimental_z100.txt`.
7. Lancer :
   `python PRGMS_PYTHON\Calibration\Calib_emetteur.py`

## Ordre des points

L'ordre est impose par la numerotation de `img/Mire_damier_points.png`.
Les 12 premiers points correspondent a `Z = 0 mm`.
Les 12 suivants correspondent aux memes points pour `Z = 100 mm`.

## Remarques

- Les fichiers `mR_*` doivent contenir deux colonnes : `uR vR`.
- `Calib_emetteur.py` lit directement les fichiers `z0` et `z100` s'ils sont renseignes.
- Le script de calibration utilise `stockage/MR.txt` pour retroprojeter les points objet.
- Si tu mesures une autre distance que `100 mm`, il faut aussi mettre a jour `stockage/Z_emetteur_experimental.txt`.
