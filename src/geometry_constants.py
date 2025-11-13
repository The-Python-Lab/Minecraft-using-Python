# --- src/geometry_constants.py (FINALE KORREKTUR V5: Linke und Rechte Seite) ---
import numpy as np

# CUBE_VERTICES: (Bleibt unverändert)
CUBE_VERTICES = np.array([
    # 0: Top (+Y)
    [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
    # 1: Bottom (-Y)
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    # 2: Left (-X) - Winding Order: (BL->TL->TR->BR)
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
    # 3: Right (+X) - Winding Order: (BL->BR->TR->TL)
    [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
    # 4: Front (+Z) - CCW ORDER: (BL->TL->TR->BR)
    [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
    # 5: Back (-Z) - OK
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
], dtype=np.float32)

# CUBE_UVS: Linke und Rechte Seite korrigiert
CUBE_UVS = np.array([
    # 0: Top (+Y) - Unverändert
    [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],

    # 1: Bottom (-Y) - Unverändert
    [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],

    # 2: Left (-X) - KORREKTUR: U-Achse gespiegelt, um die Textur auf der linken Seite richtig auszurichten
    [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], # <--- Linke Seite

    # 3: Right (+X) - KORREKTUR: Neu berechnet, um der abweichenden Eckpunktreihenfolge zu entsprechen (BL, BR, TR, TL)
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], # <--- Rechte Seite

    # 4: Front (+Z) - Korrigiert (Keeps)
    [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],

    # 5: Back (-Z) - Standard (Keeps)
    [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
], dtype=np.float32)

CUBE_NORMALS = np.array([
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]
], dtype=np.float32)