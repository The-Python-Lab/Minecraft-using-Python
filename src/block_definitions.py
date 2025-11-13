# --- block_definitions.py (FINALE KORREKTUR) ---
import numpy as np

# --- Block-IDs (Globale Konstanten) ---
ID_AIR = -1.0
ID_GRASS = 0.0
ID_DIRT = 1.0
ID_STONE = 2.0
ID_OAK_LOG = 3.0
ID_LEAVES = 4.0


# --- Textur-Index Zuweisung (MUSS mit der Sortierung von TEXTURE_CONFIG übereinstimmen!) ---
TEX_INDEX_GRASS_TOP = 0.0   # Index 0
TEX_INDEX_GRASS_SIDE = 1.0  # Index 1
TEX_INDEX_DIRT = 2.0        # Index 2
TEX_INDEX_STONE = 3.0       # Index 3
TEX_INDEX_LOG_SIDE = 4.0    # Index 4
TEX_INDEX_LOG_TOP = 5.0     # Index 5
TEX_INDEX_LEAVES = 6.0      # Index 6 (Finaler korrekter Index für Blätter!)


# --- Numba-freundliche Konstanten (bleiben gleich) ---
NON_SOLID_BLOCKS_NUMBA = np.array([ID_AIR, ID_LEAVES], dtype=np.float32)

OAK_LOG_TEXTURES = np.array([
    TEX_INDEX_LOG_TOP,   # Oben (+Y)
    TEX_INDEX_LOG_TOP,   # Unten (-Y)
    TEX_INDEX_LOG_SIDE,  # Seiten (bleiben)
    TEX_INDEX_LOG_SIDE,
    TEX_INDEX_LOG_SIDE,
    TEX_INDEX_LOG_SIDE
], dtype=np.float32)

GRASS_TEXTURES = np.array([
    TEX_INDEX_GRASS_TOP,   # Oben (+Y)
    TEX_INDEX_DIRT,        # Unten (-Y)
    TEX_INDEX_GRASS_SIDE,  # Seiten (bleiben)
    TEX_INDEX_GRASS_SIDE,
    TEX_INDEX_GRASS_SIDE,
    TEX_INDEX_GRASS_SIDE
], dtype=np.float32)


# --- Textur-Konfiguration (für den Loader) ---
TEXTURE_CONFIG = {
    TEX_INDEX_GRASS_TOP: "assets/grass.png",        # <-- Bestätigter Pfad
    TEX_INDEX_GRASS_SIDE: "assets/grass_side.png",  # <-- Bestätigter Pfad
    TEX_INDEX_DIRT: "assets/dirt.png",
    TEX_INDEX_STONE: "assets/stone.png",
    TEX_INDEX_LOG_SIDE: "assets/oak_log.png",
    TEX_INDEX_LOG_TOP: "assets/oak_log_top.png",
    TEX_INDEX_LEAVES: "assets/leaves.png",          # <-- Bestätigter Pfad
}

def get_texture_paths():
    sorted_keys = sorted(TEXTURE_CONFIG.keys())
    return [TEXTURE_CONFIG[key] for key in sorted_keys]