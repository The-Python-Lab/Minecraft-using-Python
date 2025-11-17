# --- src/chunk_mesh.py (KORRIGIERT FÜR V7) ---
import numpy as np
import concurrent.futures

from .geometry_constants import (
    CUBE_VERTICES, CUBE_UVS, CUBE_NORMALS
)

from .chunk_data import (
    CHUNK_SIZE, MAX_HEIGHT, RENDER_DISTANCE_CHUNKS, ID_AIR,
    generate_chunk_block_data  # Für den Worker
)

from .greedy_mesh import generate_face_culling_mesh_v6  # V7 statt v6!

# --- Worker-Wrapper (Threading) ---

def block_data_worker_wrapper(cx, cz):
    """Wrapper für die Blockdaten-Generierung im Thread-Pool."""
    try:
        return generate_chunk_block_data(cx, cz)
    except Exception as e:
        return Exception(f"Fehler in BlockData-Worker für ({cx},{cz}): {e}")


def mesh_worker_wrapper(cx, cz, block_data, light_map):
    """Wrapper für die Mesh-Generierung im Thread-Pool."""
    try:
        # WICHTIG: Nutze v7 mit Flat Lighting!
        return generate_face_culling_mesh_v6(cx, cz, block_data, light_map)
    except Exception as e:
        return Exception(f"Fehler in Mesh-Worker für ({cx},{cz}): {e}")
