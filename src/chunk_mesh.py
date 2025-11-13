# --- src/chunk_mesh.py (FINALE KORREKTUR) ---
import numpy as np
import concurrent.futures

from .geometry_constants import (
CUBE_VERTICES, CUBE_UVS, CUBE_NORMALS
)

from .chunk_data import (
CHUNK_SIZE, MAX_HEIGHT, RENDER_DISTANCE_CHUNKS, ID_AIR,
# HIER WIRD DIE GEFEHLTE FUNKTION HINZUGEFÜGT
generate_chunk_block_data # <--- NEU: FUNKTION FÜR DEN WORKER
)

from .greedy_mesh import generate_face_culling_mesh_v5

# --- Worker-Wrapper (bleibt hier) ---
# Diese sind für das Threading verantwortlich

def block_data_worker_wrapper(cx, cz):
    """Wrapper für die Blockdaten-Generierung im Thread-Pool."""
    try:
        return generate_chunk_block_data(cx, cz) # <--- HIER SCHLÄGT ES FEHL
    except Exception as e:
        return Exception(f"Fehler in BlockData-Worker für ({cx},{cz}): {e}")
# Rückgabe einer Exception, die im Main-Thread abgefangen wird

def mesh_worker_wrapper(cx, cz, block_data):
    """Wrapper für die Mesh-Generierung im Thread-Pool."""
    try:
    # HIER WIRD DIE NEUE FUNKTION VERWENDET
        return generate_face_culling_mesh_v5(cx, cz, block_data)
    except Exception as e:
        return Exception(f"Fehler in Mesh-Worker für ({cx},{cz}): {e}")