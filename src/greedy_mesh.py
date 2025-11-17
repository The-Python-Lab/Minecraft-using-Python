# --- src/greedy_mesh.py (MIT BELEUCHTUNG V6) ---
import numpy as np
from numba import jit

from .chunk_data import CHUNK_SIZE, MAX_HEIGHT, ID_AIR
from .geometry_constants import CUBE_VERTICES, CUBE_UVS, CUBE_NORMALS
from .block_definitions import (
    ID_LEAVES, ID_OAK_LOG, ID_GRASS, ID_DIRT, ID_STONE,
    NON_SOLID_BLOCKS_NUMBA, OAK_LOG_TEXTURES, GRASS_TEXTURES,
    TEX_INDEX_LEAVES, TEX_INDEX_DIRT, TEX_INDEX_STONE,
)


@jit(nopython=True, cache=True)
def is_nonsolid(block_id, nonsolid_array):
    """Prüft, ob block_id in NON_SOLID_BLOCKS_NUMBA enthalten ist."""
    for nonsolid_id in nonsolid_array:
        if block_id == nonsolid_id:
            return True
    return False


@jit(nopython=True, cache=True)
def get_vertex_light(light_map, x, y, z, face_index, vertex_index, channel):
    """
    Berechnet Smooth Lighting für einen Vertex.
    Mittelt das Licht der 4 umliegenden Blöcke (Ambient Occlusion Style).
    """
    max_height = light_map.shape[1]
    size_x = light_map.shape[0]
    size_z = light_map.shape[2]

    # Bestimme die Offsets basierend auf Face und Vertex
    # Vereinfachte Version: Nimm den Block selbst + 3 Nachbarn

    light_sum = 0.0
    count = 0

    # Sample vom aktuellen Block
    if 0 <= x < size_x and 0 <= y < max_height and 0 <= z < size_z:
        light_sum += light_map[x, y, z, channel]
        count += 1

    # Sample von Nachbarn basierend auf Face-Normale
    if face_index == 0:  # Top (+Y)
        sample_offsets = [(0, 1, 0), (-1, 1, 0), (0, 1, -1), (-1, 1, -1)]
    elif face_index == 1:  # Bottom (-Y)
        sample_offsets = [(0, -1, 0), (1, -1, 0), (0, -1, 1), (1, -1, 1)]
    elif face_index == 2:  # Left (-X)
        sample_offsets = [(-1, 0, 0), (-1, 1, 0), (-1, 0, 1), (-1, 1, 1)]
    elif face_index == 3:  # Right (+X)
        sample_offsets = [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
    elif face_index == 4:  # Front (+Z)
        sample_offsets = [(0, 0, 1), (-1, 0, 1), (0, 1, 1), (-1, 1, 1)]
    else:  # Back (-Z)
        sample_offsets = [(0, 0, -1), (1, 0, -1), (0, 1, -1), (1, 1, -1)]

    # Nur 1-2 zusätzliche Samples für Performance
    for i in range(min(2, len(sample_offsets))):
        dx, dy, dz = sample_offsets[i]
        nx = x + dx
        ny = y + dy
        nz = z + dz

        if 0 <= nx < size_x and 0 <= ny < max_height and 0 <= nz < size_z:
            light_sum += light_map[nx, ny, nz, channel]
            count += 1

    return light_sum / max(count, 1)


@jit(nopython=True, cache=True)
def generate_face_culling_mesh_v6(cx, cz, block_data, light_map):
    """
    Generiert das Mesh mit Beleuchtung.
    NEU: 7 Floats pro Vertex (x, y, z, u, v, texid, light)
    """
    MAX_FACES = CHUNK_SIZE * CHUNK_SIZE * MAX_HEIGHT * 6
    MAX_VERTS = MAX_FACES * 4 * 7  # 7 statt 6!
    vertices = np.empty(MAX_VERTS, dtype=np.float32)
    indices = np.empty(MAX_FACES * 6, dtype=np.uint32)

    vert_count = 0
    index_count = 0
    index_offset = 0

    base_x = cx * CHUNK_SIZE
    base_z = cz * CHUNK_SIZE

    dx, dy, dz = block_data.shape

    for x in range(1, dx - 1):
        for z in range(1, dz - 1):
            for y in range(dy):
                block_id = block_data[x, y, z]

                if block_id == ID_AIR:
                    continue

                is_current_leaves = (block_id == ID_LEAVES)
                wx = base_x + x - 1
                wz = base_z + z - 1

                for i_face in range(6):
                    nx, ny, nz = CUBE_NORMALS[i_face]

                    neighbor_x = x + int(nx)
                    neighbor_y = y + int(ny)
                    neighbor_z = z + int(nz)

                    is_face_visible = False

                    if neighbor_y < 0 or neighbor_y >= MAX_HEIGHT:
                        is_face_visible = True
                    elif 0 <= neighbor_x < dx and 0 <= neighbor_z < dz:
                        neighbor_id = block_data[neighbor_x, neighbor_y, neighbor_z]
                        is_neighbor_nonsolid = is_nonsolid(neighbor_id, NON_SOLID_BLOCKS_NUMBA)

                        if is_neighbor_nonsolid:
                            is_neighbor_leaves = (neighbor_id == ID_LEAVES)
                            if is_current_leaves and is_neighbor_leaves:
                                pass
                            else:
                                is_face_visible = True

                    if is_face_visible:
                        start_vert_idx = vert_count

                        # Textur-Zuweisung
                        texture_index = -1.0
                        if block_id == ID_GRASS:
                            texture_index = GRASS_TEXTURES[i_face]
                        elif block_id == ID_OAK_LOG:
                            texture_index = OAK_LOG_TEXTURES[i_face]
                        elif block_id == ID_LEAVES:
                            texture_index = TEX_INDEX_LEAVES
                        elif block_id == ID_DIRT:
                            texture_index = TEX_INDEX_DIRT
                        elif block_id == ID_STONE:
                            texture_index = TEX_INDEX_STONE

                        # Für jeden Vertex der Face
                        for i_vert in range(4):
                            vx = CUBE_VERTICES[i_face, i_vert, 0]
                            vy = CUBE_VERTICES[i_face, i_vert, 1]
                            vz = CUBE_VERTICES[i_face, i_vert, 2]

                            uv_u = CUBE_UVS[i_face, i_vert, 0]
                            uv_v = CUBE_UVS[i_face, i_vert, 1]

                            # NEU: Berechne Lichtlevel für diesen Vertex
                            # Kombiniere Sonnen- und Blocklicht (nimm Maximum)
                            sunlight = get_vertex_light(light_map, x, y, z, i_face, i_vert, 0)
                            blocklight = get_vertex_light(light_map, x, y, z, i_face, i_vert, 1)
                            combined_light = max(sunlight, blocklight)

                            # Schreibe Vertex-Daten (7 Floats)
                            vertices[start_vert_idx] = wx + vx
                            vertices[start_vert_idx + 1] = y + vy
                            vertices[start_vert_idx + 2] = wz + vz
                            vertices[start_vert_idx + 3] = uv_u
                            vertices[start_vert_idx + 4] = uv_v
                            vertices[start_vert_idx + 5] = texture_index
                            vertices[start_vert_idx + 6] = combined_light  # NEU!

                            start_vert_idx += 7  # 7 statt 6!

                        vert_count += 28  # 4 Vertices * 7 Floats

                        # Indices
                        indices[index_count] = index_offset
                        indices[index_count + 1] = index_offset + 1
                        indices[index_count + 2] = index_offset + 2
                        indices[index_count + 3] = index_offset + 2
                        indices[index_count + 4] = index_offset + 3
                        indices[index_count + 5] = index_offset

                        index_count += 6
                        index_offset += 4

    return (
        vertices[:vert_count].reshape(-1, 7),  # 7 statt 6!
        indices[:index_count]
    )
