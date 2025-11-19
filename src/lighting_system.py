# --- src/lighting_system.py (MINECRAFT-GENAU MIT AO) ---
import numpy as np
from collections import deque
from numba import jit

# Lichtlevel-Konstanten
MAX_LIGHT_LEVEL = 15
MIN_LIGHT_LEVEL = 0

# Separate Kanäle für Sonnen- und Blocklicht
SUNLIGHT_CHANNEL = 0
BLOCKLIGHT_CHANNEL = 1

# Richtungen für Licht-Propagierung (6 Nachbarn)
LIGHT_DIRECTIONS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=np.int32)


class LightingSystem:
    """Verwaltet Sonnen- und Blocklicht für Chunks mit perfekter Chunk-Grenzen-Synchronisation."""

    def __init__(self, chunk_size, max_height):
        self.chunk_size = chunk_size
        self.max_height = max_height
        self.light_data = {}  # {(cx, cz): np.array}
        self.chunks_need_remesh = set()  # Chunks die re-meshed werden müssen

    def init_chunk_lighting(self, coord, block_data):
        """Initialisiert die Beleuchtung für einen neuen Chunk."""
        light_shape = (self.chunk_size + 2, self.max_height, self.chunk_size + 2, 2)
        light_map = np.zeros(light_shape, dtype=np.uint8)

        # Sonnenlicht von oben propagieren
        self._propagate_sunlight_initial(block_data, light_map)

        # Blocklicht von Lichtquellen propagieren
        self._propagate_blocklight_initial(block_data, light_map)

        self.light_data[coord] = light_map
        return light_map

    def _propagate_sunlight_initial(self, block_data, light_map):
        """Propagiert Sonnenlicht von oben nach unten."""
        for x in range(1, self.chunk_size + 1):
            for z in range(1, self.chunk_size + 1):
                current_light = MAX_LIGHT_LEVEL

                for y in range(self.max_height - 1, -1, -1):
                    block_id = block_data[x, y, z]

                    if block_id == -1.0:  # ID_AIR
                        light_map[x, y, z, SUNLIGHT_CHANNEL] = current_light
                    else:
                        if block_id == 4.0:  # ID_LEAVES
                            current_light = max(0, current_light - 1)
                            light_map[x, y, z, SUNLIGHT_CHANNEL] = current_light
                        else:
                            current_light = 0
                            light_map[x, y, z, SUNLIGHT_CHANNEL] = 0

    def _propagate_blocklight_initial(self, block_data, light_map):
        """Propagiert Blocklicht von Lichtquellen mit Flood-Fill."""
        light_sources = []
        # TODO: Hier Lichtquellen-Blöcke finden (z.B. Fackeln)

        for lx, ly, lz, light_level in light_sources:
            self._flood_fill_light(light_map, block_data, lx, ly, lz,
                                   light_level, BLOCKLIGHT_CHANNEL)

    def _flood_fill_light(self, light_map, block_data, start_x, start_y, start_z,
                          light_level, channel):
        """Flood-Fill-Algorithmus für Licht-Propagierung."""
        queue = deque()
        queue.append((start_x, start_y, start_z, light_level))
        visited = set()

        while queue:
            x, y, z, current_light = queue.popleft()

            if (x < 0 or x >= self.chunk_size + 2 or
                    y < 0 or y >= self.max_height or
                    z < 0 or z >= self.chunk_size + 2):
                continue

            if (x, y, z) in visited:
                continue
            visited.add((x, y, z))

            if light_map[x, y, z, channel] < current_light:
                light_map[x, y, z, channel] = current_light

            if current_light <= 0:
                continue

            next_light = current_light - 1

            for dx, dy, dz in LIGHT_DIRECTIONS:
                nx, ny, nz = x + dx, y + dy, z + dz

                if (nx < 0 or nx >= self.chunk_size + 2 or
                        ny < 0 or ny >= self.max_height or
                        nz < 0 or nz >= self.chunk_size + 2):
                    continue

                neighbor_block = block_data[nx, ny, nz]
                if neighbor_block == -1.0 or neighbor_block == 4.0:
                    if light_map[nx, ny, nz, channel] < next_light:
                        queue.append((nx, ny, nz, next_light))

    def update_light_at_position(self, coord, block_data, x, y, z, old_block_id, new_block_id):
        """Aktualisiert die Beleuchtung nach Block-Änderung."""
        if coord not in self.light_data:
            return

        light_map = self.light_data[coord]
        local_x = x + 1
        local_z = z + 1

        if new_block_id == -1.0 and old_block_id != -1.0:
            self._handle_light_increase(light_map, block_data, local_x, y, local_z)
        elif old_block_id == -1.0 and new_block_id != -1.0:
            self._handle_light_decrease(light_map, block_data, local_x, y, local_z)

        # Markiere diesen und alle Nachbar-Chunks für Re-Mesh
        self.chunks_need_remesh.add(coord)
        cx, cz = coord
        for ncx, ncz in [(cx - 1, cz), (cx + 1, cz), (cx, cz - 1), (cx, cz + 1)]:
            self.chunks_need_remesh.add((ncx, ncz))

    def _handle_light_increase(self, light_map, block_data, x, y, z):
        """Wenn ein Block entfernt wird, propagiere Licht hinein."""
        max_neighbor_sunlight = 0
        max_neighbor_blocklight = 0

        for dx, dy, dz in LIGHT_DIRECTIONS:
            nx, ny, nz = x + dx, y + dy, z + dz

            if (0 < nx < self.chunk_size + 1 and
                    0 <= ny < self.max_height and
                    0 < nz < self.chunk_size + 1):
                max_neighbor_sunlight = max(max_neighbor_sunlight,
                                            light_map[nx, ny, nz, SUNLIGHT_CHANNEL])
                max_neighbor_blocklight = max(max_neighbor_blocklight,
                                              light_map[nx, ny, nz, BLOCKLIGHT_CHANNEL])

        if max_neighbor_sunlight > 1:
            self._flood_fill_light(light_map, block_data, x, y, z,
                                   max_neighbor_sunlight - 1, SUNLIGHT_CHANNEL)

        if max_neighbor_blocklight > 1:
            self._flood_fill_light(light_map, block_data, x, y, z,
                                   max_neighbor_blocklight - 1, BLOCKLIGHT_CHANNEL)

    def _handle_light_decrease(self, light_map, block_data, x, y, z):
        """Wenn ein Block platziert wird, entferne Licht."""
        light_map[x, y, z, SUNLIGHT_CHANNEL] = 0
        light_map[x, y, z, BLOCKLIGHT_CHANNEL] = 0

    def sync_light_padding(self, coord, world_data):
        """
        Synchronisiert das Licht-Padding mit Nachbar-Chunks.
        KRITISCH: Muss nach jedem Chunk-Update aufgerufen werden!
        """
        if coord not in self.light_data:
            return

        cx, cz = coord
        light_map = self.light_data[coord]

        # Synchronisiere mit allen 4 Nachbarn
        neighbors = [
            ((cx - 1, cz), 'left'),
            ((cx + 1, cz), 'right'),
            ((cx, cz - 1), 'back'),
            ((cx, cz + 1), 'front')
        ]

        for neighbor_coord, direction in neighbors:
            if neighbor_coord not in self.light_data:
                continue

            neighbor_light = self.light_data[neighbor_coord]

            # Kopiere Lichtdaten vom Nachbarn in unser Padding
            if direction == 'left':  # Nachbar links (-X)
                light_map[0, :, 1:-1, :] = neighbor_light[self.chunk_size, :, 1:-1, :]

            elif direction == 'right':  # Nachbar rechts (+X)
                light_map[self.chunk_size + 1, :, 1:-1, :] = neighbor_light[1, :, 1:-1, :]

            elif direction == 'back':  # Nachbar hinten (-Z)
                light_map[1:-1, :, 0, :] = neighbor_light[1:-1, :, self.chunk_size, :]

            elif direction == 'front':  # Nachbar vorne (+Z)
                light_map[1:-1, :, self.chunk_size + 1, :] = neighbor_light[1:-1, :, 1, :]

    def sync_all_chunk_boundaries(self, world_data):
        """
        Synchronisiert ALLE Chunk-Grenzen im gesamten Licht-System.
        Sollte aufgerufen werden nach großen Updates.
        """
        for coord in list(self.light_data.keys()):
            self.sync_light_padding(coord, world_data)

    def get_chunks_needing_remesh(self):
        """Gibt die Chunks zurück, die re-meshed werden müssen und leert die Liste."""
        chunks = self.chunks_need_remesh.copy()
        self.chunks_need_remesh.clear()
        return chunks

    def get_light_at_position(self, coord, x, y, z):
        """Gibt das Lichtlevel an einer Position zurück."""
        if coord not in self.light_data:
            return MAX_LIGHT_LEVEL, 0

        light_map = self.light_data[coord]
        local_x = x + 1
        local_z = z + 1

        if (0 < local_x < self.chunk_size + 1 and
                0 <= y < self.max_height and
                0 < local_z < self.chunk_size + 1):
            sunlight = light_map[local_x, y, local_z, SUNLIGHT_CHANNEL]
            blocklight = light_map[local_x, y, local_z, BLOCKLIGHT_CHANNEL]
            return sunlight, blocklight

        return MAX_LIGHT_LEVEL, 0


@jit(nopython=True, cache=True)
def calculate_minecraft_vertex_light(light_map, block_data, x, y, z, face_index, vertex_index, channel):
    """
    Berechnet Minecraft-genaues Smooth Lighting mit Ambient Occlusion.
    KORRIGIERT: Vertex-Reihenfolge stimmt jetzt mit CUBE_VERTICES überein!
    """
    max_height = light_map.shape[1]
    size_x = light_map.shape[0]
    size_z = light_map.shape[2]

    # Vertex-Offsets basierend auf CUBE_VERTICES-Reihenfolge
    # CUBE_VERTICES Format: [x, y, z] wobei Block bei (x, y, z) steht

    if face_index == 0:  # Top (+Y)
        # Vertex 0: [0, 1, 0] = hinten links
        # Vertex 1: [0, 1, 1] = vorne links
        # Vertex 2: [1, 1, 1] = vorne rechts
        # Vertex 3: [1, 1, 0] = hinten rechts
        offsets = [
            [(0, 1, 0), (-1, 1, 0), (0, 1, -1), (-1, 1, -1)],  # hinten links
            [(0, 1, 0), (-1, 1, 0), (0, 1, 1), (-1, 1, 1)],  # vorne links
            [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)],  # vorne rechts
            [(0, 1, 0), (1, 1, 0), (0, 1, -1), (1, 1, -1)]  # hinten rechts
        ]
    elif face_index == 1:  # Bottom (-Y)
        # Vertex 0: [0, 0, 0] = hinten links
        # Vertex 1: [1, 0, 0] = hinten rechts
        # Vertex 2: [1, 0, 1] = vorne rechts
        # Vertex 3: [0, 0, 1] = vorne links
        offsets = [
            [(0, -1, 0), (-1, -1, 0), (0, -1, -1), (-1, -1, -1)],  # hinten links
            [(0, -1, 0), (1, -1, 0), (0, -1, -1), (1, -1, -1)],  # hinten rechts
            [(0, -1, 0), (1, -1, 0), (0, -1, 1), (1, -1, 1)],  # vorne rechts
            [(0, -1, 0), (-1, -1, 0), (0, -1, 1), (-1, -1, 1)]  # vorne links
        ]
    elif face_index == 2:  # Left (-X)
        # Vertex 0: [0, 0, 0] = unten hinten
        # Vertex 1: [0, 1, 0] = oben hinten
        # Vertex 2: [0, 1, 1] = oben vorne
        # Vertex 3: [0, 0, 1] = unten vorne
        offsets = [
            [(-1, 0, 0), (-1, -1, 0), (-1, 0, -1), (-1, -1, -1)],  # unten hinten
            [(-1, 0, 0), (-1, 1, 0), (-1, 0, -1), (-1, 1, -1)],  # oben hinten
            [(-1, 0, 0), (-1, 1, 0), (-1, 0, 1), (-1, 1, 1)],  # oben vorne
            [(-1, 0, 0), (-1, -1, 0), (-1, 0, 1), (-1, -1, 1)]  # unten vorne
        ]
    elif face_index == 3:  # Right (+X)
        # Vertex 0: [1, 0, 0] = unten hinten
        # Vertex 1: [1, 0, 1] = unten vorne
        # Vertex 2: [1, 1, 1] = oben vorne
        # Vertex 3: [1, 1, 0] = oben hinten
        offsets = [
            [(1, 0, 0), (1, -1, 0), (1, 0, -1), (1, -1, -1)],  # unten hinten
            [(1, 0, 0), (1, -1, 0), (1, 0, 1), (1, -1, 1)],  # unten vorne
            [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)],  # oben vorne
            [(1, 0, 0), (1, 1, 0), (1, 0, -1), (1, 1, -1)]  # oben hinten
        ]
    elif face_index == 4:  # Front (+Z)
        # Vertex 0: [0, 0, 1] = unten links
        # Vertex 1: [0, 1, 1] = oben links
        # Vertex 2: [1, 1, 1] = oben rechts
        # Vertex 3: [1, 0, 1] = unten rechts
        offsets = [
            [(0, 0, 1), (-1, 0, 1), (0, -1, 1), (-1, -1, 1)],  # unten links
            [(0, 0, 1), (-1, 0, 1), (0, 1, 1), (-1, 1, 1)],  # oben links
            [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)],  # oben rechts
            [(0, 0, 1), (1, 0, 1), (0, -1, 1), (1, -1, 1)]  # unten rechts
        ]
    else:  # Back (-Z) face_index == 5
        # Vertex 0: [0, 0, 0] = unten links
        # Vertex 1: [0, 1, 0] = oben links
        # Vertex 2: [1, 1, 0] = oben rechts
        # Vertex 3: [1, 0, 0] = unten rechts
        offsets = [
            [(0, 0, -1), (-1, 0, -1), (0, -1, -1), (-1, -1, -1)],  # unten links
            [(0, 0, -1), (-1, 0, -1), (0, 1, -1), (-1, 1, -1)],  # oben links
            [(0, 0, -1), (1, 0, -1), (0, 1, -1), (1, 1, -1)],  # oben rechts
            [(0, 0, -1), (1, 0, -1), (0, -1, -1), (1, -1, -1)]  # unten rechts
        ]

    # Hole die Offsets für diesen Vertex
    vertex_offsets = offsets[vertex_index]

    # Sample Licht von den 4 umliegenden Blöcken
    light_values = []
    ao_count = 0

    for dx, dy, dz in vertex_offsets:
        nx = x + dx
        ny = y + dy
        nz = z + dz

        # Bounds check
        if 0 <= nx < size_x and 0 <= ny < max_height and 0 <= nz < size_z:
            light_val = light_map[nx, ny, nz, channel]
            light_values.append(float(light_val))

            # AO: Prüfe ob Block solid ist
            if nx < size_x and ny < max_height and nz < size_z:
                block_id = block_data[nx, ny, nz]
                if block_id != -1.0 and block_id != 4.0:  # Nicht Luft oder Blätter
                    ao_count += 1
        else:
            light_values.append(15.0)  # Außerhalb = volle Helligkeit

    # Minecraft AO Formula
    # Wenn 2 Seiten blockiert sind, oder 3 Blöcke: Dunkler
    if len(light_values) >= 3:
        side1_blocked = block_data[x + vertex_offsets[1][0],
                                   y + vertex_offsets[1][1],
                                   z + vertex_offsets[1][2]] not in [-1.0, 4.0] if (
                0 <= x + vertex_offsets[1][0] < size_x and
                0 <= y + vertex_offsets[1][1] < max_height and
                0 <= z + vertex_offsets[1][2] < size_z) else False

        side2_blocked = block_data[x + vertex_offsets[2][0],
                                   y + vertex_offsets[2][1],
                                   z + vertex_offsets[2][2]] not in [-1.0, 4.0] if (
                0 <= x + vertex_offsets[2][0] < size_x and
                0 <= y + vertex_offsets[2][1] < max_height and
                0 <= z + vertex_offsets[2][2] < size_z) else False

        # Minecraft AO Berechnung
        if side1_blocked and side2_blocked:
            ao_factor = 0.6  # Dunkle Ecke
        elif ao_count >= 3:
            ao_factor = 0.7
        elif ao_count == 2:
            ao_factor = 0.8
        elif ao_count == 1:
            ao_factor = 0.9
        else:
            ao_factor = 1.0
    else:
        ao_factor = 1.0

    # Durchschnitt des Lichts
    if len(light_values) > 0:
        avg_light = sum(light_values) / len(light_values)
    else:
        avg_light = 15.0

    # Kombiniere Licht mit AO
    return avg_light * ao_factor
