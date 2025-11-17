# --- src/lighting_system.py ---
import numpy as np
from collections import deque
from numba import jit

# Lichtlevel-Konstanten
MAX_LIGHT_LEVEL = 15
MIN_LIGHT_LEVEL = 0

# Separate Kanäle für Sonnen- und Blocklicht
SUNLIGHT_CHANNEL = 0
BLOCKLIGHT_CHANNEL = 1

# Richtungen für Licht-Propagierung (6 Nachbarn: +X, -X, +Y, -Y, +Z, -Z)
LIGHT_DIRECTIONS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=np.int32)


class LightingSystem:
    """Verwaltet Sonnen- und Blocklicht für einen Chunk."""

    def __init__(self, chunk_size, max_height):
        self.chunk_size = chunk_size
        self.max_height = max_height

        # Light-Maps: Shape (CHUNK_SIZE+2, MAX_HEIGHT, CHUNK_SIZE+2, 2)
        # Letzte Dimension: [0] = Sunlight, [1] = Blocklight
        self.light_data = {}  # {(cx, cz): np.array}

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
                # Starte mit maximalem Licht an der Oberfläche
                current_light = MAX_LIGHT_LEVEL

                for y in range(self.max_height - 1, -1, -1):
                    block_id = block_data[x, y, z]

                    # Luft lässt Licht durch
                    if block_id == -1.0:  # ID_AIR
                        light_map[x, y, z, SUNLIGHT_CHANNEL] = current_light
                    else:
                        # Blätter reduzieren Licht um 1
                        if block_id == 4.0:  # ID_LEAVES
                            current_light = max(0, current_light - 1)
                            light_map[x, y, z, SUNLIGHT_CHANNEL] = current_light
                        else:
                            # Solide Blöcke stoppen Sonnenlicht komplett
                            current_light = 0
                            light_map[x, y, z, SUNLIGHT_CHANNEL] = 0

    def _propagate_blocklight_initial(self, block_data, light_map):
        """Propagiert Blocklicht von Lichtquellen mit Flood-Fill."""
        # Sammle alle Lichtquellen (hier als Platzhalter - später Fackeln etc.)
        light_sources = []

        # TODO: Hier Lichtquellen-Blöcke finden
        # Beispiel: Fackeln würden hier erkannt werden

        # Für jede Lichtquelle: Flood-Fill
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

            # Bounds-Check
            if (x < 0 or x >= self.chunk_size + 2 or
                    y < 0 or y >= self.max_height or
                    z < 0 or z >= self.chunk_size + 2):
                continue

            # Bereits besucht?
            if (x, y, z) in visited:
                continue
            visited.add((x, y, z))

            # Aktuelles Licht setzen
            if light_map[x, y, z, channel] < current_light:
                light_map[x, y, z, channel] = current_light

            # Licht auf 0 gefallen?
            if current_light <= 0:
                continue

            # Propagiere zu Nachbarn
            next_light = current_light - 1

            for dx, dy, dz in LIGHT_DIRECTIONS:
                nx, ny, nz = x + dx, y + dy, z + dz

                # Bounds-Check für Nachbar
                if (nx < 0 or nx >= self.chunk_size + 2 or
                        ny < 0 or ny >= self.max_height or
                        nz < 0 or nz >= self.chunk_size + 2):
                    continue

                # Ist der Nachbar transparent?
                neighbor_block = block_data[nx, ny, nz]
                if neighbor_block == -1.0 or neighbor_block == 4.0:  # Luft oder Blätter
                    # Würde das Licht heller machen?
                    if light_map[nx, ny, nz, channel] < next_light:
                        queue.append((nx, ny, nz, next_light))

    def update_light_at_position(self, coord, block_data, x, y, z, old_block_id, new_block_id):
        """Aktualisiert die Beleuchtung nach Block-Änderung."""
        if coord not in self.light_data:
            return

        light_map = self.light_data[coord]
        local_x = x + 1
        local_z = z + 1

        # Block entfernt (wurde zu Luft)?
        if new_block_id == -1.0 and old_block_id != -1.0:
            self._handle_light_increase(light_map, block_data, local_x, y, local_z)

        # Block platziert (war Luft)?
        elif old_block_id == -1.0 and new_block_id != -1.0:
            self._handle_light_decrease(light_map, block_data, local_x, y, local_z)

    def _handle_light_increase(self, light_map, block_data, x, y, z):
        """Wenn ein Block entfernt wird, propagiere Licht hinein."""
        # Finde das höchste Licht der Nachbarn
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

        # Propagiere Licht mit Flood-Fill
        if max_neighbor_sunlight > 1:
            self._flood_fill_light(light_map, block_data, x, y, z,
                                   max_neighbor_sunlight - 1, SUNLIGHT_CHANNEL)

        if max_neighbor_blocklight > 1:
            self._flood_fill_light(light_map, block_data, x, y, z,
                                   max_neighbor_blocklight - 1, BLOCKLIGHT_CHANNEL)

    def _handle_light_decrease(self, light_map, block_data, x, y, z):
        """Wenn ein Block platziert wird, entferne Licht."""
        # Setze Licht auf 0
        old_sunlight = light_map[x, y, z, SUNLIGHT_CHANNEL]
        old_blocklight = light_map[x, y, z, BLOCKLIGHT_CHANNEL]

        light_map[x, y, z, SUNLIGHT_CHANNEL] = 0
        light_map[x, y, z, BLOCKLIGHT_CHANNEL] = 0

        # Entferne Licht mit BFS (komplexer Algorithmus)
        # TODO: Implementiere Light-Removal-BFS
        # Für jetzt: Re-calculate in einem größeren Bereich

    def get_light_at_position(self, coord, x, y, z):
        """Gibt das Lichtlevel an einer Position zurück."""
        if coord not in self.light_data:
            return MAX_LIGHT_LEVEL, 0  # Standard: volles Sonnenlicht

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
def calculate_smooth_lighting(light_map, x, y, z, face_index, channel):
    """
    Berechnet Smooth Lighting für die 4 Vertices einer Face.

    Returns: Array mit 4 Lichtleveln (eins pro Vertex)
    """
    vertex_lights = np.zeros(4, dtype=np.float32)

    # Face-Normale bestimmen
    if face_index == 0:  # Top (+Y)
        # Vertex-Offsets relativ zum Block
        offsets = [
            [(0, 1, 0), (-1, 1, 0), (-1, 1, -1), (0, 1, -1)],  # V0
            [(0, 1, 0), (-1, 1, 0), (-1, 1, 1), (0, 1, 1)],  # V1
            [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],  # V2
            [(0, 1, 0), (1, 1, 0), (1, 1, -1), (0, 1, -1)]  # V3
        ]
    elif face_index == 1:  # Bottom (-Y)
        offsets = [
            [(0, -1, 0), (-1, -1, 0), (-1, -1, -1), (0, -1, -1)],
            [(0, -1, 0), (1, -1, 0), (1, -1, -1), (0, -1, -1)],
            [(0, -1, 0), (1, -1, 0), (1, -1, 1), (0, -1, 1)],
            [(0, -1, 0), (-1, -1, 0), (-1, -1, 1), (0, -1, 1)]
        ]
    elif face_index == 2:  # Left (-X)
        offsets = [
            [(-1, 0, 0), (-1, -1, 0), (-1, -1, -1), (-1, 0, -1)],
            [(-1, 0, 0), (-1, 1, 0), (-1, 1, -1), (-1, 0, -1)],
            [(-1, 0, 0), (-1, 1, 0), (-1, 1, 1), (-1, 0, 1)],
            [(-1, 0, 0), (-1, -1, 0), (-1, -1, 1), (-1, 0, 1)]
        ]
    elif face_index == 3:  # Right (+X)
        offsets = [
            [(1, 0, 0), (1, -1, 0), (1, -1, -1), (1, 0, -1)],
            [(1, 0, 0), (1, -1, 0), (1, -1, 1), (1, 0, 1)],
            [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
            [(1, 0, 0), (1, 1, 0), (1, 1, -1), (1, 0, -1)]
        ]
    elif face_index == 4:  # Front (+Z)
        offsets = [
            [(0, 0, 1), (-1, 0, 1), (-1, -1, 1), (0, -1, 1)],
            [(0, 0, 1), (-1, 0, 1), (-1, 1, 1), (0, 1, 1)],
            [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
            [(0, 0, 1), (1, 0, 1), (1, -1, 1), (0, -1, 1)]
        ]
    else:  # Back (-Z)
        offsets = [
            [(0, 0, -1), (-1, 0, -1), (-1, -1, -1), (0, -1, -1)],
            [(0, 0, -1), (-1, 0, -1), (-1, 1, -1), (0, 1, -1)],
            [(0, 0, -1), (1, 0, -1), (1, 1, -1), (0, 1, -1)],
            [(0, 0, -1), (1, 0, -1), (1, -1, -1), (0, -1, -1)]
        ]

    # Für jeden Vertex: Durchschnitt der umliegenden Blöcke
    max_height = light_map.shape[1]
    size_x = light_map.shape[0]
    size_z = light_map.shape[2]

    for v in range(4):
        light_sum = 0.0
        count = 0

        for dx, dy, dz in offsets[v]:
            nx = x + dx
            ny = y + dy
            nz = z + dz

            if 0 <= nx < size_x and 0 <= ny < max_height and 0 <= nz < size_z:
                light_sum += light_map[nx, ny, nz, channel]
                count += 1

        vertex_lights[v] = light_sum / max(count, 1)

    return vertex_lights