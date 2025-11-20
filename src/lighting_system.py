# --- src/lighting_system.py (LIGHTING-RICHTUNG KORRIGIERT) ---
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


@jit(nopython=True, cache=True)
def calculate_minecraft_vertex_light(light_map, block_data, x, y, z, face_index, vertex_index, channel):
    """
    Samplet Licht von den NACHBAR-Blöcken in Richtung der Face-Normale.
    Die Face zeigt in die Richtung, wo LUFT ist - dort samplen wir das Licht.
    """
    max_height = light_map.shape[1]
    size_x = light_map.shape[0]
    size_z = light_map.shape[2]

    # Bestimme die Face-Normale (wohin die Face zeigt)
    # Das ist die Richtung, in der wir das Licht samplen müssen
    if face_index == 0:  # Top (+Y) - Face zeigt nach oben
        face_normal = (0, 1, 0)
        vertex_positions = [
            (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 0.0)
        ]
    elif face_index == 1:  # Bottom (-Y) - Face zeigt nach unten
        face_normal = (0, -1, 0)
        vertex_positions = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)
        ]
    elif face_index == 2:  # Left (-X) - Face zeigt nach links
        face_normal = (-1, 0, 0)
        vertex_positions = [
            (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)
        ]
    elif face_index == 3:  # Right (+X) - Face zeigt nach rechts
        face_normal = (1, 0, 0)
        vertex_positions = [
            (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 0.0)
        ]
    elif face_index == 4:  # Front (+Z) - Face zeigt nach vorne
        face_normal = (0, 0, 1)
        vertex_positions = [
            (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0)
        ]
    else:  # Back (-Z) - Face zeigt nach hinten
        face_normal = (0, 0, -1)
        vertex_positions = [
            (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)
        ]

    vx, vy, vz = vertex_positions[vertex_index]

    # Berechne die Basis-Position für das Sampling
    # Wir samplen IN RICHTUNG der Normale (wo Luft ist)
    base_x = x + face_normal[0]
    base_y = y + face_normal[1]
    base_z = z + face_normal[2]

    # Berechne die Offset-Richtungen für die tangentialen Achsen
    # (die Achsen parallel zur Face-Ebene)
    tangent_x = 0
    tangent_y = 0
    tangent_z = 0
    bitangent_x = 0
    bitangent_y = 0
    bitangent_z = 0

    # FIX: Wir müssen -1 verwenden, wenn die Koordinate 0.0 ist,
    # damit wir den Nachbarblock prüfen und nicht den Luftblock selbst!

    if face_index == 0 or face_index == 1:  # Top/Bottom (XZ-Ebene)
        tangent_x = 1 if vx == 1.0 else -1  # <--- WAR 0
        bitangent_z = 1 if vz == 1.0 else -1  # <--- WAR 0
    elif face_index == 2 or face_index == 3:  # Left/Right (YZ-Ebene)
        tangent_y = 1 if vy == 1.0 else -1  # <--- WAR 0
        bitangent_z = 1 if vz == 1.0 else -1  # <--- WAR 0
    else:  # Front/Back (XY-Ebene)
        tangent_x = 1 if vx == 1.0 else -1  # <--- WAR 0
        bitangent_y = 1 if vy == 1.0 else -1  # <--- WAR 0

    # Die 4 Sampling-Positionen (Minecraft-Algorithmus):
    # 1. Corner (diagonal), 2. Tangent, 3. Bitangent, 4. Base
    offsets = [
        (base_x + tangent_x + bitangent_x, base_y + tangent_y + bitangent_y, base_z + tangent_z + bitangent_z),
        (base_x + tangent_x, base_y + tangent_y, base_z + tangent_z),
        (base_x + bitangent_x, base_y + bitangent_y, base_z + bitangent_z),
        (base_x, base_y, base_z)
    ]

    # Sample Licht von den 4 Positionen
    light_sum = 0.0
    count = 0
    ao_count = 0

    for nx, ny, nz in offsets:
        if 0 <= nx < size_x and 0 <= ny < max_height and 0 <= nz < size_z:
            light_val = light_map[nx, ny, nz, channel]
            light_sum += float(light_val)
            count += 1

            # AO: Zähle solide Blöcke
            block_id = block_data[nx, ny, nz]
            if block_id != -1.0 and block_id != 4.0:  # Nicht Luft, nicht Blätter
                ao_count += 1
        else:
            # Außerhalb = volle Helligkeit
            light_sum += 15.0
            count += 1

    # Berechne Durchschnitt
    avg_light = light_sum / max(count, 1) if count > 0 else 15.0

    # Minecraft-Style AO (Ambient Occlusion)
    if ao_count >= 3:
        ao_factor = 0.6
    elif ao_count == 2:
        ao_factor = 0.75
    elif ao_count == 1:
        ao_factor = 0.9
    else:
        ao_factor = 1.0

    return avg_light * ao_factor
