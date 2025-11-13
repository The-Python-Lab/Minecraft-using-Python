import numpy as np
import math
import glfw
from pyrr import Matrix44

# Player-Konstanten für AABB und Physik
PLAYER_HEIGHT = 1.7  # Höhe für Durchgang durch 2-Block-Gänge
PLAYER_EYE_HEIGHT = 1.6  # Neue Konstante: Augenhöhe (ca. 94% der Höhe)
PLAYER_WIDTH = 0.6  # Radius 0.3
PLAYER_HALF_WIDTH = PLAYER_WIDTH / 2.0

# --- PHYSIK-KONSTANTEN ---
GRAVITY = 30.0
JUMP_VELOCITY = 10.0
FRICTION = 0.85  # Reibung/Dämpfung auf der horizontalen Ebene (0.0 bis 1.0)
# --- ENDE KONSTANTEN ---

MOVE_SPEED = 15.0  # Geschwindigkeit für horizontale Bewegung
MOUSE_SENSITIVITY = 0.15  # Empfindlichkeit der Maus


class Player:
    def __init__(self, position, yaw, pitch, speed=MOVE_SPEED, rot_speed=120.0):
        # --- Allgemeine Attribute ---
        self.pos = position
        self.yaw = yaw
        self.pitch = pitch
        self.speed = speed
        self.rot_speed = rot_speed
        self.height = PLAYER_HEIGHT
        self.eye_height = PLAYER_EYE_HEIGHT  # Neue Augenhöhe

        # Kamera-Attribute
        self.fovy = 60.0
        self.near = 0.1
        self.far = 1000.0

        self.forward = np.zeros(3, dtype=np.float32)
        self.right = np.zeros(3, dtype=np.float32)

        # --- Kollisions- und Physik-Attribute ---
        self.velocity = np.zeros(3, dtype=np.float32)
        self.target_velocity = np.zeros(3, dtype=np.float32)
        self.on_ground = False
        self.FLYING = False  # Flugmodus deaktiviert, Schwerkraft aktiviert

        # --- Maus-Steuerung ---
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.update_view_vectors()

    def update_view_vectors(self):
        """Berechnet die Vektoren Forward und Right basierend auf Yaw und Pitch."""
        # Horizontal (YAW)
        self.forward[0] = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        # Vertikal (PITCH)
        self.forward[1] = math.sin(math.radians(self.pitch))
        # Z-Achse
        self.forward[2] = -math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        self.forward = self.forward / np.linalg.norm(self.forward)

        # Rechter Vektor (rein horizontal, um Seitwärtsbewegung auf der XZ-Ebene zu halten)
        forward_xz = self.forward.copy()
        forward_xz[1] = 0.0
        if np.linalg.norm(forward_xz) > 0:
            forward_xz = forward_xz / np.linalg.norm(forward_xz)

        self.right = np.cross([0.0, 1.0, 0.0], forward_xz)
        if np.linalg.norm(self.right) > 0:
            self.right = self.right / np.linalg.norm(self.right)

    def handle_mouse_input(self):
        """
        Verarbeitet die von main.py übergebenen Maus-Deltas, um Yaw und Pitch zu aktualisieren.
        """
        if self.mouse_dx != 0.0 or self.mouse_dy != 0.0:
            # 1. Yaw (Horizontal): Standard-Verhalten
            self.yaw += self.mouse_dx * MOUSE_SENSITIVITY

            # 2. Pitch (Vertikal): Standard-FPS-Steuerung (Hochziehen = Hochblicken)
            self.pitch += self.mouse_dy * MOUSE_SENSITIVITY

            # Pitch auf -89 bis 89 Grad begrenzen (Blick nach oben/unten)
            self.pitch = max(-89.0, min(89.0, self.pitch))

            # Setze Deltas zurück, da sie verarbeitet wurden
            self.mouse_dx = 0.0
            self.mouse_dy = 0.0

            self.update_view_vectors()  # Ansichtsvektoren sofort nach Rotation aktualisieren

    def apply_movement_input(self, window, dt):
        """
        Berechnet die gewünschte Ziel-Geschwindigkeit (target_velocity)
        basierend auf den Tasten und den aktuellen Ansichtsvektoren.
        """

        move_speed = self.speed
        target_v = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Vorwärts/Rückwärts (W/S)
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS: target_v += self.forward
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS: target_v -= self.forward

        # Seitwärts (A/D)
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: target_v += self.right
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: target_v -= self.right

        # Vertikale Bewegung: Sprung (nur möglich, wenn auf dem Boden)
        if not self.FLYING:
            if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS and self.on_ground:
                # Springen: Setze vertikale Geschwindigkeit
                self.velocity[1] = JUMP_VELOCITY
                self.on_ground = False  # Ist jetzt in der Luft

            # Vertikale Komponente entfernen, da wir uns nur horizontal bewegen
            target_v[1] = 0.0

            # Nur die horizontale Komponente der Zielgeschwindigkeit normalisieren und skalieren
        horizontal_target_v = target_v[[0, 2]]
        if np.linalg.norm(horizontal_target_v) > 0:
            scale = move_speed / np.linalg.norm(horizontal_target_v)
            target_v[0] = horizontal_target_v[0] * scale
            target_v[2] = horizontal_target_v[1] * scale

        self.target_velocity[:] = target_v

    def get_aabb(self):
        """Gibt die AABB des Spielers als (min_x, min_y, min_z, max_x, max_y, max_z) zurück."""
        min_x = self.pos[0] - PLAYER_HALF_WIDTH
        max_x = self.pos[0] + PLAYER_HALF_WIDTH
        min_y = self.pos[1]
        max_y = self.pos[1] + PLAYER_HEIGHT
        min_z = self.pos[2] - PLAYER_HALF_WIDTH
        max_z = self.pos[2] + PLAYER_HALF_WIDTH
        return np.array([min_x, min_y, min_z, max_x, max_y, max_z], dtype=np.float32)

    def is_block_solid(self, block_x, block_y, block_z, world_data, chunk_size):
        """Prüft, ob der Block an der Weltkoordinate solide ist (ID != ID_AIR)."""
        if block_y < 0 or block_y >= 256: return True

        # 1. Bestimme die Chunk-Koordinaten
        cx = int(np.floor(block_x / chunk_size))
        cz = int(np.floor(block_z / chunk_size))

        # 2. Bestimme die lokalen Block-Koordinaten innerhalb des Chunks
        bx = int(block_x - cx * chunk_size)
        bz = int(block_z - cz * chunk_size)
        by = int(block_y)

        # Sicherheit: Prüfe, ob der Chunk geladen ist
        coord = (cx, cz)
        if coord in world_data:
            block_data = world_data[coord]

            # Koordinaten im Daten-Array (inkl. 1-Block Rand)
            local_x_data = bx + 1
            local_z_data = bz + 1

            # Prüfe, ob die Koordinaten im Puffer liegen (lokale y ist 0 bis MAX_HEIGHT)
            if 0 <= local_x_data < chunk_size + 2 and 0 <= by < block_data.shape[
                1] and 0 <= local_z_data < chunk_size + 2:
                # ID -1.0 ist Luft (ID_AIR)
                return block_data[local_x_data, by, local_z_data] != -1.0

        return False  # Chunk nicht geladen oder außerhalb des gültigen Bereichs

    def check_collisions(self, motion, world_data, chunk_size):
        """Achsen-weise Kollisionsprüfung (AABB-Voxel) für Entlanggleiten."""

        AABB = self.get_aabb()

        min_x_block = math.floor(AABB[0])
        max_x_block = math.ceil(AABB[3])
        min_y_block = math.floor(AABB[1])
        max_y_block = math.ceil(AABB[4])
        min_z_block = math.floor(AABB[2])
        max_z_block = math.ceil(AABB[5])

        on_ground_new = False

        # Iteriere über jede Achse separat (X, Y, Z)
        # Y-Achse zuerst, um Bodenkontakt vor horizontaler Bewegung zu klären
        for axis in [1, 0, 2]:
            if motion[axis] == 0:
                continue

            # Schritt 2a: Führe die Bewegung auf dieser Achse durch
            AABB[axis] += motion[axis]
            AABB[axis + 3] += motion[axis]

            # 2b: Überprüfe auf Block-Kollisionen entlang der Achse

            # Dynamische Grenzen für die Iteration
            if axis == 0:
                block_range_y = range(min_y_block, max_y_block)
                block_range_z = range(min_z_block, max_z_block)
                test_coord_idx = 0 if motion[0] < 0 else 3  # AABB-Index

                for y in block_range_y:
                    for z in block_range_z:
                        test_x = math.floor(AABB[test_coord_idx])
                        if self.is_block_solid(test_x, y, z, world_data, chunk_size):
                            correction = (test_x + 1.0 - AABB[0]) if motion[0] < 0 else (test_x - AABB[3])
                            AABB[0] += correction
                            AABB[3] += correction
                            motion[0] = 0.0
                            break
                    if motion[0] == 0.0: break

            elif axis == 1:
                block_range_x = range(min_x_block, max_x_block)
                block_range_z = range(min_z_block, max_z_block)
                test_coord_idx = 1 if motion[1] < 0 else 4  # AABB-Index

                for x in block_range_x:
                    for z in block_range_z:
                        test_y = math.floor(AABB[test_coord_idx])
                        if self.is_block_solid(x, test_y, z, world_data, chunk_size):
                            if motion[1] < 0:  # Nach unten (Boden)
                                correction = test_y + 1.0 - AABB[1]
                                on_ground_new = True
                            else:  # Nach oben (Decke)
                                correction = test_y - AABB[4]

                            AABB[1] += correction
                            AABB[4] += correction
                            motion[1] = 0.0
                            break
                    if motion[1] == 0.0: break

            elif axis == 2:
                block_range_x = range(min_x_block, max_x_block)
                block_range_y = range(min_y_block, max_y_block)
                test_coord_idx = 2 if motion[2] < 0 else 5  # AABB-Index

                for x in block_range_x:
                    for y in block_range_y:
                        test_z = math.floor(AABB[test_coord_idx])
                        if self.is_block_solid(x, y, test_z, world_data, chunk_size):
                            correction = (test_z + 1.0 - AABB[2]) if motion[2] < 0 else (test_z - AABB[5])
                            AABB[2] += correction
                            AABB[5] += correction
                            motion[2] = 0.0
                            break
                    if motion[2] == 0.0: break

        # 3. Aktualisiere die Position und den Bodenstatus
        self.pos[0] = (AABB[0] + AABB[3]) / 2.0
        self.pos[1] = AABB[1]  # pos ist die untere Mitte (Füße)
        self.pos[2] = (AABB[2] + AABB[5]) / 2.0
        self.on_ground = on_ground_new

        return motion

    def apply_physics(self, dt, world_data, chunk_size):
        """Aktualisiert die Spieler-Geschwindigkeit und Position mit Schwerkraft und Kollision."""

        # 1. Horizontale Bewegung
        # Beschleunigung basierend auf gewünschter Zielgeschwindigkeit
        accel_factor = 0.2  # Kontrolle der Beschleunigung, damit sie sich nicht zu "schwimmend" anfühlt

        # Geschwindigkeits-Update in XZ (Horizontal)
        target_vxz = self.target_velocity[[0, 2]]
        current_vxz = self.velocity[[0, 2]]

        # Beschleunigung (direktes Setzen wäre zu abrupt, Dämpfung ist besser)
        self.velocity[0] += (target_vxz[0] - current_vxz[0]) * accel_factor
        self.velocity[2] += (target_vxz[1] - current_vxz[1]) * accel_factor

        # Reibung/Dämpfung nur auf dem Boden anwenden
        if self.on_ground:
            self.velocity[0] *= FRICTION
            self.velocity[2] *= FRICTION

        # 2. Vertikale Bewegung (Schwerkraft)
        if not self.FLYING:
            self.velocity[1] -= GRAVITY * dt  # Schwerkraft nach unten anwenden

        # 3. Bewegung und Kollision
        motion = self.velocity * dt

        # Führe die Kollisionsprüfung durch (move_and_slide)
        corrected_motion = self.check_collisions(motion, world_data, chunk_size)

        # 4. Aktualisiere die Velocity basierend auf tatsächlicher Bewegung
        if dt > 0:
            # Falls die Bewegung auf einer Achse gestoppt wurde (Kollision), setze Velocity auf 0
            if corrected_motion[0] == 0.0 and abs(self.velocity[0]) > 0.0: self.velocity[0] = 0.0
            if corrected_motion[2] == 0.0 and abs(self.velocity[2]) > 0.0: self.velocity[2] = 0.0

            if corrected_motion[1] == 0.0:
                # Vertikale Geschwindigkeit auf 0 setzen, wenn Boden/Decke getroffen
                self.velocity[1] = 0.0

            # Der Teil der Bewegung, der durchgeführt wurde, beeinflusst die neue Velocity
            # (Dies ist optional, aber nützlich für präzisere Physik-Simulationen)
            # self.velocity = corrected_motion / dt

        # 5. Feinschliff nach Kollision
        if self.on_ground:
            # Wenn auf dem Boden, muss die vertikale Geschwindigkeit 0 sein, um Wackeln zu verhindern
            if self.velocity[1] < 0:
                self.velocity[1] = 0.0

    def get_view_matrix(self):
        """Gibt die View-Matrix zurück (Kamera-Position in AUGENHÖHE)."""
        camera_pos = self.pos.copy()
        # ** KORREKTUR: Kamera auf Augenhöhe setzen **
        camera_pos[1] += self.eye_height

        return Matrix44.look_at(camera_pos, camera_pos + self.forward, [0.0, 1.0, 0.0])

    def raycast_block_selection(self, world_data, chunk_size, max_dist=10.0):
        """Führt einen Raycast durch, um den angezielten Block und die Platzierungsposition zu finden."""
        origin = self.pos.copy()
        origin[1] += self.eye_height  # Auge auf korrigierter Höhe
        direction = self.forward

        current_pos = origin.copy()
        step = 0.1  # Kleine Schrittgröße für Genauigkeit
        steps_max = int(max_dist / step)

        last_block_pos = None  # Speichert die letzte Block-Position, die Luft war

        for _ in range(steps_max):
            current_pos += direction * step

            wx, wy, wz = current_pos

            # Block-Index des aktuellen Blocks
            bx = int(np.floor(wx))
            by = int(np.floor(wy))
            bz = int(np.floor(wz))

            # Prüfe, ob der Block solide ist
            if self.is_block_solid(bx, by, bz, world_data, chunk_size):

                # Konvertiere Weltkoordinaten zurück in Chunk/Block-Koordinaten
                cx = int(np.floor(bx / chunk_size))
                cz = int(np.floor(bz / chunk_size))

                # Lokale Block-Koordinaten (0 bis CHUNK_SIZE-1)
                local_bx = bx - cx * chunk_size
                local_bz = bz - cz * chunk_size

                # Hit Block: Chunk, lokaler Block-Index
                hit_block_info = ((cx, cz), local_bx, by, local_bz)

                # Place Block: Basierend auf dem letzten bekannten LUFT-Block vor dem Treffer
                if last_block_pos is not None:
                    place_bx, place_by, place_bz = last_block_pos
                    place_cx = int(np.floor(place_bx / chunk_size))
                    place_cz = int(np.floor(place_bz / chunk_size))
                    place_local_bx = place_bx - place_cx * chunk_size
                    place_local_bz = place_bz - place_cz * chunk_size

                    place_block_info = ((place_cx, place_cz), place_local_bx, place_by, place_local_bz)
                    return hit_block_info, place_block_info
                else:
                    # Wenn wir sofort einen Block treffen, können wir nicht platzieren
                    return hit_block_info, None

            last_block_pos = (bx, by, bz)

        return None, None