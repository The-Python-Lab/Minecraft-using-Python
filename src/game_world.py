import glfw
from OpenGL.GL import *
import numpy as np
from pyrr import Matrix44, matrix44
import concurrent.futures
import math

# Import der benötigten lokalen Module
from src.chunk_data import CHUNK_SIZE, MAX_HEIGHT, RENDER_DISTANCE_CHUNKS, ID_AIR, ID_GRASS
from src.opengl_core import (
    setup_textures,
    create_chunk_buffers_from_data, delete_chunk_buffers
)
from src.chunk_mesh import block_data_worker_wrapper, mesh_worker_wrapper
from src.player import Player
from src.lighting_system import LightingSystem

# Globale Thread-Konstante
THREAD_POOL_SIZE = 4
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)


class GameWorld:
    def __init__(self, window, shader, width, height):
        # OpenGL und Fenster-Referenzen
        self.window = window
        self.shader = shader
        self.width = width
        self.height = height

        # World-State
        self.chunk_data = {}  # {coord: (vao, count, vbo, ebo)}
        self.world_data = {}  # Blockdaten {coord: np.array}

        # NEU: Beleuchtungssystem
        self.lighting = LightingSystem(CHUNK_SIZE, MAX_HEIGHT)

        # Asynchrone Job-Verwaltung
        self.data_futures = {}
        self.mesh_futures = {}

        # Player Initialisierung
        self.player = Player(
            position=np.array([CHUNK_SIZE / 2.0, MAX_HEIGHT * 2.0, CHUNK_SIZE * 2.5], dtype=np.float32),
            yaw=-90.0, pitch=0.0
        )
        self.selected_block_id = ID_GRASS

        # Maus-Steuerungszustände
        self.mouse_last_x = 0.0
        self.mouse_last_y = 0.0
        self.mouse_first_input = True

        # Shader-Uniforms finden
        self.view_loc = glGetUniformLocation(shader, "view")
        self.proj_loc = glGetUniformLocation(shader, "projection")
        self.model_loc = glGetUniformLocation(shader, "model")

        # NEU: Ambient Light Uniform
        self.ambient_loc = glGetUniformLocation(shader, "ambientLight")

        # Statische Projektionsmatrix setzen
        self.projection = Matrix44.perspective_projection(self.player.fovy, width / height, self.player.near,
                                                          self.player.far)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection.astype('float32'))
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, Matrix44.identity().astype('float32'))

        # Texturen laden
        self.textures = setup_textures(shader)

        # GLFW Callbacks setzen
        self._setup_callbacks()

        # Mauszeiger sofort sperren
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def _setup_callbacks(self):
        """Setzt die GLFW-Callbacks mit Zugriff auf die Instanzmethoden."""

        # Setze einen User-Pointer, um die GameWorld-Instanz in den globalen Callbacks zu finden
        glfw.set_window_user_pointer(self.window, self)

        # Wrapper-Funktionen für Callbacks
        def mouse_callback_wrapper(window, xpos, ypos):
            world = glfw.get_window_user_pointer(window)
            if world: world._handle_mouse_movement(xpos, ypos)

        def key_callback_wrapper(window, key, scancode, action, mods):
            world = glfw.get_window_user_pointer(window)
            if world: world._handle_key_input(key, action)

        def mouse_button_callback_wrapper(window, button, action, mods):
            world = glfw.get_window_user_pointer(window)
            if world: world._handle_mouse_button(button, action)

        def focus_callback_wrapper(window, focused):
            world = glfw.get_window_user_pointer(window)
            if world: world._handle_focus(focused)

        glfw.set_cursor_pos_callback(self.window, mouse_callback_wrapper)
        glfw.set_key_callback(self.window, key_callback_wrapper)
        glfw.set_mouse_button_callback(self.window, mouse_button_callback_wrapper)
        glfw.set_window_focus_callback(self.window, focus_callback_wrapper)

    # ---------------- Instanz-Methoden für Input-Handling ----------------

    def _handle_mouse_movement(self, xpos, ypos):
        """Behandelt Mausbewegung, wenn Cursor gesperrt ist."""
        if glfw.get_input_mode(self.window, glfw.CURSOR) == glfw.CURSOR_DISABLED:
            if self.mouse_first_input:
                self.mouse_last_x = xpos
                self.mouse_last_y = ypos
                self.mouse_first_input = False
                return

            xoffset = xpos - self.mouse_last_x
            yoffset = self.mouse_last_y - ypos

            self.mouse_last_x = xpos
            self.mouse_last_y = ypos

            self.player.mouse_dx = xoffset
            self.player.mouse_dy = yoffset

    def _handle_key_input(self, key, action):
        """Schaltet den Mauszeiger-Modus bei ESC um."""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            current_mode = glfw.get_input_mode(self.window, glfw.CURSOR)

            if current_mode == glfw.CURSOR_DISABLED:
                glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
            else:
                glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
                self.mouse_first_input = True  # Reset nach dem Sperren

    def _handle_mouse_button(self, button, action):
        """Behandelt Block-Interaktion (Zerstören/Platzieren)."""
        if glfw.get_input_mode(self.window, glfw.CURSOR) != glfw.CURSOR_DISABLED:
            return

        if action == glfw.PRESS:
            hit, place = self.player.raycast_block_selection(self.world_data, CHUNK_SIZE, max_dist=8.0)

            if hit is not None:
                (cx, cz), bx, by, bz = hit

                if button == glfw.MOUSE_BUTTON_LEFT:  # Zerstören
                    self._update_world_block((cx, cz), bx, by, bz, ID_AIR)
                elif button == glfw.MOUSE_BUTTON_RIGHT:  # Platzieren
                    if place:
                        (pcx, pcz), pbx, pby, pbz = place
                        player_head_y = math.floor(self.player.pos[1] + self.player.height - 0.1)
                        player_feet_y = math.floor(self.player.pos[1])

                        if not (pby == player_head_y or pby == player_feet_y):
                            self._update_world_block((pcx, pcz), pbx, pby, pbz, self.selected_block_id)

                if button == glfw.MOUSE_BUTTON_MIDDLE:
                    self.selected_block_id = 2.0 if self.selected_block_id == ID_GRASS else ID_GRASS

    def _handle_focus(self, focused):
        """Setzt das Maus-Flag zurück, wenn das Fenster den Fokus verliert/gewinnt."""
        if focused:
            self.mouse_first_input = True

    # ---------------- Chunk/World-Management-Logik ----------------

    def _update_world_block(self, coord, block_x, block_y, block_z, new_id):
        """Aktualisiert Blockdaten und markiert umliegende Chunks zum Re-Meshing."""
        cx, cz = coord
        if coord not in self.world_data: return

        local_x = block_x + 1
        local_z = block_z + 1

        if not (0 < local_x < CHUNK_SIZE + 1 and 0 <= block_y < MAX_HEIGHT and 0 < local_z < CHUNK_SIZE + 1): return

        # NEU: Speichere alte Block-ID für Lighting-Update
        old_id = self.world_data[coord][local_x, block_y, local_z]

        # Update Block-Daten
        self.world_data[coord][local_x, block_y, local_z] = new_id

        # NEU: Update Beleuchtung
        self.lighting.update_light_at_position(coord, self.world_data[coord],
                                               block_x, block_y, block_z,
                                               old_id, new_id)

        chunks_to_remesh = {coord}

        # Aktualisiere PADDING des Nachbarn
        if block_x == 0:
            n_coord = (cx - 1, cz)
            if n_coord in self.world_data:
                self.world_data[n_coord][CHUNK_SIZE + 1, block_y, local_z] = new_id
                chunks_to_remesh.add(n_coord)

        if block_x == CHUNK_SIZE - 1:
            n_coord = (cx + 1, cz)
            if n_coord in self.world_data:
                self.world_data[n_coord][0, block_y, local_z] = new_id
                chunks_to_remesh.add(n_coord)

        if block_z == 0:
            n_coord = (cx, cz - 1)
            if n_coord in self.world_data:
                self.world_data[n_coord][local_x, block_y, CHUNK_SIZE + 1] = new_id
                chunks_to_remesh.add(n_coord)

        if block_z == CHUNK_SIZE - 1:
            n_coord = (cx, cz + 1)
            if n_coord in self.world_data:
                self.world_data[n_coord][local_x, block_y, 0] = new_id
                chunks_to_remesh.add(n_coord)

        for r_cx, r_cz in chunks_to_remesh:
            r_coord = (r_cx, r_cz)
            if r_coord in self.world_data and r_coord not in self.mesh_futures:
                if r_coord in self.chunk_data:
                    vao, _, vbo, ebo = self.chunk_data[r_coord]
                    delete_chunk_buffers(vao, vbo, ebo)
                    del self.chunk_data[r_coord]

                # WICHTIG: Synchronisiere Licht-Padding vor dem Re-Meshing
                self.lighting.sync_light_padding(r_coord, self.world_data)

                # NEU: Hole Light-Map für Re-Meshing
                light_map = self.lighting.light_data.get(r_coord, None)
                if light_map is not None:
                    future = EXECUTOR.submit(mesh_worker_wrapper, r_cx, r_cz,
                                             self.world_data[r_coord], light_map)
                    self.mesh_futures[r_coord] = future

    def _manage_chunks(self):
        """Lädt, generiert und entlädt Chunks basierend auf der Spielerposition."""
        player_chunk_x = int(self.player.pos[0] // CHUNK_SIZE)
        player_chunk_z = int(self.player.pos[2] // CHUNK_SIZE)
        R = RENDER_DISTANCE_CHUNKS

        needed_coords = set()

        # Bestimme benötigte Chunk-Koordinaten
        for cx in range(player_chunk_x - R, player_chunk_x + R + 1):
            for cz in range(player_chunk_z - R, player_chunk_z + R + 1):
                coord = (cx, cz)
                needed_coords.add(coord)

                # 1. Blockdaten-Job starten
                if coord not in self.world_data and coord not in self.data_futures:
                    future = EXECUTOR.submit(block_data_worker_wrapper, cx, cz)
                    self.data_futures[coord] = future

                # 2. Mesh-Job starten, wenn Blockdaten vorhanden sind
                elif coord in self.world_data and coord not in self.chunk_data and coord not in self.mesh_futures:
                    # NEU: Prüfe ob Light-Map existiert, sonst erstelle sie
                    if coord not in self.lighting.light_data:
                        try:
                            self.lighting.init_chunk_lighting(coord, self.world_data[coord])
                        except Exception as e:
                            print(f"Lighting-Init Fehler für {coord}: {e}")
                            continue

                    light_map = self.lighting.light_data.get(coord, None)
                    if light_map is not None:
                        try:
                            future = EXECUTOR.submit(mesh_worker_wrapper, cx, cz,
                                                     self.world_data[coord], light_map)
                            self.mesh_futures[coord] = future
                        except Exception as e:
                            print(f"Mesh-Future Fehler für {coord}: {e}")

        # 3. Sammeln fertiger Futures und speichern der Ergebnisse
        finished_data_futures = []
        for coord, future in self.data_futures.items():
            if future.done():
                finished_data_futures.append(coord)
                try:
                    result = future.result()
                    if isinstance(result, Exception): raise result
                    self.world_data[coord] = result

                    # NEU: Initialisiere Beleuchtung für neuen Chunk
                    self.lighting.init_chunk_lighting(coord, result)

                    # WICHTIG: Synchronisiere Licht-Padding mit Nachbarn
                    self.lighting.sync_light_padding(coord, self.world_data)

                    # Synchronisiere auch die Nachbarn zurück
                    cx, cz = coord
                    for neighbor_coord in [(cx - 1, cz), (cx + 1, cz), (cx, cz - 1), (cx, cz + 1)]:
                        if neighbor_coord in self.lighting.light_data:
                            self.lighting.sync_light_padding(neighbor_coord, self.world_data)

                except Exception as e:
                    print(f"BlockData-Fehler für {coord}: {e}")
        for coord in finished_data_futures: del self.data_futures[coord]

        finished_mesh_futures = []
        for coord, future in self.mesh_futures.items():
            if future.done():
                finished_mesh_futures.append(coord)
                try:
                    result = future.result()
                    if isinstance(result, Exception): raise result
                    verts, inds = result

                    if inds.size > 0:
                        vao, count, vbo, ebo = create_chunk_buffers_from_data(verts, inds)
                        self.chunk_data[coord] = (vao, count, vbo, ebo)
                except Exception as e:
                    print(f"Mesh-Generierungsfehler für {coord}: {e}")
        for coord in finished_mesh_futures: del self.mesh_futures[coord]

        # 4. Veraltete Chunks entladen/löschen
        chunks_to_delete_data = [coord for coord in self.world_data if
                                 coord not in needed_coords and coord not in self.data_futures]
        for coord in chunks_to_delete_data:
            del self.world_data[coord]
            # NEU: Lösche auch Light-Daten
            if coord in self.lighting.light_data:
                del self.lighting.light_data[coord]

        chunks_to_delete_mesh = [coord for coord in self.chunk_data if
                                 coord not in needed_coords and coord not in self.mesh_futures]
        for coord in chunks_to_delete_mesh:
            vao, _, vbo, ebo = self.chunk_data[coord]
            delete_chunk_buffers(vao, vbo, ebo)
            del self.chunk_data[coord]

    # ---------------- Frustum Culling Logik ----------------
    def _extract_frustum_planes(self, view_proj):
        planes = np.zeros((6, 4), dtype=np.float32)
        col3 = np.array(view_proj[:, 3]).reshape(4)
        col0 = np.array(view_proj[:, 0]).reshape(4)
        col1 = np.array(view_proj[:, 1]).reshape(4)
        col2 = np.array(view_proj[:, 2]).reshape(4)

        planes[0] = col3 + col0  # Left
        planes[1] = col3 - col0  # Right
        planes[2] = col3 + col1  # Bottom
        planes[3] = col3 - col1  # Top
        planes[4] = col3 + col2  # Near
        planes[5] = col3 - col2  # Far

        for i in range(6): planes[i, :] /= np.linalg.norm(planes[i, :3])
        return planes

    def _is_chunk_visible(self, planes, cx, cz):
        min_x = cx * CHUNK_SIZE
        max_x = min_x + CHUNK_SIZE
        min_y = 0.0
        max_y = MAX_HEIGHT
        min_z = cz * CHUNK_SIZE
        max_z = min_z + CHUNK_SIZE

        points = [
            [min_x, min_y, min_z], [max_x, min_y, min_z],
            [min_x, max_y, min_z], [max_x, max_y, min_z],
            [min_x, min_y, max_z], [max_x, min_y, max_z],
            [min_x, max_y, max_z], [max_x, max_y, max_z],
        ]

        for i in range(6):
            plane = planes[i]
            outside_count = 0
            for p in points:
                distance = plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3]
                if distance < 0: outside_count += 1
            if outside_count == 8: return False
        return True

    # ---------------- Update/Render-Methoden ----------------

    def update(self, dt):
        """Führt alle Updates pro Frame aus (Physik, Chunk-Management)."""

        self._manage_chunks()

        # Player-Update nur, wenn Cursor gesperrt
        if glfw.get_input_mode(self.window, glfw.CURSOR) == glfw.CURSOR_DISABLED:
            self.player.handle_mouse_input()
            self.player.apply_movement_input(self.window, dt)
        else:
            # Setze Bewegungsvektoren zurück, wenn die Steuerung freigegeben ist
            self.player.mouse_dx = 0.0
            self.player.mouse_dy = 0.0
            self.player.target_velocity[:] = 0.0

        self.player.apply_physics(dt, self.world_data, CHUNK_SIZE)

        # NEU: Optional - Tag/Nacht-Zyklus (auskommentiert)
        # self._update_day_night_cycle()

    def _update_day_night_cycle(self):
        """Optional: Implementiert einen Tag/Nacht-Zyklus."""
        # 2 Minuten für einen kompletten Zyklus
        time_of_day = (glfw.get_time() % 120.0) / 120.0

        # Berechne Ambient Light (0.05 bis 0.5)
        ambient = 0.05 + 0.45 * abs(math.sin(time_of_day * math.pi))

        # Update Shader Uniform
        if self.ambient_loc != -1:
            glUniform1f(self.ambient_loc, ambient)

    def render(self):
        """Rendert die Szene."""

        # 1. Update View Matrix
        view = self.player.get_view_matrix()
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view.astype('float32'))

        # 2. Frustum Culling vorbereiten
        view_proj = self.projection * view
        frustum_planes = self._extract_frustum_planes(view_proj)

        # 3. Hintergrund löschen
        glClearColor(0.53, 0.8, 0.95, 1.0)  # Himmelblau
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 4. Texturen binden
        for i, tex in enumerate(self.textures):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, tex)

        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, Matrix44.identity().astype('float32'))

        # 5. Chunks rendern (mit Frustum Culling)
        for coord, (vao, count, _, _) in self.chunk_data.items():
            if count > 0 and vao is not None:
                if self._is_chunk_visible(frustum_planes, coord[0], coord[1]):
                    glBindVertexArray(vao)
                    glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, None)

    def shutdown(self):
        """Gibt Ressourcen frei."""
        EXECUTOR.shutdown(wait=True)


def run_game(window, shader, width, height):
    """Haupt-Game-Loop-Funktion, die von main.py aufgerufen wird."""
    game_world = GameWorld(window, shader, width, height)

    prev_time = glfw.get_time()

    try:
        while not glfw.window_should_close(window):
            now = glfw.get_time()
            dt = now - prev_time
            prev_time = now
            glfw.poll_events()

            game_world.update(dt)
            game_world.render()

            glfw.swap_buffers(window)
    finally:
        game_world.shutdown()
