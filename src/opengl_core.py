# --- src/opengl_core.py (MIT BELEUCHTUNG) ---
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from PIL import Image
import ctypes
import numpy as np
from pyrr import Matrix44

from .block_definitions import get_texture_paths

# --- AKTUALISIERTE SHADER MIT BELEUCHTUNG ---
VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texcoord;
layout(location = 2) in float a_texid;
layout(location = 3) in float a_light;  // NEU: Lichtlevel (0-15)

out vec2 v_texcoord;
flat out int v_texid;
out float v_light;  // NEU

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_texcoord = a_texcoord;
    v_texid = int(round(a_texid));
    v_light = a_light / 15.0;  // Normalisiere auf 0.0 - 1.0
}
"""

FRAGMENT_SRC = """
#version 330 core
in vec2 v_texcoord;
flat in int v_texid;
in float v_light;  // NEU

out vec4 out_color;

uniform sampler2D textures[7];
uniform float ambientLight;  // NEU: Minimales Umgebungslicht

void main() {
    if (v_texid < 0) {
        out_color = vec4(1.0, 0.0, 1.0, 1.0);
        return;
    }

    vec4 texColor;
    if (v_texid >= 0 && v_texid < 7) {
        texColor = texture(textures[v_texid], v_texcoord);
    } else {
        texColor = vec4(1.0, 0.0, 1.0, 1.0);
    }

    // Beleuchtungsberechnung
    float minLight = ambientLight;  // Minimales Licht (z.B. 0.05 = 5%)
    float finalLight = mix(minLight, 1.0, v_light);

    out_color = vec4(texColor.rgb * finalLight, texColor.a);
}
"""


def init_window(width, height, title):
    """Initialisiert GLFW, erstellt das Fenster und kompiliert die Shader."""
    if not glfw.init():
        raise Exception("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise Exception("Failed to create window")

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    vao_placeholder = glGenVertexArrays(1)
    glBindVertexArray(vao_placeholder)

    shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(VERTEX_SRC, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )
    glUseProgram(shader)

    # Setze Ambient Light Uniform
    ambient_loc = glGetUniformLocation(shader, "ambientLight")
    if ambient_loc != -1:
        glUniform1f(ambient_loc, 0.05)  # 5% minimales Licht

    return window, shader


def load_texture(path):
    """Lädt eine Textur und gibt die OpenGL-ID zurück."""
    try:
        img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    except FileNotFoundError:
        print(f"FEHLER: Texturdatei nicht gefunden: {path}")
        return None

    img_data = img.convert("RGBA").tobytes()
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    return tex


def setup_textures(shader):
    """Lädt Texturen und setzt die Uniform-Werte."""
    texture_paths = get_texture_paths()
    textures = []

    for i, path in enumerate(texture_paths):
        tex_id = load_texture(path)
        if tex_id is None:
            raise Exception("Texturen konnten nicht vollständig geladen werden.")

        textures.append(tex_id)
        loc = glGetUniformLocation(shader, f"textures[{i}]")
        if loc != -1:
            glUniform1i(loc, i)

    return textures


def create_chunk_buffers_from_data(verts, inds):
    """
    Erstellt VAO/VBO/EBO aus fertigen Mesh-Daten.
    NEU: Unterstützt 7 Floats pro Vertex (Position, UV, TexID, Light)
    """
    if inds.size == 0:
        return None, 0, None, None

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, inds.nbytes, inds, GL_STATIC_DRAW)

    stride = 7 * verts.itemsize  # NEU: 7 Floats statt 6

    # Position (3 floats)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # UV (2 floats)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    # Texture ID (1 float)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(5 * 4))
    glEnableVertexAttribArray(2)

    # Light Level (1 float) - NEU
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
    glEnableVertexAttribArray(3)

    glBindVertexArray(0)
    return vao, inds.size, vbo, ebo


def delete_chunk_buffers(vao, vbo, ebo):
    """Löscht OpenGL-Ressourcen eines Chunks."""
    if vao is not None:
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])
        glDeleteBuffers(1, [ebo])
