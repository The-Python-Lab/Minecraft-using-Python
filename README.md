# ⛏️Minecraft-in-Python (Minecraft-Klon)
Willkommen beim Repository meiner Voxel-Engine, einem minimalistischen, aber funktionsfähigen Minecraft-Klon, der vollständig in Python implementiert ist.

Dieses Projekt konzentriert sich auf die effiziente, parallele Generierung und das Echtzeit-Rendering einer unendlichen 3D-Welt.

✨ Features (Aktueller Stand)

Derzeit verfügt das Projekt über die folgenden Kernfunktionen:

Unendliche Prozedurale Weltgenerierung:

Verwendet Perlin-Noise zur Generierung von realistisch anmutendem Terrain (Berge, Ebenen).

Implementiert eine parallele Chunkerzeugung, um die Welt asynchron im Hintergrund zu laden.

Chunk-Management:

Dynamisches Laden und Entladen von Chunks basierend auf der Spielerposition (RENDER_DISTANCE).

Verwendet eine einfache Face Culling-Technik (Greedy Meshing ist in Vorbereitung), um unsichtbare Flächen zu eliminieren und die Framerate zu optimieren.

Physik & Interaktion:

Volle 3D-Kollisionserkennung, die es dem Spieler ermöglicht, sich flüssig im Terrain zu bewegen, zu springen und zu fallen (Schwerkraft).

Implementierung einer First-Person-View (FPV)-Kamera mit Maussteuerung.

Grafik-Pipeline (OpenGL):

Verwendet PyOpenGL und GLFW für die plattformunabhängige 3D-Darstellung.

Texturiertes Rendering mit Array-Texturen (Texture Atlasing ist für zukünftige Optimierungen geplant).

Blöcke: Unterstützt verschiedene Blocktypen wie Gras, Erde, Stein, Eichenholz und Blätter.

🛠️ Verwendete Technologien

Python 3.x

PyOpenGL / GLFW: Für Grafik-Rendering und Fensterverwaltung.

PyRR: Für Vektor-, Matrix- und Quaternion-Operationen (Kamera, View-Matrix).

NumPy: Für effizientes Arbeiten mit großen Chunk-Daten-Arrays.

python-noise: Für die prozedurale Generierung des Terrains.

PIL (Pillow): Zum Laden von Texturdateien.

concurrent.futures: Für das Threading zur asynchronen Weltgenerierung.

🚀 Installation & Start

Um die Engine lokal auszuführen, folgen Sie diesen Schritten:

Repository klonen
git clone [https://github.com/Ihr-Github-Name/Python-Voxel-Engine.git](https://github.com/Ihr-Github-Name/Python-Voxel-Engine.git

cd Python-Voxel-Engine

Umgebung einrichten
Es wird dringend empfohlen, eine virtuelle Umgebung zu verwenden:

python3 -m venv .venv source .venv/bin/activate # Unter Windows: .venv\Scripts\activate

Abhängigkeiten installieren
Installieren Sie alle benötigten Bibliotheken:

pip install -r requirements.txt

(Hinweis: Stellen Sie sicher, dass eine requirements.txt mit allen Abhängigkeiten (glfw, PyOpenGL, numpy, pyrr, noise, pillow) im Wurzelverzeichnis vorhanden ist.)

Ausführen
Starten Sie das Hauptskript:

python main.py

🎮 Steuerung

Taste

Aktion

W, A, S, D

Bewegung (Vorwärts, Links, Rückwärts, Rechts)

Leertaste

Springen

Maus

Kamera drehen / Blickrichtung ändern

ESC

Programm beenden

🚧 Zukünftige Pläne

Crafting

Einfache Beleuchtung: Hinzufügen einer rudimentären Beleuchtung (Ambient Occlusion/Sonnenschatten).

🤝 Mitwirken

Dieses Projekt ist Open Source und freut sich über Beiträge! Bei Fragen, Fehlerberichten oder Feature-Vorschlägen öffnen Sie bitte ein Issue oder senden Sie einen Pull Request.
