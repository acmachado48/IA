import sys
import time
import heapq
import random
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
                             QHBoxLayout, QMessageBox, QGridLayout, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

GRID_SIZE = 3
TILE_SIZE = 100  
IMAGE_PATH = "/Users/anacarolinamachado/iA/IA/PUZZLE/IMG_4052.jpg"
GOAL_STATE = tuple(range(GRID_SIZE * GRID_SIZE))


def cortar_imagem(image_path, tile_size):
    img = Image.open(image_path).resize((tile_size * GRID_SIZE, tile_size * GRID_SIZE))
    tiles = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            box = (j * tile_size, i * tile_size, (j + 1) * tile_size, (i + 1) * tile_size)
            tile = img.crop(box).convert("RGB")
            qim = QImage(tile.tobytes(), tile.size[0], tile.size[1], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qim)
            tiles.append(pixmap)
    tiles[-1] = QPixmap(tile_size, tile_size)
    tiles[-1].fill(Qt.black)
    return tiles


def get_vizinhos(state):
    empty_idx = state.index(GRID_SIZE * GRID_SIZE - 1)
    neighbors = []
    row, col = divmod(empty_idx, GRID_SIZE)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
            new_idx = new_row * GRID_SIZE + new_col
            new_state = list(state)
            new_state[empty_idx], new_state[new_idx] = new_state[new_idx], new_state[empty_idx]
            neighbors.append(tuple(new_state))
    return neighbors


def bfs(start, goal):
    queue = [start]
    parent = {start: None}
    visited = set()
    while queue:
        curr = queue.pop(0)
        if curr == goal:
            break
        visited.add(curr)
        for neigh in get_vizinhos(curr):
            if neigh not in parent:
                parent[neigh] = curr
                queue.append(neigh)
    path = []
    while curr:
        path.insert(0, curr)
        curr = parent[curr]
    return path, len(visited)


def dfs(start, goal):
    stack = [start]
    parent = {start: None}
    visited = set()
    while stack:
        curr = stack.pop()
        if curr == goal:
            break
        visited.add(curr)
        for neigh in reversed(get_vizinhos(curr)):
            if neigh not in parent:
                parent[neigh] = curr
                stack.append(neigh)
    path = []
    while curr:
        path.insert(0, curr)
        curr = parent[curr]
    return path, len(visited)


def manhattan(state):
    dist = 0
    for idx, val in enumerate(state):
        if val == GRID_SIZE * GRID_SIZE - 1:
            continue
        goal_row, goal_col = divmod(val, GRID_SIZE)
        curr_row, curr_col = divmod(idx, GRID_SIZE)
        dist += abs(goal_row - curr_row) + abs(goal_col - curr_col)
    return dist


def misplaced(state):
    return sum(1 for i, val in enumerate(state) if val != i)


def a_star(start, goal, heuristic_func):
    heap = [(heuristic_func(start), 0, start, [])]
    visited = set()
    while heap:
        _, cost, state, path = heapq.heappop(heap)
        if state == goal:
            return path + [state], len(visited)
        visited.add(state)
        for neighbor in get_vizinhos(state):
            if neighbor not in visited:
                g = cost + 1
                h = heuristic_func(neighbor)
                heapq.heappush(heap, (g + h, g, neighbor, path + [state]))
    return [], len(visited)


class PuzzleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slide Puzzle Solver")
        self.state = list(range(GRID_SIZE * GRID_SIZE))
        self.tile_size = TILE_SIZE
        self.tiles = cortar_imagem(IMAGE_PATH, self.tile_size)
        self.init_ui()
        self.embaralhar()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.grid_layout = QGridLayout()
        self.buttons_layout = QVBoxLayout()

        self.tile_labels = [QLabel() for _ in range(GRID_SIZE * GRID_SIZE)]
        for label in self.tile_labels:
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setAlignment(Qt.AlignCenter)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.grid_layout.addWidget(self.tile_labels[i * GRID_SIZE + j], i, j)

        self.btn_bfs = QPushButton("Resolver BFS")
        self.btn_dfs = QPushButton("Resolver DFS")
        self.btn_astar_manhattan = QPushButton("Resolver A* (Manhattan)")
        self.btn_astar_misplaced = QPushButton("Resolver A* (Misplaced)")
        self.btn_reset = QPushButton("Embaralhar novamente")

        for btn in [self.btn_bfs, self.btn_dfs, self.btn_astar_manhattan, self.btn_astar_misplaced, self.btn_reset]:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.buttons_layout.addWidget(btn)

        self.btn_bfs.clicked.connect(lambda: self.resolver("BFS"))
        self.btn_dfs.clicked.connect(lambda: self.resolver("DFS"))
        self.btn_astar_manhattan.clicked.connect(lambda: self.resolver("A*_manhattan"))
        self.btn_astar_misplaced.clicked.connect(lambda: self.resolver("A*_misplaced"))
        self.btn_reset.clicked.connect(self.embaralhar)

        self.layout.addLayout(self.grid_layout)
        self.layout.addLayout(self.buttons_layout)
        self.setLayout(self.layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.atualizar_pecas)

    def render_tiles(self):
        for idx, val in enumerate(self.state):
            scaled = self.tiles[val].scaled(self.tile_size, self.tile_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.tile_labels[idx].setPixmap(scaled)

    def resizeEvent(self, event):
        self.render_tiles()

    def embaralhar(self):
        state = list(range(GRID_SIZE * GRID_SIZE))
        for _ in range(100):
            state = list(random.choice(get_vizinhos(tuple(state))))
        self.state = state
        self.render_tiles()

    def resolver(self, metodo):
        from time import time
        start = tuple(self.state)
        goal = GOAL_STATE
        t0 = time()

        if metodo == "BFS":
            path, visitados = bfs(start, goal)
        elif metodo == "DFS":
            path, visitados = dfs(start, goal)
        elif metodo == "A*_manhattan":
            path, visitados = a_star(start, goal, manhattan)
        else:
            path, visitados = a_star(start, goal, misplaced)

        tempo_execucao = time() - t0
        if not path:
            QMessageBox.warning(self, "Erro", "Não foi possível resolver o puzzle.")
            return

        movimentos = len(path) - 1
        self.caminho_atual = path
        self.index_caminho = 0
        self.resultado_pendente = (metodo, tempo_execucao, visitados, movimentos)
        self.timer.start(100)

    def salvar_resultados(self, metodo, tempo_execucao, visitados, movimentos):
        file_path = "resultados_puzzle.txt"
        with open(file_path, "a") as file:
            file.write(f"Algoritmo: {metodo}\n")
            file.write(f"Tempo: {tempo_execucao:.2f}s\n")
            file.write(f"Nós visitados: {visitados}\n")
            file.write(f"Movimentos: {movimentos}\n\n")

    def atualizar_pecas(self):
        if self.index_caminho < len(self.caminho_atual):
            self.state = list(self.caminho_atual[self.index_caminho])
            self.render_tiles()
            self.index_caminho += 1
        else:
            self.timer.stop()
            if hasattr(self, "resultado_pendente"):
                metodo, tempo_execucao, visitados, movimentos = self.resultado_pendente
                self.salvar_resultados(metodo, tempo_execucao, visitados, movimentos)
                QMessageBox.information(self, "Resolvido",
                    f"Algoritmo: {metodo}\nTempo: {tempo_execucao:.2f}s\nNós visitados: {visitados}\nMovimentos: {movimentos}")
                del self.resultado_pendente


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PuzzleApp()
    window.show()
    sys.exit(app.exec_())
