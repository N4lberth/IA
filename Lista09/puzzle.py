import random
import time
from collections import deque
import heapq

# Gerar estado aleatório para o Puzzle 8
def generate_puzzle():
    puzzle = list(range(9))
    random.shuffle(puzzle)
    return tuple(puzzle)

# Movimentos válidos
moves = {'up': -3, 'down': 3, 'left': -1, 'right': 1}

def get_neighbors(state):
    neighbors = []
    zero_index = state.index(0)
    row, col = divmod(zero_index, 3)
    for move, delta in moves.items():
        new_index = zero_index + delta
        if move == 'up' and row == 0: continue
        if move == 'down' and row == 2: continue
        if move == 'left' and col == 0: continue
        if move == 'right' and col == 2: continue
        new_state = list(state)
        new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
        neighbors.append(tuple(new_state))
    return neighbors

# Busca em Largura (BFS)
def bfs(start, goal):
    visited = set()
    queue = deque([(start, [])])
    visited_nodes = 0
    start_time = time.time()
    while queue:
        state, path = queue.popleft()
        visited_nodes += 1
        if state == goal:
            return path + [state], time.time() - start_time, visited_nodes
        if state in visited:
            continue
        visited.add(state)
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                queue.append((neighbor, path + [state]))
    return None, time.time() - start_time, visited_nodes

# Busca em Profundidade (DFS)
def dfs(start, goal, depth_limit=30):
    visited = set()
    stack = [(start, [], 0)]
    visited_nodes = 0
    start_time = time.time()
    while stack:
        state, path, depth = stack.pop()
        visited_nodes += 1
        if state == goal:
            return path + [state], time.time() - start_time, visited_nodes
        if state in visited or depth > depth_limit:
            continue
        visited.add(state)
        for neighbor in get_neighbors(state):
            stack.append((neighbor, path + [state], depth + 1))
    return None, time.time() - start_time, visited_nodes

# Heurísticas para A*
def h_misplaced(state, goal):
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != goal[i])

def h_manhattan(state, goal):
    distance = 0
    for i in range(1, 9):
        xi, yi = divmod(state.index(i), 3)
        xg, yg = divmod(goal.index(i), 3)
        distance += abs(xi - xg) + abs(yi - yg)
    return distance

# A* com heurística variável
def astar(start, goal, heuristic):
    visited = set()
    heap = []
    heapq.heappush(heap, (heuristic(start, goal), 0, start, []))
    visited_nodes = 0
    start_time = time.time()
    while heap:
        est_total_cost, cost, state, path = heapq.heappop(heap)
        visited_nodes += 1
        if state == goal:
            return path + [state], time.time() - start_time, visited_nodes
        if state in visited:
            continue
        visited.add(state)
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                g = cost + 1
                h = heuristic(neighbor, goal)
                heapq.heappush(heap, (g + h, g, neighbor, path + [state]))
    return None, time.time() - start_time, visited_nodes

# Execução
initial_state = (2, 3, 4, 8, 1, 5, 7, 6, 0)
goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)

results = {}
results["BFS"] = bfs(initial_state, goal_state)
results["DFS"] = dfs(initial_state, goal_state)
results["A*_Misplaced"] = astar(initial_state, goal_state, h_misplaced)
results["A*_Manhattan"] = astar(initial_state, goal_state, h_manhattan)

for k, v in results.items():
    print(f"{k}: {'Solução encontrada' if v[0] else 'Falha'}, Tempo: {v[1]:.3f}s, Nós: {v[2]}")