import random
import matplotlib.pyplot as plt
import numpy as np
import heapq
from matplotlib.animation import FuncAnimation
import time


def generate_random_grid(rows, cols):
    total_cells = rows * cols
    num_zero_cells = int(0.25 * total_cells)
    num_hundred_cells = int(0.25 * total_cells)
    num_other_cells = total_cells - num_zero_cells - num_hundred_cells

    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    # Place continuous sequences of 0-value cells (walls)
    continuous_zeros = random.randint(1, min(15, cols))  # Maximum of 15 cells side by side
    zero_start_col = random.randint(0, cols - continuous_zeros)
    zero_start_row = random.randint(0, rows - 1)
    for row in range(zero_start_row, rows):
        for col in range(zero_start_col, zero_start_col + continuous_zeros):
            if col < cols:
                grid[row][col] = 0
    
    # Place continuous sequences of 100-value cells (clear paths)
    continuous_hundreds = random.randint(1, min(15, cols))  # Maximum of 15 cells side by side
    hundred_start_col = random.randint(0, cols - continuous_hundreds)
    hundred_start_row = random.randint(0, rows - 1)
    for row in range(hundred_start_row, rows):
        for col in range(hundred_start_col, hundred_start_col + continuous_hundreds):
            if col < cols:
                grid[row][col] = 100
    
    # Place other value cells
    placed_cells = continuous_zeros + continuous_hundreds
    for _ in range(num_other_cells - placed_cells):
        while True:
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            if grid[row][col] == 0:
                grid[row][col] = random.randint(1, 99)
                break

    return grid


def heuristic_cost(current, goal):
    heuristic_cost = abs(current[0] - goal[0]) + abs(current[1] - goal[1])
    return heuristic_cost

def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {position: float('inf') for position in np.ndindex(rows, cols)}
    g_score[start] = 0
    f_score = {position: float('inf') for position in np.ndindex(rows, cols)}
    f_score[start] = heuristic_cost(start, goal)
    
    # If the goal is lethal, find the nearest nonlethal cell
    if grid[goal] > 80:
        nonlethal_goal = find_nearest_nonlethal(grid, goal)
        if nonlethal_goal is None:
            return None  # No navigable path
        goal = nonlethal_goal
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] <= 80:
                tentative_g_score = g_score[current] + grid[neighbor]
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic_cost(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

def find_nearest_nonlethal(grid, position):
    rows, cols = grid.shape
    queue = [position]
    visited = set(queue)
    
    while queue:
        current = queue.pop(0)
        
        if grid[current] <= 80:
            return current
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] <= 80 and neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    
    return None  # No nonlethal cell found

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# Dijkstra's Algorithm Function
def dijkstra(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost = {position: float('inf') for position in np.ndindex(rows, cols)}
    cost[start] = 0
    
    # If the goal is lethal, find the nearest nonlethal cell
    if grid[goal] > 80:
        nonlethal_goal = find_nearest_nonlethal(grid, goal)
        if nonlethal_goal is None:
            return None  # No navigable path
        goal = nonlethal_goal
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] <= 80:
                tentative_cost = cost[current] + grid[neighbor]
                
                if tentative_cost < cost[neighbor]:
                    came_from[neighbor] = current
                    cost[neighbor] = tentative_cost
                    heapq.heappush(open_set, (cost[neighbor], neighbor))
    
    return None  # No path found


def rewire_nodes(nodes, new_node, neighbors, radius):
    for neighbor in neighbors:
        tentative_cost = nodes[new_node] + np.linalg.norm(np.array(new_node) - np.array(neighbor))
        if tentative_cost < nodes[neighbor]:
            nodes[neighbor] = tentative_cost
            rewire_neighbors = get_neighbors_within_radius(nodes, neighbor, radius)
            rewire_nodes(nodes, neighbor, rewire_neighbors, radius)


def get_neighbors_within_radius(nodes, point, radius):
    return [node for node in nodes if np.linalg.norm(np.array(node) - np.array(point)) <= radius]

def reconstruct_rrt_path(nodes, goal):
    path = [goal]
    current = goal
    while nodes[current] is not None:
        current = nodes[current]
        path.append(current)
    path.reverse()
    return path

def is_path_clear(grid, point1, point2):
    # Check if the path between point1 and point2 is clear (no lethal cells)
    x1, y1 = point1
    x2, y2 = point2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    n = 1 + dx + dy
    x_inc = 1 if x2 > x1 else -1
    y_inc = 1 if y2 > y1 else -1
    error = dx - dy

    for _ in range(n):
        if grid[x][y] > 80:
            return False
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return True

def rrt(grid, start, goal, max_iter=10000, step_size=5):
    rows, cols = grid.shape
    nodes = {start: 0}
    path = []
    
    # Determine the nearest non-lethal cell to the goal
    nonlethal_goal = find_nearest_nonlethal(grid, goal)
    if nonlethal_goal is None:
        print("RRT Pathfinding: No non-lethal goal found.")
        return None
    
    for _ in range(max_iter):
        if random.random() < 0.5:
            rand_point = nonlethal_goal
        else:
            rand_point = (random.randint(0, rows - 1), random.randint(0, cols - 1))
        
        nearest_node = min(nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(rand_point)))
        normalized_direction = np.array(rand_point) - np.array(nearest_node)
        
        norm = np.linalg.norm(normalized_direction)
        if norm != 0:
            normalized_direction = normalized_direction / norm  # Ensure floating-point division
            
            new_point_float = np.array(nearest_node) + normalized_direction * step_size
            new_point_int = tuple(np.clip(new_point_float.astype(int), 0, (rows - 1, cols - 1)))
            
            if is_path_clear(grid, nearest_node, new_point_int):
                nodes[new_point_int] = nodes[nearest_node] + np.linalg.norm(np.array(new_point_int) - np.array(nearest_node))
                path.append(new_point_int)
                
                if np.linalg.norm(np.array(new_point_int) - np.array(nonlethal_goal)) < step_size:
                    return path
        
    return None  # No path found

def rrt_star(grid, start, goal, max_iter=10000, step_size=5, radius=15):
    rows, cols = grid.shape
    nodes = {start: 0}
    path = []
    
    # Determine the nearest non-lethal cell to the goal
    nonlethal_goal = find_nearest_nonlethal(grid, goal)
    if nonlethal_goal is None:
        print("RRT* Pathfinding: No non-lethal goal found.")
        return None
    
    for _ in range(max_iter):
        if random.random() < 0.5:
            rand_point = nonlethal_goal
        else:
            rand_point = (random.randint(0, rows - 1), random.randint(0, cols - 1))
        
        nearest_node = min(nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(rand_point)))
        normalized_direction = np.array(rand_point) - np.array(nearest_node)
        
        norm = np.linalg.norm(normalized_direction)
        if norm != 0:
            normalized_direction = normalized_direction / norm  # Ensure floating-point division
            
            new_point_float = np.array(nearest_node) + normalized_direction * step_size
            new_point_int = tuple(np.clip(new_point_float.astype(int), 0, (rows - 1, cols - 1)))
            
            if is_path_clear(grid, nearest_node, new_point_int):
                neighbors = get_neighbors_within_radius(nodes, new_point_int, radius)
                min_cost_neighbor = min(neighbors, key=lambda neighbor: nodes[neighbor] + np.linalg.norm(np.array(neighbor) - np.array(new_point_int)))
                
                nodes[new_point_int] = nodes[min_cost_neighbor] + np.linalg.norm(np.array(new_point_int) - np.array(min_cost_neighbor))
                rewire_nodes(nodes, new_point_int, neighbors, radius)
                
                path.append(new_point_int)
                
                if np.linalg.norm(np.array(new_point_int) - np.array(nonlethal_goal)) < step_size:
                    return path
        
    return None  # No path found



# def animate_pathfinding(ax1, ax2, grid, start, end):
#     def update(frame):
#         ax1.clear()
#         ax2.clear()
        
#         ax1.imshow(grid, cmap='gray_r', vmin=0, vmax=100, interpolation='nearest')
#         ax2.imshow(grid, cmap='gray_r', vmin=0, vmax=100, interpolation='nearest')
        
#         current_a_star = a_star_path[frame]
#         current_dijkstra = dijkstra_path[frame]
        
#         ax1.plot([col for row, col in a_star_path[:frame + 1]], [row for row, col in a_star_path[:frame + 1]],
#                  color='green', linewidth=2, marker='o')
#         ax2.plot([col for row, col in dijkstra_path[:frame + 1]], [row for row, col in dijkstra_path[:frame + 1]],
#                  color='green', linewidth=2, marker='o')
        
#         ax1.set_title("A* Pathfinding")
#         ax2.set_title("Dijkstra's Pathfinding")
        
#         ax1.text(0.5, -0.15, f"A* Pathfinding Speed: {len(a_star_path)} steps in {a_star_time:.4f} seconds",
#                  size=12, ha="center", transform=ax1.transAxes)
#         ax2.text(0.5, -0.15, f"Dijkstra's Pathfinding Speed: {len(dijkstra_path)} steps in {dijkstra_time:.4f} seconds",
#                  size=12, ha="center", transform=ax2.transAxes)
    
#     start_time = time.time()
#     a_star_path = a_star(grid, start, end)
#     a_star_time = time.time() - start_time
    
#     start_time = time.time()
#     dijkstra_path = dijkstra(grid, start, end)
#     dijkstra_time = time.time() - start_time
    
#     if a_star_path is None:
#         print("A* Pathfinding could not find a path.")
#         return
    
#     if dijkstra_path is None:
#         print("Dijkstra's Pathfinding could not find a path.")
#         return
    
#     ani = FuncAnimation(fig, update, frames=min(len(a_star_path), len(dijkstra_path)), repeat=False)
#     plt.show()

# # Example usage
# rows = 50
# cols = 50
# start = (0, 0)
# end = (rows - 1, cols - 1)
# grid = generate_random_grid(rows, cols)
# grid_array = np.array(grid)

# # Create a figure for side-by-side animation with pathfinding speeds
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# # Animate A* and Dijkstra's Pathfinding simultaneously with pathfinding speeds
# animate_pathfinding(ax1, ax2, grid_array, start, end)

def animate_pathfinding(ax1, ax2, ax3, ax4, grid, start, end):
    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        ax1.imshow(grid, cmap='gray_r', vmin=0, vmax=100, interpolation='nearest')
        ax2.imshow(grid, cmap='gray_r', vmin=0, vmax=100, interpolation='nearest')
        ax3.imshow(grid, cmap='gray_r', vmin=0, vmax=100, interpolation='nearest')
        ax4.imshow(grid, cmap='gray_r', vmin=0, vmax=100, interpolation='nearest')

        current_a_star = a_star_path[min(frame, len(a_star_path) - 1)]
        current_dijkstra = dijkstra_path[min(frame, len(dijkstra_path) - 1)]
        current_rrt = rrt_path[min(frame, len(rrt_path) - 1)]
        current_rrt_star = rrt_star_path[min(frame, len(rrt_star_path) - 1)]

        ax1.plot([col for row, col in a_star_path[:frame + 1]], [row for row, col in a_star_path[:frame + 1]],
                    color='green', linewidth=2, marker='o')
        ax2.plot([col for row, col in dijkstra_path[:frame + 1]], [row for row, col in dijkstra_path[:frame + 1]],
                    color='green', linewidth=2, marker='o')
        ax3.plot([col for row, col in rrt_path[:frame + 1]], [row for row, col in rrt_path[:frame + 1]],
                    color='green', linewidth=2, marker='o')
        ax4.plot([col for row, col in rrt_star_path[:frame + 1]], [row for row, col in rrt_star_path[:frame + 1]],
                    color='green', linewidth=2, marker='o')

        ax1.set_title("A* Pathfinding")
        ax2.set_title("Dijkstra's Pathfinding")
        ax3.set_title("RRT Pathfinding")
        ax4.set_title("RRT* Pathfinding")

        ax1.text(0.5, -0.15, f"A* Pathfinding Speed: {len(a_star_path)} steps in {a_star_time:.4f} seconds",
                    size=12, ha="center", transform=ax1.transAxes)
        ax2.text(0.5, -0.15, f"Dijkstra's Pathfinding Speed: {len(dijkstra_path)} steps in {dijkstra_time:.4f} seconds",
                    size=12, ha="center", transform=ax2.transAxes)
        ax3.text(0.5, -0.15, f"RRT Pathfinding Speed: {len(rrt_path)} steps in {rrt_time:.4f} seconds",
                    size=12, ha="center", transform=ax3.transAxes)
        ax4.text(0.5, -0.15, f"RRT* Pathfinding Speed: {len(rrt_star_path)} steps in {rrt_star_time:.4f} seconds",
                    size=12, ha="center", transform=ax4.transAxes)
    
    start_time = time.time()
    a_star_path = a_star(grid, start, end)
    a_star_time = time.time() - start_time
    
    start_time = time.time()
    dijkstra_path = dijkstra(grid, start, end)
    dijkstra_time = time.time() - start_time
    
    start_time = time.time()
    rrt_path = rrt(grid, start, end)
    rrt_time = time.time() - start_time
    
    start_time = time.time()
    rrt_star_path = rrt_star(grid, start, end)
    rrt_star_time = time.time() - start_time
    
    if a_star_path is None:
        print("A* Pathfinding could not find a path.")
        return
    
    if dijkstra_path is None:
        print("Dijkstra's Pathfinding could not find a path.")
        return
    
    if rrt_path is None:
        print("RRT Pathfinding could not find a path.")
        return
    
    if rrt_star_path is None:
        print("RRT* Pathfinding could not find a path.")
        return
    
    ani = FuncAnimation(fig, update, frames=min(len(a_star_path), len(dijkstra_path), len(rrt_path), len(rrt_star_path)), repeat=False)
    plt.show()

# Example usage
rows = 100
cols = 100
start = (0, 0)
end = (rows - 1, cols - 1)
grid = generate_random_grid(rows, cols)
grid_array = np.array(grid)

# Create a figure for side-by-side animation with pathfinding speeds
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))

# Animate A*, Dijkstra's, RRT, and RRT* Pathfinding simultaneously with pathfinding speeds
animate_pathfinding(ax1, ax2, ax3, ax4, grid_array, start, end)
