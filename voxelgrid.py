import numpy as np
import cv2
import pyvista as pv
from scipy.spatial import Delaunay
import heapq

# Load the depth image
depth_image_path = "./depth-estimation-output.png"  # Update with your image path
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# Downsample the depth image
scale_factor = 0.25
depth_image_downsampled = cv2.resize(depth_image, (0, 0), fx=scale_factor, fy=scale_factor)

# Normalize depth image
depth_image_downsampled = depth_image_downsampled.astype(np.float32)
depth_image_downsampled = depth_image_downsampled / np.max(depth_image_downsampled)

# Define voxel grid size
voxel_grid_size = (depth_image_downsampled.shape[0], depth_image_downsampled.shape[1], 100)

# Initialize voxel occupancy
voxel_occupancy = np.zeros(voxel_grid_size, dtype=bool)

# Populate voxel grid
for i in range(depth_image_downsampled.shape[0]):
    for j in range(depth_image_downsampled.shape[1]):
        depth_value = depth_image_downsampled[i, j]
        z = int(depth_value * (voxel_grid_size[2] - 1))
        voxel_occupancy[i, j, z] = True

# Convert the voxel occupancy to a list of coordinates
voxel_coords = np.argwhere(voxel_occupancy)

# Perform Delaunay triangulation
triangulation = Delaunay(voxel_coords[:, :2])  # Delaunay triangulation for the xy-plane

# Create the mesh using PyVista
mesh = pv.PolyData(voxel_coords)
mesh.faces = np.hstack([[3] + list(face) for face in triangulation.simplices])  # Convert simplices to faces

# Create a dictionary of neighbors for Dijkstra's algorithm
def build_adjacency_list(vertices, faces):
    adjacency_list = {i: set() for i in range(len(vertices))}
    for face in faces:
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                adjacency_list[face[i]].add(face[j])
                adjacency_list[face[j]].add(face[i])
    return adjacency_list

adjacency_list = build_adjacency_list(voxel_coords, triangulation.simplices)

# Dijkstra's algorithm for pathfinding on the mesh
def dijkstra_pathfinding(start_idx, end_idx, adjacency_list, vertices):
    queue = []
    heapq.heappush(queue, (0, start_idx))
    distances = {i: float('inf') for i in adjacency_list}
    distances[start_idx] = 0
    previous_nodes = {i: None for i in adjacency_list}
    
    while queue:
        current_distance, current_vertex = heapq.heappop(queue)
        
        if current_vertex == end_idx:
            path = []
            while previous_nodes[current_vertex] is not None:
                path.append(current_vertex)
                current_vertex = previous_nodes[current_vertex]
            path.append(start_idx)
            path.reverse()
            return path
        
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor in adjacency_list[current_vertex]:
            distance = np.linalg.norm(vertices[neighbor] - vertices[current_vertex])
            new_distance = current_distance + distance
            
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_vertex
                heapq.heappush(queue, (new_distance, neighbor))
    
    return None

# Hardcoded start and end points
start_point = np.array([110, 130, 70])
end_point = np.array([66, 106, 94])

# Find the closest vertices to the start and end points
def find_closest_vertex(point, vertices):
    distances = np.linalg.norm(vertices - point, axis=1)
    return np.argmin(distances)

start_idx = find_closest_vertex(start_point, voxel_coords)
end_idx = find_closest_vertex(end_point, voxel_coords)

# Compute the path using Dijkstra's algorithm
path_indices = dijkstra_pathfinding(start_idx, end_idx, adjacency_list, voxel_coords)

# Convert path indices to coordinates
if path_indices:
    path_coords = voxel_coords[path_indices]
else:
    path_coords = []

# Create a PyVista plotter object
plotter = pv.Plotter()

# Add the voxel mesh to the plotter
plotter.add_mesh(mesh, color='cyan', opacity=0.5)

# Add the path to the plotter if it exists
if len(path_coords) > 1:
    # Create a PolyData object for the path
    path_lines = np.array([[i, i+1] for i in range(len(path_coords) - 1)])
    path_polydata = pv.PolyData(path_coords)
    path_polydata.lines = np.hstack([[2] + line.tolist() for line in path_lines])
    plotter.add_mesh(path_polydata, color='green', line_width=4)

# Visualize start and end points
plotter.add_mesh(pv.Sphere(center=start_point, radius=2), color='red')
plotter.add_mesh(pv.Sphere(center=end_point, radius=2), color='green')

# Function to handle clicks and print voxel coordinates
def on_click(picked_point):
    click_pos = np.array(picked_point)  # Convert to numpy array if needed
    if click_pos is not None:
        # Convert click_pos to voxel coordinates
        voxel_coords = np.floor(click_pos).astype(int)
        if (0 <= voxel_coords[0] < voxel_occupancy.shape[0] and
            0 <= voxel_coords[1] < voxel_occupancy.shape[1] and
            0 <= voxel_coords[2] < voxel_occupancy.shape[2]):
            print(f"Clicked voxel coordinates: {voxel_coords}")
        else:
            print(f"Clicked position {click_pos} is outside the voxel grid.")
        print(f"Clicked position: {click_pos}")

# Enable picking and register the click event handler
plotter.enable_point_picking(callback=on_click)

# Display the plot
plotter.show()
