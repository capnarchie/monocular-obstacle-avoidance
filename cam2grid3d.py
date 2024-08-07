import cv2
import torch
import numpy as np
import pyvista as pv
from torchvision import transforms
from scipy.spatial import Delaunay
import heapq

class DepthEstimator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def predict(self, frame):
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            depth_map = self.model(input_tensor)
        
        depth_map = depth_map.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        return depth_map

def create_voxel_grid(depth_map, scale_factor=0.25, voxel_depth=50):
    # Downsample depth image
    depth_image_downsampled = cv2.resize(depth_map, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Normalize depth image
    depth_image_downsampled = depth_image_downsampled.astype(np.float32)
    depth_image_downsampled = depth_image_downsampled / np.max(depth_image_downsampled)
    
    # Define voxel grid size
    voxel_grid_size = (depth_image_downsampled.shape[0], depth_image_downsampled.shape[1], voxel_depth)
    
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
    
    return voxel_coords

def build_adjacency_list(vertices, faces):
    adjacency_list = {i: set() for i in range(len(vertices))}
    for face in faces:
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                adjacency_list[face[i]].add(face[j])
                adjacency_list[face[j]].add(face[i])
    return adjacency_list

def main():
    depth_estimator = DepthEstimator()
    cap = cv2.VideoCapture(0)
    target_size = (256, 256)  # Resize input to 256x256 for depth estimation

    # Create a PyVista plotter
    plotter = pv.Plotter()
        # Custom mouse interaction
    plotter.enable_trackball_style()  # Enable trackball style as a base
    
    plotter.show(interactive=True, interactive_update=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize frame for depth estimation
        small_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # Get depth map
        depth_map = depth_estimator.predict(small_frame)
        
        # Create voxel grid point cloud
        voxel_coords = create_voxel_grid(depth_map)
        
        # Perform Delaunay triangulation to create a mesh
        triangulation = Delaunay(voxel_coords[:, :2])  # Delaunay triangulation for the xy-plane
        mesh = pv.PolyData(voxel_coords)
        mesh.faces = np.hstack([[3] + list(face) for face in triangulation.simplices])  # Convert simplices to faces

        # Add the mesh to the plotter
        plotter.clear()
        plotter.add_mesh(mesh, color='blue', opacity=0.25)

        # Display the original frame and resized depth image
        cv2.imshow('Original Video Feed', frame)
        depth_image_display = cv2.resize((depth_map * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Depth Image', depth_image_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
