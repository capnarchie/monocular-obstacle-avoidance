import cv2
import torch
import numpy as np
import open3d as o3d
from torchvision import transforms
from scipy.spatial import Delaunay
import matplotlib.cm as cm

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

def create_voxel_grid(depth_map, scale_factor=0.25, voxel_depth=20):
    depth_image_downsampled = cv2.resize(depth_map, (0, 0), fx=scale_factor, fy=scale_factor)
    depth_image_downsampled = depth_image_downsampled.astype(np.float32)
    depth_image_downsampled = depth_image_downsampled / np.max(depth_image_downsampled)
    
    voxel_grid_size = (depth_image_downsampled.shape[0], depth_image_downsampled.shape[1], voxel_depth)
    voxel_occupancy = np.zeros(voxel_grid_size, dtype=bool)
    
    for i in range(depth_image_downsampled.shape[0]):
        for j in range(depth_image_downsampled.shape[1]):
            depth_value = depth_image_downsampled[i, j]
            z = int(depth_value * (voxel_grid_size[2] - 1))
            voxel_occupancy[i, j, z] = True
    
    voxel_coords = np.argwhere(voxel_occupancy)
    depth_values = depth_image_downsampled[voxel_coords[:, 0], voxel_coords[:, 1]]
    
    return voxel_coords, depth_values

def create_mesh_from_voxel_coords(voxel_coords, depth_values):
    if len(voxel_coords) < 4:
        return o3d.geometry.TriangleMesh()
    
    triangulation = Delaunay(voxel_coords[:, :2])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(voxel_coords)
    mesh.triangles = o3d.utility.Vector3iVector(triangulation.simplices)
    
    # Normalize depth values for color mapping
    normalized_depth = (depth_values - np.min(depth_values)) / (np.max(depth_values) - np.min(depth_values))
    
    # Map normalized depth values to colors
    cmap = cm.get_cmap('plasma')  # You can choose other colormaps like 'viridis', 'inferno', etc.
    colors = cmap(normalized_depth)[:, :3]  # RGB colors (ignore alpha channel)
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    return mesh

def main():
    depth_estimator = DepthEstimator()
    cap = cv2.VideoCapture(0)
    target_size = (256, 256)

    # Create Open3D visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Mesh Viewer')
    
    # Create an initial empty mesh
    mesh = o3d.geometry.TriangleMesh()
    vis.add_geometry(mesh)

    # Access the view control
    ctr = vis.get_view_control()
    
    # Define initial camera parameters
    front = [0, 0, -1]  # Looking towards negative z-axis
    up = [0, 1, 0]  # Up direction

    # Set initial view
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_lookat([0, 0, 0])  # Look at the origin
    ctr.set_zoom(0.8)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip the frame both horizontally and vertically
        frame = cv2.flip(frame, -1)  # -1 means flipping both horizontally and vertically
        
        small_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        depth_map = depth_estimator.predict(small_frame)
        voxel_coords, depth_values = create_voxel_grid(depth_map)
        new_mesh = create_mesh_from_voxel_coords(voxel_coords, depth_values)
        
        if len(voxel_coords) > 0:
            vis.remove_geometry(mesh)
            mesh = new_mesh
            vis.add_geometry(mesh)
            vis.update_geometry(mesh)
        
        # Rotate camera by 180 degrees around the vertical axis (i.e., up direction)
        front_rotated = [-x for x in front]  # Flip the front vector to achieve 180-degree rotation
        ctr.set_front(front_rotated)
        
        vis.poll_events()
        vis.update_renderer()
        
        if not vis.poll_events():
            break
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()

if __name__ == "__main__":
    main()
