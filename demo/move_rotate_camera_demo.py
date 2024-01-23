import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
from utils.io import read_pts
from utils.render import init_visualizer, move_camera, rotate_camera_around_object, rotate_camera

def get_args():
    parser = argparse.ArgumentParser(description='Demo of open3d camera moving and rotation.')
    parser.add_argument('--filepath', type=str, help='The path to the point cloud file.')
    parser.add_argument('--data-format', type=str, choices=['xyzirgb', 'xyzrgb'], 
                        help='The data format of the pts file (xyzirgb or xyzrgb).')
    parser.add_argument('--width', type=int, default=1980, help='Width of the window.')
    parser.add_argument('--height', type=int, default=1080, help='Height of the window.')
    parser.add_argument('--point-size', type=int, default=1, help='The visulized size of each point.')
    parser.add_argument('--demo-mode', type=int, 
                        help='0: Move the camera at specific coordinates, 1: Rotate a specific angle around an object, 2: Rotate the camera at a specific angle.')
    parser.add_argument('--axis', type=str, help='Move & rotate axis.')
    parser.add_argument('--save-images', action='store_true', help='Save the rendered images or not.')
    parser.add_argument('--vis-camera', action='store_true', help='Visualize the camera or not.')

    return parser.parse_args()

def main(args):
    filepath, data_format, point_size, width, height, demo_mode, axis, save_images, vis_camera = \
        args.filepath, args.data_format, args.point_size, args.width, args.height, args.demo_mode, \
        args.axis, args.save_images, args.vis_camera
    
    # Read point cloud data from file.
    pcd = read_pts(filepath, data_format=data_format)

    # Initialize the point cloud data visualizer.
    vis = init_visualizer(point_cloud_data=pcd, 
                          window_name=filepath, 
                          point_size=point_size,
                          width=width,
                          height=height)
    
    # Get original parameters of render camera.
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    ref_extrinsic = params.extrinsic.copy()
    
    if demo_mode == 0:
        # Place the camera at specific coordinates
        if save_images:
            save_dir = os.path.join('data', 'move_{}'.format(axis)) 
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        axis_idxs = {'x': 0, 'y': 1, 'z': 2}
        for i in range(10):
            disp = [0, 0, 0]
            disp[axis_idxs[axis]] = 1
            save_path = os.path.join(save_dir, '{}.png'.format(i)) if save_images else None
            move_camera(vis, params, disp=disp, save_path=save_path, vis_camera=vis_camera, 
                        width=width, height=height)
    elif demo_mode == 1:
        # Rotate a specific angle around an object
        if save_images:
            save_dir = os.path.join('data', 'rotate_o_{}'.format(axis)) 
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for i in range(0, 360, 30):
            save_path = os.path.join(save_dir, '{}.png'.format(i)) if save_images else None
            rotate_camera_around_object(vis, ref_extrinsic, axis, i, save_path=save_path, vis_camera=vis_camera, 
                                        width=width, height=height)
    elif demo_mode == 2:
        # Rotate the camera at a specific angle
        if save_images:
            save_dir = os.path.join('data', 'rotate_s_{}'.format(axis)) 
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for i in range(0, 360, 30):
            save_path = os.path.join(save_dir, '{}.png'.format(i)) if save_images else None
            rotate_camera(vis, ref_extrinsic, axis, i, save_path=save_path, vis_camera=vis_camera, 
                          width=width, height=height)
            
    # Run the visualizer.
    vis.run()

    # Destroy the visualization window
    vis.destroy_window()
            
if __name__=='__main__':
    args = get_args()
    main(args)