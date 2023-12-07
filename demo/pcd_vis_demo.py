import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
from utils.io import read_pts
from utils.render import init_visualizer, draw_camera

def get_args():
    parser = argparse.ArgumentParser(description='Visualize the point cloud file.')
    parser.add_argument('--filepath', type=str, help='The path to the point cloud file.')
    parser.add_argument('--data-format', type=str, choices=['xyzirgb', 'xyzrgb'], 
                        help='The data format of the pts file (xyzirgb or xyzrgb).')
    parser.add_argument('--point-size', type=int, help='The visulized size of each point.')
    parser.add_argument('--width', type=int, default=1980, help='Width of the window.')
    parser.add_argument('--height', type=int, default=1080, help='Height of the window.')
    parser.add_argument('--vis-camera', action='store_true', help='Visualize the camera or not.')

    return parser.parse_args()

def main(args):
    filepath, data_format, point_size, width, height, vis_camera = \
        args.filepath, args.data_format, args.point_size, args.width, args.height, args.vis_camera

    # Read point cloud data from file.
    pcd = read_pts(filepath, data_format=data_format)

    # Initialize the point cloud data visualizer.
    vis = init_visualizer(point_cloud_data=pcd, 
                          window_name=filepath, 
                          point_size=point_size,
                          width=width,
                          height=height)

    if vis_camera:
        draw_camera(vis, width, height)

    # Run the visualizer.
    vis.run()
    
    # Destroy the visualization window
    vis.destroy_window()

if __name__=='__main__':
    args = get_args()
    main(args)