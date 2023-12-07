import open3d as o3d
import numpy as np 
import os
import random

def render_with_object_rotation(point_cloud_data, visualizer, **kwargs):
    """Rotate the point cloud and render.

    Args:
        point_cloud_data (o3d.open3d.geometry.PointCloud): The point cloud data.
        visualizer (o3d.visualization.Visualizer): The point cloud visualizer.
        steps (tuple, optional): The rotation step for each axis 
                                 (If set to -1 then rotation is performed using random steps in that axis). 
                                 Defaults to (-1, -1, -1).
        iters (int, optional): The number of rotation iterations. Defaults to 1.
        save_dir (str, optional): The save directory for rendered images. Defaults to None.
    """    

    assert point_cloud_data and visualizer
    steps = kwargs.get('steps', (-1, -1, -1)) 
    assert min(steps) > -2
    iters = kwargs.get('iters', 1) 
    save_dir = kwargs.get('save_dir', None) 
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate the center point of the point cloud (as the rotation center)
    pcd_center = tuple(np.mean(point_cloud_data.points, axis=0))

    for i in range(iters):
        x_step = steps[0] if steps[0] != -1 else random.randint(1, 5)
        y_step = steps[1] if steps[1] != -1 else random.randint(1, 5)
        z_step = steps[2] if steps[2] != -1 else random.randint(1, 5)
        rotation_matrix = point_cloud_data.get_rotation_matrix_from_xyz((np.pi / 180 * x_step, 
                                                                         np.pi / 180 * y_step, 
                                                                         np.pi / 180 * z_step))
        point_cloud_data.rotate(rotation_matrix, center=pcd_center)
        visualizer.update_geometry(point_cloud_data)
        visualizer.poll_events()
        visualizer.update_renderer()
        if save_dir:
            save_path = os.path.join(save_dir, '{}.jpg'.format(i))
            visualizer.capture_screen_image(save_path)
            print('Rednered image saved to {}'.format(save_path))

def init_visualizer(**kwargs):
    """Initialize the visualizer.

    Args:
        point_cloud_data (o3d.open3d.geometry.PointCloud): The point cloud data.
        window_name (str, optional): The name of the visualization window. Defaults to 'Open3D'.
        point_size (int, optional): The visulized size of each point. Defaults to 1.
        width (int, optional): Width of the window. Defaults to 1920.
        height (int, optional): Height of window. Defaults to 1080.
        left (int, optional): Left margin of the window to the screen. Defaults to 50.
        top (int, optional): Top margin of the window to the screen. Defaults to 50.

    Returns:
        o3d.visualization.Visualizer: The point cloud visualizer.
    """    

    point_cloud_data = kwargs.get('point_cloud_data', None)
    window_name = kwargs.get('window_name', 'Open3D')
    point_size = kwargs.get('point_size', 1)
    width = kwargs.get('width ', 1920)
    height = kwargs.get('height ', 1080)
    left = kwargs.get('left ', 50)
    top = kwargs.get('top ', 50)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name,
                      width=width,
                      height=height,
                      left=left,
                      top=top)

    # Set the visulized size of each point.
    vis.get_render_option().point_size = point_size

    if point_cloud_data:
        # Add point cloud data to the scene.
        vis.add_geometry(point_cloud_data)

    return vis

def downsample_voxel(point_cloud_data, **kwargs):
    """Voxel downsampling.

    Args:
        point_cloud_data (o3d.open3d.geometry.PointCloud): The point cloud data.
        voxel_size (float, optional): Voxel size to downsample into. Defaults to 0.05.
        estimate_norm (bool, optional): Compute normal or not. Defaults to False.
        radius (float, optional): Search radius. Defaults to 0.1.
        max_nn (int, optional): Maximum nearest neighbor. Defaults to 30.
    Returns:
        o3d.open3d.geometry.PointCloud: Downsampled voxel
    """    
    assert point_cloud_data
    voxel_size = kwargs.get('voxel_size', 0.05)
    estimate_norm = kwargs.get('estimate_norm', False)
    
    # Voxel downsampling uses a regular voxel grid to create 
    # a uniformly downsampled point cloud from an input point cloud.
    downpcd = point_cloud_data.voxel_down_sample(voxel_size=voxel_size)

    if estimate_norm:
        # Compute normal for every point
        radius = kwargs.get('radius', 0.1)
        max_nn = kwargs.get('max_nn', 30)
        downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, 
                                                                                    max_nn=max_nn))

    o3d.visualization.draw_geometries([downpcd])

    return downpcd

def draw_camera(visualizer, width, height, **kwargs):
    """Draw render camera.

    Args:
        visualizer (o3d.visualization.Visualizer): The point cloud visualizer.
        width (int): Width of the window. 
        height (int): Height of window. 
        scale (int): Camera model scale. Defaults to 1.
        color (list): Color of the image plane and pyramid lines. Defaults to [0.8, 0.2, 0.8].
    """    

    assert visualizer and width and height
    scale = kwargs.get('scale', 1)
    color = kwargs.get('color', [0.8, 0.2, 0.8])

    # Get intrinsic and extrinsic parameters of render camera.
    intrinsic, extrinsic = get_camera_params(visualizer)

    # convert extrinsics matrix to rotation and translation matrix
    extrinsic = np.linalg.inv(extrinsic)
    rotate_mat = extrinsic[0:3, 0:3]
    trans_mat = extrinsic[0:3, 3]

    geometries = _draw_camera_geometries(intrinsic.intrinsic_matrix, rotate_mat, trans_mat, 
                                         width, height, scale, color)
    for g in geometries:
        visualizer.add_geometry(g)

def get_camera_params(visualizer, copy=False):
    """Get intrinsic and extrinsic parameters of render camera.

    Args:
        visualizer (o3d.visualization.Visualizer): The point cloud visualizer.
        copy (bool, optional): Perform shallow copy operation or not. Defaults to False.

    Returns:
        open3d.camera.PinholeCameraIntrinsic: The intrinsic parameters.
        ndarray: The extrinsic parameters.
    """    

    assert visualizer
    ctr = visualizer.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    intrinsic = params.intrinsic
    extrinsic = params.extrinsic
    return intrinsic.copy(), extrinsic.copy() if copy else intrinsic, extrinsic

def save_view_point(point_cloud_data, visualizer, save_path):
    assert point_cloud_data and visualizer and save_path
    params = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(save_path, params)


def _draw_camera_geometries(intrinsic_mat, rotate_mat, trans_mat, width, height, **kwargs):
    """Create axis, plane and pyramid geometries in Open3D format

    Args:
        intrinsic_mat (_type_): The calibration matrix (camera intrinsics)
        rotate_mat (_type_): The rotation matrix
        trans_mat (_type_): The translation
        width (_type_): image width
        height (_type_): image height
        scale (int): Camera model scale. Defaults to 1.
        color (list): Color of the image plane and pyramid lines. Defaults to [0.8, 0.2, 0.8].

    Returns:
        open3d.geometry.TriangleMesh: axis.
        open3d.geometry.TriangleMesh: plane.
        o3d.geometry.LineSet: lines set.

    """

    assert intrinsic_mat and rotate_mat and trans_mat and width and height
    scale = kwargs.get('scale', 1)
    color = kwargs.get('color', [0.8, 0.2, 0.8])

    # The intrinsics parameters
    scaled_intrinsics = np.array([[intrinsic_mat[0, 0] / scale,                           0, intrinsic_mat[0, 2]],
                                  [                          0, intrinsic_mat[1, 1] / scale, intrinsic_mat[1, 2]],
                                  [                          0,                           0, intrinsic_mat[2, 2]]])

    inv_scaled_intrinsics = np.linalg.inv(scaled_intrinsics)

    # The extrinsic parameters (4x4 transformation)
    extrinsic = np.column_stack((rotate_mat, trans_mat))
    extrinsic = np.vstack((extrinsic, (0, 0, 0, 1)))

    # Generate the axis
    axis = _create_coordinate_frame(trans_mat, scale=scale*0.5)

    # points in pixel
    points_pixel = [[    0,      0, 0],
                    [    0,      0, 1],
                    [width,      0, 1],
                    [    0, height, 1],
                    [width, height, 1]]

    # pixel to camera coordinate system
    points = [scale * inv_scaled_intrinsics @ p for p in points_pixel]

    # image plane
    plane_width = abs(points[1][0]) + abs(points[3][0])
    plane_height = abs(points[1][1]) + abs(points[3][1])
    plane = o3d.geometry.TriangleMesh.create_box(plane_width, plane_height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.transform(trans_mat)
    plane.translate(rotate_mat @ [points[1][0], points[1][1], scale])

    # pyramid
    points_in_world = [(rotate_mat @ p + trans_mat) for p in points]
    lines = [[0, 1],
             [0, 2],
             [0, 3],
             [0, 4]]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points_in_world),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return [axis, plane, line_set]


def _create_coordinate_frame(trans_mat, scale=0.25):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.transform(trans_mat)
    return frame


def draw_points3D(points, **kwargs):
    """Draw 3d points.

    Args:
        points (open3d.geometry.Geometry3D): 3D points.
        color (list): 3D points Color. Defaults to [0.8, 0.2, 0.8].
        radius (float, optional): 3D points radius. Defaults to 0.01.
        resolution (int): 3D points resolution. Defaults to 20.

    Returns:
        list: the geometries of 3D points.
    """    
    assert points
    color = kwargs.get('color', [0.8, 0.2, 0.8])
    radius = kwargs.get('radius', 0.01)
    resolution = kwargs.get('resolution', 20)

    geometries = []
    for pt in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius,
                                                         resolution=resolution)
        sphere.translate(pt)
        sphere.paint_uniform_color(np.array(color))
        geometries.append(sphere)

    return geometries