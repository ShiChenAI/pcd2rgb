import open3d as o3d
import numpy as np 
import os
import random
import copy
from scipy.spatial.transform import Rotation as R # scipy >= 1.4.0

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

    assert point_cloud_data is not None 
    assert visualizer is not None
    assert min(steps) > -2
    steps = kwargs.get('steps', (-1, -1, -1)) 
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

    assert point_cloud_data is not None
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
        width (int): Image width. 
        height (int): Image height. 
        scale (int, optional): Camera model scale. Defaults to 1.
        color (list, optional): Color of the image plane and pyramid lines. Defaults to [0.8, 0.2, 0.8].
    """    

    assert visualizer is not None 
    assert width > 0 
    assert height > 0
    scale = kwargs.get('scale', 1)
    color = kwargs.get('color', [0.8, 0.2, 0.8])

    # Get intrinsic and extrinsic parameters of render camera.
    intrinsic, extrinsic = get_camera_params(visualizer)

    # convert extrinsics matrix to rotation and translation matrix
    extrinsic = np.linalg.inv(extrinsic)
    rotate_mat = extrinsic[0:3, 0:3]
    trans_mat = extrinsic[0:3, 3]

    geometries = _draw_camera_geometries(intrinsic.intrinsic_matrix, rotate_mat, trans_mat, 
                                         width, height, scale=scale, color=color)
    for g in geometries:
        visualizer.add_geometry(g)

def move_camera(visualizer, params, **kwargs):
    """Move the camera at specific coordinates

    Args:
        visualizer (o3d.visualization.Visualizer): The point cloud visualizer.
        params (open3d.camera.PinholeCameraParameters): The camera parameters.
        pos (list, optional): Targrt position. Defaults to None.
        disp (list, optional): Displacement. Defaults to None.
        save_path (str, optional): The save path to the rendered image. Defaults to None.
        vis_camera (bool, optional): Visualize the camera or not. Defaults to False.
        width (int, optional): Image width. Defaults to 1920.
        height (int, optional): Image height. Defaults to 1080.
    """    

    assert visualizer is not None
    pos = kwargs.get('pos', None)
    disp = kwargs.get('disp', None)
    assert pos is not None or disp is not None
    save_path = kwargs.get('save_path', None)
    vis_camera = kwargs.get('vis_camera', False)

    # Get current extrinsic parameters of render camera.
    ctr = visualizer.get_view_control()
    #params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = params.extrinsic.copy()

    # Reset the camera position
    if pos:
        extrinsic[:3, 3] = pos
    elif disp:
        extrinsic[:3, 3] += disp
    params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(params)

    if vis_camera:
        width = kwargs.get('width ', 1920)
        height = kwargs.get('height ', 1080)
        draw_camera(visualizer, width, height)
    if save_path:
        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.capture_screen_image(save_path)

def rotate_camera(visualizer, ref_extrinsic, axis, degree, **kwargs):
    """Rotate the camera at a specific degree.

    Args:
        visualizer (o3d.visualization.Visualizer): The point cloud visualizer.
        ref_extrinsic (ndarray): The reference matrix (extrinsic parameters) for this rotation.
        axis (str): The axis around which the rotation is performed.
        degree (float): The degree of rotation.
        save_path (str, optional): The save path to the rendered image. Defaults to None.
        vis_camera (bool, optional): Visualize the camera or not. Defaults to False.
        width (int, optional): Image width. Defaults to 1920.
        height (int, optional): Image height. Defaults to 1080.
    """

    assert visualizer is not None
    assert ref_extrinsic is not None
    assert axis is not None
    assert degree is not None
    save_path = kwargs.get('save_path', None)
    vis_camera = kwargs.get('vis_camera', False)

    # Get current parameters of render camera.
    ctr = visualizer.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()

    # Calculate the extrinsic parameters of camera.
    rot = np.eye(4)
    rot[:3, :3] = R.from_euler(axis, degree, degrees=True).as_matrix()
    params.extrinsic = np.dot(rot, ref_extrinsic)
    ctr.convert_from_pinhole_camera_parameters(params)

    if vis_camera:
        width = kwargs.get('width ', 1920)
        height = kwargs.get('height ', 1080)
        draw_camera(visualizer, width, height)
    if save_path:
        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.capture_screen_image(save_path)

def rotate_camera_around_object(visualizer, ref_extrinsic, axis, degree, **kwargs):
    """Rotate the camera around the object (point cloud) at a specific angle.

    Args:
        visualizer (o3d.visualization.Visualizer): The point cloud visualizer.
        ref_extrinsic (ndarray): The reference matrix (extrinsic parameters) for this rotation.
        axis (str): The axis around which the rotation is performed.
        degree (float): The degree of rotation.
        save_path (str, optional): The save path to the rendered image. Defaults to None.
        vis_camera (bool, optional): Visualize the camera or not. Defaults to False.
        width (int, optional): Image width. Defaults to 1920.
        height (int, optional): Image height. Defaults to 1080.
    """

    assert visualizer is not None
    assert ref_extrinsic is not None
    assert axis is not None
    assert degree is not None
    save_path = kwargs.get('save_path', None)
    vis_camera = kwargs.get('vis_camera', False)

    # Get current parameters of render camera.
    ctr = visualizer.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()

    # Calculate the extrinsic parameters of camera.
    rot = np.eye(4)
    rot[:3, :3] = R.from_euler(axis, degree, degrees=True).as_matrix()
    params.extrinsic = np.dot(ref_extrinsic, rot)
    ctr.convert_from_pinhole_camera_parameters(params)

    if vis_camera:
        width = kwargs.get('width ', 1920)
        height = kwargs.get('height ', 1080)
        draw_camera(visualizer, width, height)
    if save_path:
        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.capture_screen_image(save_path)


def get_camera_params(visualizer, shallow_copy=False):
    """Get intrinsic and extrinsic parameters of render camera.

    Args:
        visualizer (o3d.visualization.Visualizer): The point cloud visualizer.
        shallow_copy (bool, optional): Perform shallow copy operation or not. Defaults to False.

    Returns:
        open3d.camera.PinholeCameraIntrinsic: The intrinsic parameters.
        ndarray: The extrinsic parameters.
    """    

    assert visualizer is not None
    ctr = visualizer.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    intrinsic = params.intrinsic
    extrinsic = params.extrinsic

    return (copy.copy(intrinsic), extrinsic.copy()) if shallow_copy else (intrinsic, extrinsic)

def save_view_point(point_cloud_data, visualizer, save_path):
    assert point_cloud_data is not None
    assert visualizer is not None 
    assert save_path is not None

    params = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(save_path, params)


def _draw_camera_geometries(intrinsic_mat, rotate_mat, trans_mat, width, height, **kwargs):
    """Create axis, plane and pyramid geometries in Open3D format

    Args:
        intrinsic_mat (ndarray): The calibration matrix (camera intrinsics)
        rotate_mat (ndarray): The rotation matrix
        trans_mat (ndarray): The translation
        width (int): image width
        height (int): image height
        scale (int, optional): Camera model scale. Defaults to 1.
        color (list, optional): Color of the image plane and pyramid lines. 
                                Defaults to [0.8, 0.2, 0.8].

    Returns:
        open3d.geometry.TriangleMesh: axis.
        open3d.geometry.TriangleMesh: plane.
        o3d.geometry.LineSet: lines set.

    """

    assert intrinsic_mat is not None
    assert rotate_mat is not None
    assert trans_mat is not None
    assert width 
    assert height

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
    axis = _create_coordinate_frame(extrinsic, scale=scale*0.5)

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
    plane.transform(extrinsic)
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
        color (list, optional): 3D points Color. Defaults to [0.8, 0.2, 0.8].
        radius (float, optional): 3D points radius. Defaults to 0.01.
        resolution (int, optional): 3D points resolution. Defaults to 20.

    Returns:
        list: the geometries of 3D points.
    """    

    assert points is not None
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