import open3d as o3d
import numpy as np 

def read_pts(filepath, data_format='xyzirgb'):
    """Read point cloud data in pts format.

    Args:
        filepath (str): The point cloud data file (*.pts).
        data_format (str, optional): The data format of the pts file ('xyzirgb' or 'xyzrgb'). 
                                     Defaults to 'xyzirgb'.

    Returns:
        o3d.open3d.geometry.PointCloud: The point cloud data.
    """  

    assert data_format in ['xyzirgb', 'xyzrgb']

    # Reads lines from the pts file.
    # The first line gives the number of points to follow.
    # Each subsequent line has 7 values, the first three are the (x,y,z) coordinates of the point, 
    # the fourth is an "intensity" value (only avaiable in 'xyzirgb' format), 
    # and the last three are the (r,g,b) colour values (range from 0 to 255).
    try:
        f = open(filepath)
        contents = f.readlines()
        f.close()
    except Exception as e:
        print(e)
        return None, None
    
    if len(contents) == 0:
        return None, None

    # Placeholders for points and colors.
    points = np.zeros((int(contents[0]), 3))
    colors = np.zeros((int(contents[0]), 3))

    for i in range(1, len(contents)):
        info = contents[i].strip().split(' ')
        
        for j in range(3):
            points[i-1, j] = float(info[j]) # points
            skip_place = 1 if data_format == 'xyzirgb' else 0 # disregard the intensity value
            colors[i-1, j] = int(info[j+3+skip_place]) # colors

    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.)

    return pcd