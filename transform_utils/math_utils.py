import numpy as np 
import transformations as tf_transformations

def angle_boxminus(a, b):
        res = a - b
        return res - 2 * np.pi * np.floor((res + np.pi) / (2 * np.pi))
    
def rotate(angle, vec):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([vec[0]*c - vec[1]*s, vec[0]*s + vec[1]*c])

def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def transform(pose, vec):
    vec = rotate(pose[2], vec)
    return pose[:2] + vec

def pose_boxminus(a, b):
    angle_diff = angle_boxminus(a[2], b[2])
    pos_diff = rotate(-b[2], a[:2] - b[:2])
    return np.array([pos_diff[0], pos_diff[1], angle_diff])

def get_position_and_yaw(pose):
    x = pose[0, 3]
    y = pose[1, 3]       
    yaw = np.arctan2(pose[1, 0], pose[0, 0])   
    return np.array([x, y, yaw])

def relative_T(T_w_from_v1, T_w_from_v2):
    """Return T_{v2<-v1} that maps points from frame1's ego to frame2's ego."""
    return np.linalg.inv(T_w_from_v2) @ T_w_from_v1

def make_4x4_from_xy_yaw(x, y, yaw):
        """Create a 4x4 pure-Yaw transform from x,y,yaw."""
        T = tf_transformations.euler_matrix(0.0, 0.0, yaw)
        T[0, 3] = x
        T[1, 3] = y
        return T    
    


