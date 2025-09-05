import math
import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mph
import os


src_dir = r"comsol\multilayer_type2(new)"

client = mph.start()
name = client.names()
pymodel = client.load(r"comsol\Hand-built-model-type2.mph")
model = pymodel.java

def rotate_point(pivot, point, angle_deg, clockwise=True):
    angle_rad = math.radians(-angle_deg if clockwise else angle_deg)
    ox, oy = pivot
    px, py = point
    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    return (round(qx, 4), round(qy, 4))

def angle_between_vectors(p1, p2, center):
    v1 = np.array([p1[0] - center[0], p1[1] - center[1]])
    v2 = np.array([p2[0] - center[0], p2[1] - center[1]])
    dot_prod = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    cos_angle = dot_prod / (mag_v1 * mag_v2)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(angle_rad)

# Initial Points
points = [
    (49.90, 45),    # p0
    (49.90, 44.51), # p1
    (49.55, 42.84), # p2 (rotates)
    (49.66, 42.82), # p3 (rotates)
    (49.90, 43.94), # p4 (centre of rotation)
    (49.90, 42),    # p5
    (50.05, 42),    # p6
    (50.05, 43.24), # p7 (centre of rotation)
    (50.34, 41.89), # p8 (rotates)
    (50.46, 41.91), # p9 (rotates)
    (50.05, 43.81), # p10
    (50.05, 45),    # p11
]



# rotation parameters
outer_rotation_step = 10
inner_rotation_step = 1

cir_current = [5175, 2835, 2070, 1665, 1485, 1035, 990, 900, 855, 675]
freq = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

def is_valid_polygon(poly_points):
    poly = Polygon(poly_points)
    return poly.is_valid

# Three-layer nested loop transformation
for outer_ang in range(15, 76, outer_rotation_step):  # Outer rotation
    # Rotate all points on the outer layer (except p0 and p11) around the line segment p0p11.
    pivot_line = LineString([points[0], points[11]])
    pivot_center = pivot_line.interpolate(0.5, normalized=True).coords[0]

    rotated_points_outer = [points[0]]
    for i in range(1, 11):
        rotated_p = rotate_point(pivot_center, points[i], outer_ang, clockwise=True)
        rotated_points_outer.append(rotated_p)
    rotated_points_outer.append(points[11])

    # Second layer loop: p2 and p3 rotate around p4
    for mid_angle in range(0, 20, inner_rotation_step):
        rotated_points_mid = rotated_points_outer.copy()

        rotated_points_mid[2] = rotate_point(rotated_points_outer[4], rotated_points_outer[2], mid_angle, clockwise=True)
        rotated_points_mid[3] = rotate_point(rotated_points_outer[4], rotated_points_outer[3], mid_angle, clockwise=True)
        angle1 = angle_between_vectors(rotated_points_mid[3], rotated_points_mid[5], rotated_points_mid[4])

        # Innermost loop: p8 and p9 rotate around p7 (counterclockwise)
        for inner_angle in range(0, 20, inner_rotation_step):
            final_points = rotated_points_mid.copy()

            final_points[8] = rotate_point(rotated_points_mid[7], rotated_points_mid[8], inner_angle, clockwise=False)
            final_points[9] = rotate_point(rotated_points_mid[7], rotated_points_mid[9], inner_angle, clockwise=False)

            angle2 = angle_between_vectors(final_points[6], final_points[8], final_points[7])
            # Check polygon validity (no self-intersection)
            if is_valid_polygon(final_points):
                rotated_points = final_points

                new_x1 = rotated_points[1][0]
                new_z1 = rotated_points[1][1]
                new_x2 = rotated_points[2][0]
                new_z2 = rotated_points[2][1]
                new_x3 = rotated_points[3][0]
                new_x4 = rotated_points[4][0]
                new_z3 = rotated_points[3][1]
                new_z4 = rotated_points[4][1]
                new_x5 = rotated_points[5][0]
                new_z5 = rotated_points[5][1]
                new_x6 = rotated_points[6][0]
                new_z6 = rotated_points[6][1]
                new_x7 = rotated_points[7][0]
                new_z7 = rotated_points[7][1]
                new_x8 = rotated_points[8][0]
                new_z8 = rotated_points[8][1]
                new_x9 = rotated_points[9][0]
                new_z9 = rotated_points[9][1]
                new_x10 = rotated_points[10][0]
                new_z10 = rotated_points[10][1]

                # Update Params
                model.param().set('x1', str(new_x1))
                model.param().set('z1', str(new_z1))
                model.param().set('x2', str(new_x2))
                model.param().set('z2', str(new_z2))
                model.param().set('x3', str(new_x3))
                model.param().set('z3', str(new_z3))
                model.param().set('x4', str(new_x4))
                model.param().set('z4', str(new_z4))
                model.param().set('x5', str(new_x5))
                model.param().set('z5', str(new_z5))
                model.param().set('x6', str(new_x6))
                model.param().set('z6', str(new_z6))
                model.param().set('x7', str(new_x7))
                model.param().set('z7', str(new_z7))
                model.param().set('x8', str(new_x8))
                model.param().set('z8', str(new_z8))
                model.param().set('x9', str(new_x9))
                model.param().set('z9', str(new_z9))
                model.param().set('x10', str(new_x10))
                model.param().set('z10', str(new_z10))

                save_path = os.path.join(src_dir, f'multilayer_type2_{90-outer_ang}deg_{angle1}deg_{angle2}deg')

                unstr_dir = os.path.join(save_path, 'unstructured_data')
                os.makedirs(unstr_dir, exist_ok=True)
                str_dir = os.path.join(save_path, 'structured_data')
                os.makedirs(str_dir, exist_ok=True)
                surf_dir = os.path.join(save_path, 'structured_surf_data')
                os.makedirs(surf_dir, exist_ok=True)

                for i, f in enumerate(freq):
                    cir = cir_current[i]
                    model.param().set('f0', str(f) + '[kHz]')
                    model.param().set('I0', str(cir) + '[A]')

                    model.study("std1").run()

                    model.result().export('data1').set('coordfilename', r'comsol/mixed_distribution_points.txt')
                    model.result().export('data3').set('coordfilename', r'comsol/mixed_distribution_points.txt')
                    model.result().export('data4').set('coordfilename', r'comsol/uniform_distribution_points.txt')
                    model.result().export('data1').set('filename', os.path.join(unstr_dir, f'multilayer_type2_{90-outer_ang}deg_{angle1}deg_{angle2}deg_{f}kHz_{cir}A' + '_T.vtu'))
                    model.result().export('data3').set('filename', os.path.join(unstr_dir, f'multilayer_type2_{90-outer_ang}deg_{angle1}deg_{angle2}deg_{f}kHz_{cir}A' + '_Q.vtu'))
                    model.result().export('data4').set('filename', os.path.join(surf_dir, f'multilayer_type2_{90-outer_ang}deg_{angle1}deg_{angle2}deg_{f}kHz_{cir}A' + '_Tsurf.vtu'))
                    model.result().export('data1').run()
                    model.result().export('data3').run()
                    model.result().export('data4').run()
                    model.result().export('data1').set('coordfilename', r'comsol/uniform_linear_distribution_points.txt')
                    model.result().export('data3').set('coordfilename', r'comsol/uniform_linear_distribution_points.txt')
                    model.result().export('data1').set('filename', os.path.join(str_dir, f'multilayer_type2_{90-outer_ang}deg_{angle1}deg_{angle2}deg_{f}kHz_{cir}A' + '_T.vtu'))
                    model.result().export('data3').set('filename', os.path.join(str_dir, f'multilayer_type2_{90-outer_ang}deg_{angle1}deg_{angle2}deg_{f}kHz_{cir}A' + '_Q.vtu'))
                    model.result().export('data1').run()
                    model.result().export('data3').run()

                result_file = os.path.join(save_path, f'3_layer_type2_{90-outer_ang}deg_{angle1}deg_{angle2}deg.mph')
                pymodel.save(result_file)
                print(f'SAVE: multilayer_type2_{90-outer_ang}deg_{angle1}deg_{angle2}deg')
