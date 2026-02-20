import trimesh
import numpy as np
import os
import xml.etree.ElementTree as ET
from sklearn.cluster import DBSCAN

# --- 1. PRE-PROCESSING (Škálování) ---

def normalize_mesh_scale(mesh):
    """
    Zjistí velikost meshe a pokud je mimo normu, vypočítá scale factor.
    Cílem je, aby židle měla výšku cca 1.0 metr (standardní velikost).
    """
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0] # Velikosti [x, y, z]
    height = extents[2]

    target_height = 1.0 # Cílová výška židle v metrech

    # Pokud je židle extrémně velká (např. > 3m) nebo malá (< 0.2m)
    if height > 2.0 or height < 0.2:
        scale_factor = target_height / height
        print(f"  -> DETEKOVÁNA ŠPATNÁ VELIKOST (Výška: {height:.2f}m)")
        print(f"  -> Aplikuji Auto-Scale factor: {scale_factor:.4f} (Cíl: 1.0m)")
    else:
        scale_factor = 1.0
        print(f"  -> Velikost se zdá OK (Výška: {height:.2f}m). Scale: 1.0")

    # Aplikujeme scale přímo na mesh v paměti, aby analýza bodů seděla
    matrix = np.eye(4)
    matrix[:3, :3] *= scale_factor
    mesh.apply_transform(matrix)

    return mesh, scale_factor

# --- 2. ANALÝZA ---

def analyze_chair_features(mesh):
    if mesh.is_empty: return None

    bounds = mesh.bounds
    min_z_abs = bounds[0][2]
    max_z = bounds[1][2]

    # 1. Těžiště
    com = mesh.center_mass

    # 2. Opěradlo
    top_slice_threshold = max_z - 0.05
    top_indices = np.where(mesh.vertices[:, 2] > top_slice_threshold)[0]

    if len(top_indices) > 0:
        top_points = mesh.vertices[top_indices]
        center_xy = np.mean(top_points[:, :2], axis=0)
        dists = np.linalg.norm(top_points[:, :2] - center_xy, axis=1)
        backrest_top_center = top_points[np.argmin(dists)]

        # --- OPRAVA OSY RUKOU ---
        # Uživatel chce osu X.
        # Offset 7.5 cm na každou stranu = 15 cm rozestup.
        offset_vector = np.array([0.15, 0, 0]) # X-axis offset

        target_left = backrest_top_center + offset_vector
        target_right = backrest_top_center - offset_vector
    else:
        fallback = np.array([0, 0, max_z])
        backrest_top_center = target_left = target_right = fallback

    # 3. Nohy (Adaptivní hledání)
    feet_points = []
    current_slice = 0.02
    max_slice = 0.25
    step = 0.01

    # Protože jsme židli zmenšili, musíme zmenšit i parametry hledání!
    # Pokud byla židle 10x větší, 15cm rádius byl malý. Teď je 15cm akorát.
    search_eps = 0.15 # 15 cm rádius pro spojení bodů nohy

    while current_slice <= max_slice:
        threshold = min_z_abs + current_slice
        bottom_indices = np.where(mesh.vertices[:, 2] < threshold)[0]

        if len(bottom_indices) == 0:
            current_slice += step
            continue

        bottom_points = mesh.vertices[bottom_indices]
        clustering = DBSCAN(eps=search_eps, min_samples=1).fit(bottom_points)
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}

        if len(unique_labels) >= 4:
            for label in unique_labels:
                cluster = bottom_points[labels == label]
                center_xy = np.mean(cluster[:, :2], axis=0)
                z_floor = np.min(cluster[:, 2])
                feet_points.append(np.array([center_xy[0], center_xy[1], z_floor]))
            break
        current_slice += step

    # Fallback pro nohy
    if not feet_points and len(bottom_indices) > 0:
        clustering = DBSCAN(eps=search_eps, min_samples=1).fit(bottom_points)
        unique_labels = set(clustering.labels_) - {-1}
        for label in unique_labels:
            cluster = bottom_points[clustering.labels_ == label]
            center_xy = np.mean(cluster[:, :2], axis=0)
            z_floor = np.min(cluster[:, 2])
            feet_points.append(np.array([center_xy[0], center_xy[1], z_floor]))

    return {
        "com": com,
        "top_center": backrest_top_center,
        "target_left": target_left,
        "target_right": target_right,
        "feet": feet_points,
        "min_z": min_z_abs
    }

# --- 3. GENERÁTOR URDF ---

def create_urdf_from_mesh(mesh_path, output_urdf_path):
    if not os.path.exists(mesh_path):
        print(f"Chyba: {mesh_path} neexistuje.")
        return

    print(f"--- Načítám mesh: {mesh_path} ---")

    # Loader
    loaded_obj = trimesh.load(mesh_path)
    mesh = None
    if isinstance(loaded_obj, trimesh.Scene):
        try: mesh = loaded_obj.dump(concatenate=True)
        except: mesh = trimesh.util.concatenate(list(loaded_obj.geometry.values()))
    elif isinstance(loaded_obj, trimesh.Trimesh): mesh = loaded_obj
    elif isinstance(loaded_obj, list): mesh = trimesh.util.concatenate(loaded_obj)

    if mesh is None: return

    # --- KROK 1: NORMALIZACE VELIKOSTI ---
    # Zde zmenšíme židli v paměti a získáme scale factor
    mesh, scale_factor = normalize_mesh_scale(mesh)

    # --- KROK 2: ANALÝZA (Už na zmenšené židli) ---
    print("Provádím analýzu geometrie...")
    features = analyze_chair_features(mesh)
    if not features: return

    # Korekce výšky (posun k zemi)
    visual_offset_z = -features["min_z"]

    def adjust_point(p):
        return p + np.array([0, 0, visual_offset_z])

    adj_features = {
        "com": adjust_point(features["com"]),
        "top_center": adjust_point(features["top_center"]),
        "target_left": adjust_point(features["target_left"]),
        "target_right": adjust_point(features["target_right"]),
        "feet": [adjust_point(f) for f in features["feet"]]
    }

    print(f"  -> Nalezeno nohou: {len(adj_features['feet'])}")
    print(f"  -> Terče pro ruce nastaveny v ose X (rozestup 15cm).")

    # --- KROK 3: TVORBA XML ---
    robot_name = "foldable_chair_gen"
    robot = ET.Element("robot", name=robot_name)

    def make_link(name, mass="0.01"):
        l = ET.SubElement(robot, "link", name=name)
        if mass != "0":
            i = ET.SubElement(l, "inertial")
            ET.SubElement(i, "mass", value=mass)
            ET.SubElement(i, "inertia", ixx="0.001", ixy="0", ixz="0", iyy="0.001", iyz="0", izz="0.001")
        return l

    make_link("world", "0")
    make_link("slide_link_x")
    make_link("slide_link_y")

    def add_prismatic(name, p, c, axis):
        j = ET.SubElement(robot, "joint", name=name, type="prismatic")
        ET.SubElement(j, "parent", link=p)
        ET.SubElement(j, "child", link=c)
        ET.SubElement(j, "axis", xyz=axis)
        ET.SubElement(j, "limit", lower="-10", upper="10", effort="1000", velocity="5")

    add_prismatic("floor_slide_x", "world", "slide_link_x", "1 0 0")
    add_prismatic("floor_slide_y", "slide_link_x", "slide_link_y", "0 1 0")

    j_rot = ET.SubElement(robot, "joint", name="floor_rotate_z", type="continuous")
    ET.SubElement(j_rot, "parent", link="slide_link_y")
    ET.SubElement(j_rot, "child", link="base_link")
    ET.SubElement(j_rot, "axis", xyz="0 0 1")
    ET.SubElement(j_rot, "dynamics", damping="1.0")

    # BASE LINK
    base = ET.SubElement(robot, "link", name="base_link")

    # VISUAL - Zde musíme použít vypočítaný SCALE FACTOR!
    vis = ET.SubElement(base, "visual")
    ET.SubElement(vis, "origin", xyz=f"0 0 {visual_offset_z}", rpy="0 0 0")
    geo = ET.SubElement(vis, "geometry")
    # scale="X X X" zajistí, že vizuál bude stejně velký jako naše analýza
    ET.SubElement(geo, "mesh", filename=os.path.abspath(mesh_path), scale=f"{scale_factor} {scale_factor} {scale_factor}")

    # COLLISION - Také se scalem
    col = ET.SubElement(base, "collision")
    ET.SubElement(col, "origin", xyz=f"0 0 {visual_offset_z}", rpy="0 0 0")
    geo_c = ET.SubElement(col, "geometry")
    ET.SubElement(geo_c, "mesh", filename=os.path.abspath(mesh_path), scale=f"{scale_factor} {scale_factor} {scale_factor}")

    inertial = ET.SubElement(base, "inertial")
    ET.SubElement(inertial, "mass", value="5.0")
    ET.SubElement(inertial, "origin", xyz=f"{adj_features['com'][0]} {adj_features['com'][1]} {adj_features['com'][2]}")
    ET.SubElement(inertial, "inertia", ixx="0.1", ixy="0", ixz="0", iyy="0.1", iyz="0", izz="0.1")

    # DEBUG MARKERS
    def add_marker(name, pos, color, radius, shape="sphere", length=None):
        l = ET.SubElement(robot, "link", name=name)
        v = ET.SubElement(l, "visual")
        ET.SubElement(v, "origin", xyz="0 0 0", rpy="0 0 0")
        g = ET.SubElement(v, "geometry")
        if shape == "sphere": ET.SubElement(g, "sphere", radius=str(radius))
        elif shape == "cylinder": ET.SubElement(g, "cylinder", radius=str(radius), length=str(length))
        m = ET.SubElement(v, "material", name=f"m_{name}")
        ET.SubElement(m, "color", rgba=color)
        j = ET.SubElement(robot, "joint", name=f"j_{name}", type="fixed")
        ET.SubElement(j, "parent", link="base_link")
        ET.SubElement(j, "child", link=name)
        ET.SubElement(j, "origin", xyz=f"{pos[0]} {pos[1]} {pos[2]}", rpy="0 0 0")

    add_marker("debug_top_center", adj_features['top_center'], "1 0 0 1", 0.04)
    add_marker("target_hand_left", adj_features['target_left'], "1 1 0 1", 0.03)
    add_marker("target_hand_right", adj_features['target_right'], "1 1 0 1", 0.03)
    add_marker("debug_com", adj_features['com'], "0 0 1 1", 0.05)
    add_marker("debug_vec", adj_features['com']+[0,0,0.25], "0 0 1 0.5", 0.01, "cylinder", 0.5)

    for i, foot in enumerate(adj_features['feet']):
        add_marker(f"debug_foot_{i}", foot, "0 1 0 1", 0.04)

    tree = ET.ElementTree(robot)
    tree.write(output_urdf_path)
    print(f"Hotovo! URDF uloženo s měřítkem {scale_factor:.4f}: {output_urdf_path}")

# --- SPUŠTĚNÍ ---
base_dir = "/home/roboversepc/Downloads/officeChairGrey"
mesh_file = os.path.join(base_dir, "mesh", "OfficeChairGrey.dae")
output_file = os.path.join(base_dir, "office_chair_grey_debug.urdf")

create_urdf_from_mesh(mesh_file, output_file)
