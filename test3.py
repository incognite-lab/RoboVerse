import trimesh
import numpy as np
import os
import xml.etree.ElementTree as ET
from sklearn.cluster import DBSCAN

# --- 1. OPRAVA MATERIÁLŮ (MTL FIXER) ---
def fix_texture_link(base_folder):
    """
    1. Najde .obj, .mtl a .png/.jpg ve složce.
    2. Přečte skutečný název obrázku na disku.
    3. Opraví .mtl soubor tak, aby odkazoval přesně na tento obrázek.
    """
    meshes_dir = os.path.join(base_folder, "meshes")
    if not os.path.exists(meshes_dir):
        meshes_dir = base_folder # Zkusíme root, pokud meshes neexistuje

    # Najdeme soubory
    mtl_file = None
    image_file = None
    obj_file = None

    # Prohledáme složku
    for f in os.listdir(meshes_dir):
        if f.endswith(".mtl") and "_Col" not in f:
            mtl_file = os.path.join(meshes_dir, f)
        elif f.lower().endswith(('.png', '.jpg', '.jpeg', '.tga')):
            image_file = f # Uložíme jen název souboru
        elif f.endswith(".obj") and "_Col" not in f:
            obj_file = os.path.join(meshes_dir, f)

    if not mtl_file:
        print("  -> CHYBA: Nenalezen soubor .mtl")
        return None, None

    if not image_file:
        print("  -> CHYBA: Nenalezen žádný obrázek (.png/.jpg) ve složce!")
        return None, None

    print(f"  -> Opravuji materiál '{os.path.basename(mtl_file)}'...")
    print(f"     Nastavuji texturu na: '{image_file}'")

    # Přečteme a opravíme MTL
    with open(mtl_file, 'r') as f:
        lines = f.readlines()

    with open(mtl_file, 'w') as f:
        found_map = False
        for line in lines:
            if line.strip().startswith("map_Kd"):
                # Nahradíme jakoukoliv cestu pouze názvem souboru
                f.write(f"map_Kd {image_file}\n")
                found_map = True
            else:
                f.write(line)
        # Pokud map_Kd chybělo, přidáme ho
        if not found_map:
            f.write(f"\nmap_Kd {image_file}\n")

    return obj_file, image_file

# --- 2. ANALÝZA GEOMETRIE ---
def analyze_geometry(mesh):
    if mesh.is_empty: return None

    bounds = mesh.bounds
    min_z_abs = bounds[0][2]
    max_z = bounds[1][2]
    com = mesh.center_mass

    # --- Opěradlo ---
    top_slice_threshold = max_z - 0.05
    top_indices = np.where(mesh.vertices[:, 2] > top_slice_threshold)[0]

    if len(top_indices) > 0:
        top_points = mesh.vertices[top_indices]
        center_xy = np.mean(top_points[:, :2], axis=0)
        dists = np.linalg.norm(top_points[:, :2] - center_xy, axis=1)
        backrest_top_center = top_points[np.argmin(dists)]

        # Detekce orientace (X vs Y) podle rozptylu bodů
        std_x = np.std(top_points[:, 0])
        std_y = np.std(top_points[:, 1])

        # Ruce 20cm od sebe (offset 0.10 na každou stranu)
        offset_dist = 0.20
        if std_y > std_x:
            offset = np.array([0, offset_dist, 0])
        else:
            offset = np.array([offset_dist, 0, 0])

        target_left = backrest_top_center + offset
        target_right = backrest_top_center - offset
    else:
        fallback = np.array([0, 0, max_z])
        backrest_top_center = target_left = target_right = fallback

    # --- Nohy ---
    feet_points = []
    current_slice = 0.02
    max_slice = 0.40 # Kancelářské židle mají vysoký kříž
    step = 0.01
    search_eps = 0.15

    print("  -> Hledám nohy...")
    while current_slice <= max_slice:
        threshold = min_z_abs + current_slice
        bottom_indices = np.where(mesh.vertices[:, 2] < threshold)[0]

        if len(bottom_indices) == 0:
            current_slice += step
            continue

        bottom_points = mesh.vertices[bottom_indices]
        clustering = DBSCAN(eps=search_eps, min_samples=1).fit(bottom_points)
        unique_labels = set(clustering.labels_) - {-1}

        # Hledáme alespoň 4 ramena
        if len(unique_labels) >= 4:
            print(f"     -> Nalezeno {len(unique_labels)} ramen (výška řezu {current_slice:.2f}m).")
            for label in unique_labels:
                cluster = bottom_points[clustering.labels_ == label]
                center_xy = np.mean(cluster[:, :2], axis=0)
                z_floor = np.min(cluster[:, 2])
                feet_points.append(np.array([center_xy[0], center_xy[1], z_floor]))
            break
        current_slice += step

    # Fallback pokud se nohy nenajdou
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

# --- 3. HLAVNÍ PROCES ---
def process_chair(base_folder):
    if not os.path.exists(base_folder):
        print(f"Chyba: Složka {base_folder} neexistuje!")
        return

    print(f"=== Zpracovávám židli v: {base_folder} ===")

    # 1. OPRAVA TEXTUR A ZÍSKÁNÍ CEST
    obj_path, img_name = fix_texture_link(base_folder)

    if not obj_path:
        print("Nepodařilo se najít OBJ soubor.")
        return

    # 2. NAČTENÍ A ŠKÁLOVÁNÍ
    print(f"--- Analyzuji model: {obj_path} ---")
    try:
        # Načteme geometrii pro výpočty
        mesh = trimesh.load(obj_path, force='mesh')
    except:
        scene = trimesh.load(obj_path, force='scene')
        mesh = scene.dump(concatenate=True)

    # Změření výšky
    bounds = mesh.bounds
    height = bounds[1][2] - bounds[0][2]

    # Cílová výška pro kancelářskou židli
    target_height = 1.15
    scale_factor = 1.0

    if height > 2.0 or height < 0.2:
        scale_factor = target_height / height
        print(f"  -> Původní výška: {height:.2f}m (Moc velké/malé)")
        print(f"  -> Scale Factor: {scale_factor:.6f}")
    else:
        print(f"  -> Výška OK: {height:.2f}m")

    # 3. ANALÝZA GEOMETRIE (na zmenšeném modelu v paměti)
    matrix = np.eye(4)
    matrix[:3, :3] *= scale_factor
    mesh.apply_transform(matrix)

    features = analyze_geometry(mesh)

    # Korekce Z (aby stála na zemi)
    visual_offset_z = -features["min_z"]

    def adjust(p):
        return p + np.array([0, 0, visual_offset_z])

    adj_features = {k: adjust(v) if k != 'feet' else [adjust(f) for f in v]
                    for k, v in features.items() if k != 'min_z'}

    print(f"  -> Nalezeno nohou: {len(adj_features['feet'])}")

    # 4. GENEROVÁNÍ URDF
    robot_name = "office_chair_fixed"
    robot = ET.Element("robot", name=robot_name)

    # Links
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

    # Joints
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

    # Absolutní cesta k modelu
    abs_obj_path = os.path.abspath(obj_path)
    str_scale = f"{scale_factor:.6f} {scale_factor:.6f} {scale_factor:.6f}"

    # Visual (Mesh)
    vis = ET.SubElement(base, "visual")
    ET.SubElement(vis, "origin", xyz=f"0 0 {visual_offset_z}", rpy="0 0 0")
    geo = ET.SubElement(vis, "geometry")
    ET.SubElement(geo, "mesh", filename=abs_obj_path, scale=str_scale)

    # Collision
    col = ET.SubElement(base, "collision")
    ET.SubElement(col, "origin", xyz=f"0 0 {visual_offset_z}", rpy="0 0 0")
    geo_c = ET.SubElement(col, "geometry")
    ET.SubElement(geo_c, "mesh", filename=abs_obj_path, scale=str_scale)

    # Inertial
    inertial = ET.SubElement(base, "inertial")
    ET.SubElement(inertial, "mass", value="15.0")
    ET.SubElement(inertial, "origin", xyz=f"{adj_features['com'][0]} {adj_features['com'][1]} {adj_features['com'][2]}")
    ET.SubElement(inertial, "inertia", ixx="0.1", ixy="0", ixz="0", iyy="0.1", iyz="0", izz="0.1")

    # Debug Markery
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

    output_path = os.path.join(base_folder, "office_chair_fixed.urdf")
    tree = ET.ElementTree(robot)
    tree.write(output_path)
    print(f"Hotovo! URDF uloženo: {output_path}")

# --- SPUŠTĚNÍ ---
folder = "/home/roboversepc/Downloads/OfficeChairGrey"
process_chair(folder)
