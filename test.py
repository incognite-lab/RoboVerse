import trimesh
import numpy as np
import os
import xml.etree.ElementTree as ET
from sklearn.cluster import DBSCAN
import copy

# --- MATH UTILS ---
def rpy_to_matrix(coords):
    r, p, y = [float(x) for x in coords.split()]
    return trimesh.transformations.compose_matrix(angles=[r, p, y], translate=[0, 0, 0])

def xyz_to_matrix(coords):
    x, y, z = [float(val) for val in coords.split()]
    return trimesh.transformations.translation_matrix([x, y, z])

def parse_origin(origin_elem):
    if origin_elem is None: return np.eye(4)
    xyz = origin_elem.get('xyz', '0 0 0')
    rpy = origin_elem.get('rpy', '0 0 0')
    return np.dot(xyz_to_matrix(xyz), rpy_to_matrix(rpy))

# --- GEOMETRY PARSER ---
def load_chair_geometry(urdf_path, root_link_name="base_link"):
    """
    Načte geometrii židle od zadaného root linku (base_link).
    Ignoruje vše nad tím (slide jointy, world atd.).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {l.get('name'): l for l in root.findall('link')}

    # Najdeme jointy jen v rámci podstromu židle
    joints = {}
    for j in root.findall('joint'):
        p = j.find('parent').get('link')
        c = j.find('child').get('link')
        # Ignorujeme jointy, které připojují base_link k světu (např. slide_jointy)
        # Chceme jen vnitřní strukturu židle
        if c == root_link_name:
            continue

        tr = parse_origin(j.find('origin'))
        joints.setdefault(p, []).append((c, tr))

    meshes = []
    # Začínáme analýzu v [0,0,0] base_linku.
    queue = [(root_link_name, np.eye(4))]
    processed = set()

    while queue:
        c_name, c_tf = queue.pop(0)
        processed.add(c_name)

        if c_name in links:
            for vis in links[c_name].findall('visual'):
                geom = vis.find('geometry')
                local_tf = parse_origin(vis.find('origin'))
                final_tf = np.dot(c_tf, local_tf)

                mesh = None
                if geom.find('box') is not None:
                    s = [float(x) for x in geom.find('box').get('size').split()]
                    mesh = trimesh.creation.box(extents=s)
                elif geom.find('cylinder') is not None:
                    r, l = float(geom.find('cylinder').get('radius')), float(geom.find('cylinder').get('length'))
                    mesh = trimesh.creation.cylinder(radius=r, height=l)
                elif geom.find('sphere') is not None:
                    r = float(geom.find('sphere').get('radius'))
                    mesh = trimesh.creation.icosphere(radius=r)

                if mesh:
                    mesh.apply_transform(final_tf)
                    meshes.append(mesh)

        if c_name in joints:
            for child, tf in joints[c_name]:
                if child not in processed:
                    queue.append((child, np.dot(c_tf, tf)))

    if not meshes:
        # Fallback: create empty mesh to prevent crash
        print("Warning: No visual geometry found!")
        return trimesh.Trimesh()

    return trimesh.util.concatenate(meshes)

# --- ANALÝZA ---

def analyze_chair_features(mesh):
    if mesh.is_empty:
        return {"com": [0,0,0], "top_center": [0,0,1], "top_left": [0,0,1], "top_right": [0,0,1], "feet": []}

    bounds = mesh.bounds
    min_z, max_z = bounds[0][2], bounds[1][2]
    height = max_z - min_z

    # 1. Těžiště
    com = mesh.center_mass

    # 2. Opěradlo (vylepšená logika)
    # Vezmeme horních 5 cm
    top_slice_threshold = max_z - 0.05
    top_indices = np.where(mesh.vertices[:, 2] > top_slice_threshold)[0]

    if len(top_indices) > 0:
        top_points = mesh.vertices[top_indices]

        # Střed: Průměr všech bodů nahoře (funguje pro rovné i oblé)
        # Použijeme ale MAX Z z těchto bodů, aby byl bod na vršku, ne uvnitř
        center_x = np.mean(top_points[:, 0])
        center_y = np.mean(top_points[:, 1])
        # Najdeme bod nejblíže tomuto XY středu a vezmeme jeho Z
        dists = np.linalg.norm(top_points[:, :2] - np.array([center_x, center_y]), axis=1)
        closest_idx = np.argmin(dists)
        center_z = top_points[closest_idx, 2]

        backrest_top_center = np.array([center_x, center_y, center_z])

        # Kraje (Left/Right - osa Y)
        # Najdeme bod s min Y a max Y v horním řezu
        left_idx = np.argmax(top_points[:, 1])
        right_idx = np.argmin(top_points[:, 1])
        backrest_left = top_points[left_idx]
        backrest_right = top_points[right_idx]
    else:
        backrest_top_center = np.array([0, 0, max_z])
        backrest_left = backrest_top_center
        backrest_right = backrest_top_center

    # 3. Nohy (Původní logika: spodních 5% + minima)
    bottom_threshold = min_z + (height * 0.05)
    bottom_indices = np.where(mesh.vertices[:, 2] < bottom_threshold)[0]
    foot_centers = []

    if len(bottom_indices) > 0:
        bottom_points = mesh.vertices[bottom_indices]
        # Clustering
        clustering = DBSCAN(eps=0.05, min_samples=3).fit(bottom_points)
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}

        for label in unique_labels:
            cluster = bottom_points[labels == label]
            # Vrátíme lokální minimum v Z pro každý cluster (dotyk se zemí)
            min_z_idx = np.argmin(cluster[:, 2])
            foot_centers.append(cluster[min_z_idx])

    return {
        "com": com,
        "top_center": backrest_top_center,
        "top_left": backrest_left,
        "top_right": backrest_right,
        "feet": foot_centers
    }

# --- URDF GENERÁTOR ---

def create_full_urdf(input_urdf_path, output_urdf_path, chair_root_name="base_link"):
    if not os.path.exists(input_urdf_path): return

    print(f"--- Zpracovávám: {input_urdf_path} ---")

    # A. Analýza geometrie (čistá židle)
    try:
        mesh = load_chair_geometry(input_urdf_path, chair_root_name)
    except Exception as e:
        print(f"Chyba při načítání geometrie: {e}")
        return

    features = analyze_chair_features(mesh)
    print(f"  -> CoM: {features['com']}")
    print(f"  -> Top Center: {features['top_center']}")

    # B. Sestavení nového URDF
    # Načteme původní elementy
    tree = ET.parse(input_urdf_path)
    orig_root = tree.getroot()

    # Vytvoříme nový kořen
    new_robot = ET.Element("robot", name=orig_root.get("name", "chair"))

    # 1. Přeneseme materiály
    for mat in orig_root.findall("material"):
        new_robot.append(mat)

    # 2. Vytvoříme "Podvozek" (Slide joints)
    # world -> slide_x -> slide_y -> rotate_z -> base_link

    def make_link(name, mass="0.01"):
        l = ET.SubElement(new_robot, "link", name=name)
        inertial = ET.SubElement(l, "inertial")
        ET.SubElement(inertial, "mass", value=mass)
        ET.SubElement(inertial, "inertia", ixx="0.001", ixy="0", ixz="0", iyy="0.001", iyz="0", izz="0.001")
        return l

    make_link("world", "100") # Static world anchor
    make_link("slide_link_x")
    make_link("slide_link_y")

    # Joint: World -> X
    j_x = ET.SubElement(new_robot, "joint", name="floor_slide_x", type="prismatic")
    ET.SubElement(j_x, "parent", link="world")
    ET.SubElement(j_x, "child", link="slide_link_x")
    ET.SubElement(j_x, "axis", xyz="1 0 0")
    ET.SubElement(j_x, "limit", lower="-10", upper="10", effort="1000", velocity="5")

    # Joint: X -> Y
    j_y = ET.SubElement(new_robot, "joint", name="floor_slide_y", type="prismatic")
    ET.SubElement(j_y, "parent", link="slide_link_x")
    ET.SubElement(j_y, "child", link="slide_link_y")
    ET.SubElement(j_y, "axis", xyz="0 1 0")
    ET.SubElement(j_y, "limit", lower="-10", upper="10", effort="1000", velocity="5")

    # Joint: Y -> Chair Base (Rotation)
    j_z = ET.SubElement(new_robot, "joint", name="floor_rotate_z", type="continuous")
    ET.SubElement(j_z, "parent", link="slide_link_y")
    ET.SubElement(j_z, "child", link=chair_root_name)
    ET.SubElement(j_z, "axis", xyz="0 0 1")
    ET.SubElement(j_z, "dynamics", damping="1.0")

    # 3. Přeneseme původní strukturu židle
    # Kopírujeme linky a jointy, ale vynecháme ty staré slide jointy pokud tam byly

    # Linky
    for link in orig_root.findall("link"):
        name = link.get("name")
        # Ignorujeme staré pomocné linky pokud se jmenují stejně jako naše nové
        if name in ["world", "slide_link_x", "slide_link_y"]: continue
        new_robot.append(link)

    # Jointy
    for joint in orig_root.findall("joint"):
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")

        # Ignorujeme staré připojení k world
        if parent == "world" or parent == "slide_link_y": continue
        # Ignorujeme pokud child je base_link (to už jsme vyřešili nahoře)
        if child == chair_root_name: continue

        new_robot.append(joint)

    # 4. Přidáme DEBUG MARKERY
    # DŮLEŽITÉ: Parent musí být chair_root_name (base_link), NE world!

    def add_marker(name, pos, color, radius, shape="sphere", length=None):
        l = ET.SubElement(new_robot, "link", name=name)
        v = ET.SubElement(l, "visual")
        ET.SubElement(v, "origin", xyz="0 0 0", rpy="0 0 0")
        g = ET.SubElement(v, "geometry")
        if shape == "sphere":
            ET.SubElement(g, "sphere", radius=str(radius))
        elif shape == "cylinder":
            ET.SubElement(g, "cylinder", radius=str(radius), length=str(length))

        m = ET.SubElement(v, "material", name=f"m_{name}")
        ET.SubElement(m, "color", rgba=color)

        # Joint připojený k BASE LINK židle
        j = ET.SubElement(new_robot, "joint", name=f"j_{name}", type="fixed")
        ET.SubElement(j, "parent", link=chair_root_name)
        ET.SubElement(j, "child", link=name)
        ET.SubElement(j, "origin", xyz=f"{pos[0]} {pos[1]} {pos[2]}", rpy="0 0 0")

    # Markers
    add_marker("debug_top_center", features['top_center'], "1 0 0 1", 0.04)
    add_marker("debug_top_left", features['top_left'], "1 1 0 1", 0.03)
    add_marker("debug_top_right", features['top_right'], "1 1 0 1", 0.03)
    add_marker("debug_com", features['com'], "0 0 1 1", 0.05)

    vec_h = 0.5
    vec_pos = features['com'] + [0,0,vec_h/2]
    add_marker("debug_vec", vec_pos, "0 0 1 0.5", 0.01, "cylinder", vec_h)

    for i, foot in enumerate(features['feet']):
        add_marker(f"debug_foot_{i}", foot, "0 1 0 1", 0.04)

    # Uložení
    ET.ElementTree(new_robot).write(output_urdf_path)
    print(f"Hotovo! Uloženo: {output_urdf_path}")

# --- CONFIG ---
# Zde si dej pozor, aby vstupní URDF bylo to "čisté" (bez tvých slide jointů),
# nebo alespoň aby base_link byl skutečně tělo židle.
# Pokud máš ve vstupu už slide jointy, můj skript se je pokusí odfiltrovat,
# za předpokladu, že base_link je jméno hlavního těla.

original_urdf = "/home/roboversepc/code/RoboVerse/roboverse_data/assets/humanoidbench/chair/urdf/chair.urdf"
debug_urdf = "/home/roboversepc/code/RoboVerse/roboverse_data/assets/humanoidbench/chair/urdf/chair_debug.urdf"

# Předpokládáme, že hlavní tělo se jmenuje "base_link"
create_full_urdf(original_urdf, debug_urdf, chair_root_name="base_link")
