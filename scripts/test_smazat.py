from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("/home/roboversepc/code/RoboVerse/urdf2usd_convert/g1/usd/cube_2.usd")

# najdi root prim
root_prim = stage.GetPrimAtPath("/cube_2")  # cesta k prim, zkontroluj ve stage vieweru

if root_prim:
    articulation_attr = root_prim.GetAttribute("physxArticulation:articulationEnabled")
    if articulation_attr:
        articulation_attr.Set(False)

stage.GetRootLayer().Save()
