"""Build self-contained OmniGibson-ready USDs from NVIDIA YCB meshes (layout mirrors bottle_v3.usd)."""
import sys
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf

src_path, out_path, name, mass = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
# Optional rotation about X (degrees) applied to the meshes, e.g. -90 to stand a Y-up YCB model upright.
rotate_x = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
# Optional target size "x,y,z" in metres (after rotation): the meshes are scaled per axis to match it,
# e.g. measurements of the real object.
target_size = [float(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else None
src = Usd.Stage.Open(src_path)
mesh_src = next(p for p in src.Traverse() if p.GetTypeName() == "Mesh")

out = Usd.Stage.CreateNew(out_path)
UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(out, 1.0)
root = UsdGeom.Xform.Define(out, f"/{name}")
out.SetDefaultPrim(root.GetPrim())
link = UsdGeom.Xform.Define(out, f"/{name}/base_link")
UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(mass)

src_layer, dst_layer = src.GetRootLayer(), out.GetRootLayer()
Sdf.CopySpec(src_layer, "/Root/Looks", dst_layer, f"/{name}/Looks")
Sdf.CopySpec(src_layer, mesh_src.GetPath(), dst_layer, f"/{name}/base_link/visual")
Sdf.CopySpec(src_layer, mesh_src.GetPath(), dst_layer, f"/{name}/base_link/collider")

material = UsdShade.Material(out.GetPrimAtPath(f"/{name}/Looks/OmniPBR"))
visual = out.GetPrimAtPath(f"/{name}/base_link/visual")
UsdShade.MaterialBindingAPI.Apply(visual).Bind(material)

collider = out.GetPrimAtPath(f"/{name}/base_link/collider")
collider.RemoveAPI(UsdShade.MaterialBindingAPI)
collider.RemoveProperty("material:binding")
UsdGeom.Imageable(collider).CreatePurposeAttr(UsdGeom.Tokens.guide)
UsdPhysics.CollisionAPI.Apply(collider)
UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr("convexDecomposition")
# PhysX's default decomposition (32 hulls, 10% error) half-fills thin-walled containers like a mug and
# closes the finger rings of scissors, so objects collide with space that looks empty. Authored by
# attribute name because PhysxSchema is not importable from the standalone USD libs.
collider.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
for attr_name, value_type, value in (
        ("physxConvexDecomposition:maxConvexHulls", Sdf.ValueTypeNames.Int, 128),
        ("physxConvexDecomposition:voxelResolution", Sdf.ValueTypeNames.Int, 2_000_000),
        ("physxConvexDecomposition:errorPercentage", Sdf.ValueTypeNames.Float, 1.0),
        ("physxConvexDecomposition:hullVertexLimit", Sdf.ValueTypeNames.Int, 64),
        ("physxConvexDecomposition:minThickness", Sdf.ValueTypeNames.Float, 0.001),
        ("physxConvexDecomposition:shrinkWrap", Sdf.ValueTypeNames.Bool, True)):
    collider.CreateAttribute(attr_name, value_type).Set(value)

if rotate_x:
    for mesh_name in ("visual", "collider"):
        xformable = UsdGeom.Xformable(out.GetPrimAtPath(f"/{name}/base_link/{mesh_name}"))
        ops = xformable.GetOrderedXformOps()
        rot = xformable.AddRotateXOp(opSuffix="upright")
        rot.Set(rotate_x)
        xformable.SetXformOpOrder([rot] + ops)

if target_size:
    # OmniGibson ignores extra scale ops on meshes, so fold the fit into the mesh's own xformOp:scale.
    # That scale acts before the upright rotation, which for +-90 deg about X swaps the y and z axes.
    assert rotate_x in (0.0, 90.0, -90.0), "target size only supports rotate_x of 0 or +-90"
    size = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange().GetSize()
    fx, fy, fz = [t / s for t, s in zip(target_size, size)]
    local_fit = (fx, fz, fy) if rotate_x else (fx, fy, fz)
    for mesh_name in ("visual", "collider"):
        prim = out.GetPrimAtPath(f"/{name}/base_link/{mesh_name}")
        scale_attr = prim.GetAttribute("xformOp:scale")
        old = scale_attr.Get()
        scale_attr.Set(type(old)(*[o * f for o, f in zip(old, local_fit)]))

out.GetRootLayer().Save()
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
print(out_path, "size", bb.GetSize(), "min", bb.GetMin(), "max", bb.GetMax(), "defaultPrim", out.GetDefaultPrim().GetPath())
for p in out.Traverse():
    print("  ", p.GetPath(), p.GetTypeName(), list(p.GetAppliedSchemas()))
