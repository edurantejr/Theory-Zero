import bpy, json, os

print("🚀  Auto-setup Phase 5 starting…")

# 1 – Hard-coded folder paths
project_dir = r"C:\Users\lcpld\Documents\Theory Zero Project\Black_Hole_Lab"
ref1_path   = os.path.join(project_dir, "phase1_reference.json")
print(" •  Looking for JSON at:", ref1_path)

# 2 – Load frame-1 positions
try:
    with open(ref1_path, "r") as f:
        positions = json.load(f).get("1", [])
    print(f" •  Loaded {len(positions)} position(s)")
except Exception as e:
    print(" ❌  Failed to load JSON:", e)
    positions = []

# 3 – Reset the scene
bpy.ops.wm.read_factory_settings(use_empty=True)
print(" •  Cleared default scene")

# 4 – Camera & light
cam = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam)
bpy.context.scene.collection.objects.link(cam_obj)
cam_obj.location = (0, -10, 5)
cam_obj.rotation_euler = (1.1, 0, 0)
bpy.context.scene.camera = cam_obj
print(" •  Camera created")

lt = bpy.data.lights.new("Light", type="POINT")
lt_obj = bpy.data.objects.new("Light", lt)
bpy.context.scene.collection.objects.link(lt_obj)
lt_obj.location = (5, 5, 5)
print(" •  Light created")

# 5 – Particles collection
pcoll = bpy.data.collections.get("Particles") or bpy.data.collections.new("Particles")
if pcoll.name not in bpy.context.scene.collection.children:
    bpy.context.scene.collection.children.link(pcoll)
# Clean any old objects
for o in list(pcoll.objects):
    pcoll.objects.unlink(o)
print(" •  'Particles' collection ready")

# 6 – Add spheres WITHOUT relying on context
for idx, pos in enumerate(positions):
    # Track existing object names
    before = set(bpy.data.objects.keys())
    # Add primitive sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=pos)
    # The new object is the one whose name wasn’t there before
    after  = set(bpy.data.objects.keys())
    new_name = (after - before).pop()
    obj = bpy.data.objects[new_name]
    # Unlink from default collection(s) and link to Particles
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    pcoll.objects.link(obj)
print(f" •  Created {len(positions)} sphere(s) in 'Particles'")

# 7 – Render output path
scene = bpy.context.scene
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.filepath = os.path.join(project_dir, "phase5_cosmic_curvature_fixed.mp4")
print(" •  Render path set to:", scene.render.filepath)

print("✅  Auto-setup complete—File → Save your .blend, then bake!")
