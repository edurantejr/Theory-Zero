bl_info = {
    "name": "Phase 5 Blackhole Simulation Baker",
    "author": "ChatGPT / Eddie Durante Jr.",
    "version": (1, 4),
    "blender": (3, 0, 0),
    "location": "Object → Bake Blackhole Simulation",
    "description": "Bake & render the Phase 5 cosmic curvature multi‑singularity particle simulation with optional Phase 1 / Phase 2 validation.",
    "category": "Object",
}

import bpy
import mathutils
import json
import os


class OBJECT_OT_bake_blackhole(bpy.types.Operator):
    """Bake per‑frame particle motion, insert keyframes, optionally validate against reference JSON, and render to MP4."""

    bl_idname = "object.bake_blackhole_sim"
    bl_label = "Bake Blackhole Simulation"
    bl_options = {'REGISTER', 'UNDO'}

    # ───────────────────────── parameters ─────────────────────────
    start_frame: bpy.props.IntProperty(name="Start Frame", default=1, min=1)
    end_frame: bpy.props.IntProperty(name="End Frame", default=250, min=1)
    fps: bpy.props.IntProperty(name="Frame Rate", default=20, min=1)
    tolerance: bpy.props.FloatProperty(name="Validation Tolerance", default=1e-3)
    collection_name: bpy.props.StringProperty(name="Particle Collection", default="Particles")
    output_path: bpy.props.StringProperty(name="Output MP4 Path", default="//phase5_cosmic_curvature_fixed.mp4", subtype='FILE_PATH')

    # ───────────────────────── helpers ─────────────────────────
    @staticmethod
    def compute_force(pos: mathutils.Vector) -> mathutils.Vector:
        """Replace with your Phase 1–2 curvature / lensing force law."""
        r = pos.length
        if r == 0:
            return mathutils.Vector((0.0, 0.0, 0.0))
        return -pos.normalized() / (r * r)

    # ───────────────────────── main entry ─────────────────────────
    def execute(self, context):
        print("🔥 BakeBlackhole: execute() called")
        self.report({'INFO'}, "BakeBlackhole: starting…")

        scene = context.scene
        scene.render.fps = self.fps
        scene.frame_start = self.start_frame
        scene.frame_end = self.end_frame
        dt = 1.0 / float(self.fps)

        # — collect particles —
        coll = bpy.data.collections.get(self.collection_name)
        if not coll:
            self.report({'ERROR'}, f"Collection not found: {self.collection_name}")
            return {'CANCELLED'}
        particles = list(coll.objects)
        if not particles:
            self.report({'ERROR'}, f"No objects in collection: {self.collection_name}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Found {len(particles)} particles in '{self.collection_name}'")

        # — load reference JSONs (optional) —
        blend_dir = os.path.dirname(bpy.data.filepath)
        def load_json(fname):
            path = os.path.join(blend_dir, fname)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
            self.report({'WARNING'}, f"Reference file missing: {fname}")
            return {}
        ref1 = load_json("phase1_reference.json")
        ref2 = load_json("phase2_reference.json")

        # — bake loop —
        for frame in range(self.start_frame, self.end_frame + 1):
            scene.frame_set(frame)

            # advance simulation one step
            for obj in particles:
                obj.location += self.compute_force(obj.location) * dt

            # insert keyframes
            for obj in particles:
                obj.keyframe_insert(data_path="location", frame=frame)

            # validate (if reference present)
            key = str(frame)
            if key in ref1 and key in ref2:
                for idx, obj in enumerate(particles):
                    d1 = (obj.location - mathutils.Vector(ref1[key][idx])).length
                    d2 = (obj.location - mathutils.Vector(ref2[key][idx])).length
                    if max(d1, d2) > self.tolerance:
                        self.report({'INFO'}, f"Val fail F{frame} P{idx}: dev={max(d1, d2):.3e}")

        # — render —
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.filepath = bpy.path.abspath(self.output_path)
        try:
            bpy.ops.render.render(animation=True)
        except RuntimeError as err:
            self.report({'ERROR'}, f"Render failed: {err}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Rendered to {scene.render.filepath}")
        return {'FINISHED'}


# ───────────────────────── registration ─────────────────────────

def menu_func(self, context):
    self.layout.operator(OBJECT_OT_bake_blackhole.bl_idname)

def register():
    bpy.utils.register_class(OBJECT_OT_bake_blackhole)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    bpy.utils.unregister_class(OBJECT_OT_bake_blackhole)


if __name__ == "__main__":
    register()
