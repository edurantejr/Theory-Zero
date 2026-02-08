bl_info = {
    "name": "Phase 5 Blackhole Simulation Baker",
    "author": "ChatGPT / Eddie Durante Jr.",
    "version": (1, 3),
    "blender": (3, 0, 0),
    "location": "Object > Bake Blackhole Simulation",
    "description": "Bake & render the Phase 5 cosmic curvature multi-singularity particle simulation with validation vs Phases 1–2 (collection-based).",
    "category": "Object",
}

import bpy
import mathutils
import json
import os

class OBJECT_OT_bake_blackhole_sim(bpy.types.Operator):
    bl_idname = "object.bake_blackhole_sim"
    bl_label = "Bake Blackhole Simulation"
    bl_options = {'REGISTER', 'UNDO'}

    start_frame: bpy.props.IntProperty(
        name="Start Frame",
        default=1,
        min=1
    )
    end_frame: bpy.props.IntProperty(
        name="End Frame",
        default=250,
        min=1
    )
    fps: bpy.props.IntProperty(
        name="Frame Rate",
        default=20,
        min=1
    )
    tolerance: bpy.props.FloatProperty(
        name="Validation Tolerance",
        default=1e-3
    )
    collection_name: bpy.props.StringProperty(
        name="Particle Collection",
        default="Particles"
    )
    output_path: bpy.props.StringProperty(
        name="Output MP4 Path",
        default="//phase5_cosmic_curvature_fixed.mp4",
        subtype='FILE_PATH'
    )

    def compute_force(self, position: mathutils.Vector) -> mathutils.Vector:
        # Replace with your Phase 1–2 curvature/lensing force law
        r = position.length
        if r == 0:
            return mathutils.Vector((0.0, 0.0, 0.0))
        return - position.normalized() / (r * r)

    def execute(self, context):
        print("🔥 BakeBlackhole: execute() called")
        self.report({'INFO'}, "BakeBlackhole: starting…")

        scene = context.scene
        scene.render.fps = self.fps
        scene.frame_start = self.start_frame
        scene.frame_end = self.end_frame
        dt = 1.0 / self.fps

        # Collect particles
        coll = bpy.data.collections.get(self.collection_name)
        if not coll:
            self.report({'ERROR'}, f"Collection not found: {self.collection_name}")
            return {'CANCELLED'}
        particles = list(coll.objects)
        if not particles:
            self.report({'ERROR'}, f"No objects in collection: {self.collection_name}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Found {len(particles)} particles in '{self.collection_name}'")

        # Load reference data using Blender's relative path resolution
        # This uses the .blend's directory
        ref1_path = bpy.path.abspath("//phase1_reference.json")
        ref2_path = bpy.path.abspath("//phase2_reference.json")
        print(f"Loading Phase1 JSON from: {ref1_path}")
        print(f"Loading Phase2 JSON from: {ref2_path}")
        try:
            with open(ref1_path, 'r') as f:
                ref_phase1 = json.load(f)
        except Exception as e:
            self.report({'WARNING'}, f"Could not load {ref1_path}: {e}")
            ref_phase1 = {}
        try:
            with open(ref2_path, 'r') as f:
                ref_phase2 = json.load(f)
        except Exception as e:
            self.report({'WARNING'}, f"Could not load {ref2_path}: {e}")
            ref_phase2 = {}

        # Bake loop
        for frame in range(self.start_frame, self.end_frame + 1):
            scene.frame_set(frame)
            for obj in particles:
                force = self.compute_force(obj.location)
                obj.location += force * dt
            # validate
            key = str(frame)
            if key in ref_phase1 and key in ref_phase2:
                for idx, obj in enumerate(particles):
                    d1 = (obj.location - mathutils.Vector(ref_phase1[key][idx])).length
                    d2 = (obj.location - mathutils.Vector(ref_phase2[key][idx])).length
                    if max(d1, d2) > self.tolerance:
                        self.report({'INFO'}, f"Val fail F{frame} P{idx}: dev={max(d1,d2):.3e}")
            for obj in particles:
                obj.keyframe_insert(data_path="location", frame=frame)

        # Render
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.filepath = bpy.path.abspath(self.output_path)
        try:
            bpy.ops.render.render(animation=True)
        except RuntimeError as e:
            self.report({'ERROR'}, f"Render failed: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Rendered to {scene.render.filepath}")
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(OBJECT_OT_bake_blackhole_sim.bl_idname)


def register():
    bpy.utils.register_class(OBJECT_OT_bake_blackhole_sim)
    bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    bpy.utils.unregister_class(OBJECT_OT_bake_blackhole_sim)


if __name__ == "__main__":
    register()
