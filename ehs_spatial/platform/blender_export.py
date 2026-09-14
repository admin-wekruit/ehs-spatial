"""Immutable mesh exports. Asset-ID resolver only; no user paths or URLs."""
from __future__ import annotations

import gzip
import base64
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import Any, Callable, Mapping
import numpy as np

from .contracts import PlatformError, canonical, digest, validate_document
from .spatial import MeshData, affine, camera_intrinsics, primitive_mesh, transform_matrix, transform_points


def blender_camera_parameters(camera: Mapping[str, Any]) -> dict[str, float]:
    k = camera_intrinsics(camera["K"])
    affine(camera["cameraToWorld"], rigid=True)
    width, height = camera["width"], camera["height"]
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise PlatformError("invalid_image_dimensions")
    if abs(k[0, 1]) > 1e-8:
        raise PlatformError("camera_projection_unrepresentable", cameraId=camera["id"], reason="skew")
    aspect = float(k[0, 0]/k[1, 1])
    # Blender clamps each component below 1. Keep the ratio while expressing
    # both in its supported range, including source cameras where fx < fy.
    aspect_x,aspect_y = 1/min(1.,aspect),aspect/min(1.,aspect)
    if max(aspect_x,aspect_y) > 200:
        raise PlatformError("camera_projection_unrepresentable",cameraId=camera["id"],reason="pixel_aspect_range")
    parameters = {"lens": float(k[0, 0]*36/width), "sensorWidth": 36., "pixelAspectX":aspect_x,"pixelAspectY":aspect_y,
                  "shiftX": float((width/2-(k[0, 2]+.5))/width), "shiftY": float(((k[1, 2]+.5)-height/2)*aspect/width)}
    reconstructed = np.array([[parameters["lens"]*width/36,0,width/2-.5-parameters["shiftX"]*width],
                              [0,parameters["lens"]*width/36/aspect,height/2-.5+parameters["shiftY"]*width/aspect],[0,0,1]])
    conversion_error = float(np.max(np.abs(reconstructed-k)))
    if conversion_error > 1e-8:
        raise PlatformError("camera_intrinsic_conversion_failed",cameraId=camera["id"])
    # A numerical serialization gate, not reconstruction accuracy: Blender uses
    # float32 camera matrices/frusta. Bound the reopened projection to 16 float32
    # ULPs at source-image magnitude, with the existing .001 px minimum.
    magnitude = np.float32(max(width,height,float(k[0,0]),float(k[1,1])))
    parameters.update(intrinsicConversionErrorPixels=conversion_error,
        reopenTolerancePixels=max(.001,float(np.spacing(magnitude))*16),reopenToleranceFloat32Ulps=16)
    return parameters


def _array(data: bytes, offset: int, count: int, dtype: str, columns: int, stride: int | None = None) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    step = stride or columns*itemsize
    if any(type(x) is not int or x < 0 for x in (offset, count)) or step < columns*itemsize or offset + max(0,count-1)*step + (columns*itemsize if count else 0) > len(data):
        raise PlatformError("invalid_mesh_buffer")
    return np.ndarray((count, columns), dtype=dtype, buffer=data, offset=offset, strides=(step, itemsize)).copy()


def _read_glb(payload: bytes) -> tuple[dict[str, Any], bytes]:
    if len(payload) < 20 or struct.unpack_from("<4sII", payload) != (b"glTF", 2, len(payload)):
        raise PlatformError("invalid_glb")
    offset, document, binary = 12, None, b""
    while offset < len(payload):
        if offset+8 > len(payload):
            raise PlatformError("invalid_glb")
        length, kind = struct.unpack_from("<I4s", payload, offset)
        offset += 8
        if length % 4 or offset+length > len(payload):
            raise PlatformError("invalid_glb")
        chunk = payload[offset:offset+length]
        if kind == b"JSON":
            if document is not None:
                raise PlatformError("invalid_glb")
            document = json.loads(chunk)
        elif kind == b"BIN\x00":
            binary = chunk
        offset += length
    if not isinstance(document, dict) or any(b.get("uri") for b in document.get("buffers", [])) or document.get("extensionsRequired"):
        raise PlatformError("unsupported_glb_dependency")
    return document, binary


def _glb_accessor(spec: dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    accessor = spec["accessors"][index]
    if "sparse" in accessor or "bufferView" not in accessor:
        raise PlatformError("unsupported_glb_accessor")
    view = spec["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise PlatformError("unsupported_glb_dependency")
    dtypes = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    sizes = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    if accessor["componentType"] not in dtypes or accessor["type"] not in sizes:
        raise PlatformError("unsupported_glb_accessor")
    data = _array(binary, view.get("byteOffset", 0)+accessor.get("byteOffset", 0), accessor["count"], dtypes[accessor["componentType"]], sizes[accessor["type"]], view.get("byteStride"))
    if accessor.get("normalized"):
        if data.dtype.kind not in "iu":
            raise PlatformError("invalid_glb_normalization")
        data = np.maximum(data.astype(float)/np.iinfo(data.dtype).max, -1)
    return data


def mesh_from_asset(payload: bytes | MeshData, metadata: Mapping[str, Any]) -> MeshData:
    if isinstance(payload, MeshData):
        return payload
    if not isinstance(payload, bytes):
        raise PlatformError("invalid_mesh_asset")
    expected = metadata.get("sha256")
    if expected and hashlib.sha256(payload).hexdigest() != expected:
        raise PlatformError("asset_hash_mismatch")
    if metadata.get("contentEncoding") == "gzip":
        payload = gzip.decompress(payload)
    if metadata.get("format") == "panoptes-mesh-v1":
        layout = metadata["byteLayout"]
        vertices = _array(payload, layout.get("byteOffset", 0), layout["vertexCount"], "<f4", 9, 36)
        indices = _array(payload, layout["indexByteOffset"], layout["indexCount"], "<u4", 1).reshape(-1,3)
        return MeshData(vertices[:, :3], indices, vertices[:, 6:9])
    spec, binary = _read_glb(payload)
    vertices, faces, parts = [], [], []
    roots = spec.get("scenes", [{"nodes": list(range(len(spec.get("nodes", []))))}])[spec.get("scene", 0)].get("nodes", [])
    def visit(index: int, parent: np.ndarray, ancestors: set[int]):
        if index in ancestors:
            raise PlatformError("cyclic_glb_nodes")
        node = spec["nodes"][index]
        if "skin" in node:
            raise PlatformError("unsupported_glb_skin")
        local = affine(np.asarray(node["matrix"]).reshape(4,4).T) if "matrix" in node else transform_matrix({
            "coordinateFrameId": "asset", "position": node.get("translation", [0,0,0]),
            "quaternion": node.get("rotation", [0,0,0,1]), "scale": node.get("scale", [1,1,1])})
        pose = parent @ local
        if "mesh" in node:
            for primitive in spec["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("mode", 4) != 4 or primitive.get("targets") or primitive.get("extensions"):
                    raise PlatformError("unsupported_glb_primitive")
                v = _glb_accessor(spec, binary, primitive["attributes"]["POSITION"])
                f = _glb_accessor(spec, binary, primitive["indices"]).reshape(-1,3) if "indices" in primitive else np.arange(len(v),dtype=np.uint32).reshape(-1,3)
                if f.dtype.kind not in "iu":
                    raise PlatformError("invalid_glb_indices")
                material = spec.get("materials", [])[primitive["material"]] if "material" in primitive else {}
                pbr = material.get("pbrMetallicRoughness", {})
                if material.get("extensions") or any(k in material for k in ("normalTexture","occlusionTexture","emissiveTexture")) or "metallicRoughnessTexture" in pbr:
                    raise PlatformError("unsupported_pbr_texture_channel")
                uv,texture_bytes,mime,sampler = None,None,None,None
                if "baseColorTexture" in pbr:
                    tex = pbr["baseColorTexture"]
                    if tex.get("texCoord",0) != 0 or tex.get("extensions"):
                        raise PlatformError("unsupported_texture_coordinate_transform")
                    texture = spec["textures"][tex["index"]]
                    image = spec["images"][texture["source"]]
                    if image.get("uri") or image.get("mimeType") not in ("image/png","image/jpeg") or "bufferView" not in image:
                        raise PlatformError("unsupported_glb_texture_source")
                    view = spec["bufferViews"][image["bufferView"]]
                    if view.get("buffer",0) != 0:
                        raise PlatformError("unsupported_glb_dependency")
                    start = view.get("byteOffset",0)
                    texture_bytes = binary[start:start+view["byteLength"]]
                    if len(texture_bytes) != view["byteLength"]:
                        raise PlatformError("invalid_mesh_buffer")
                    mime = image["mimeType"]
                    sampler = spec.get("samplers",[])[texture["sampler"]] if "sampler" in texture else None
                    uv = _glb_accessor(spec,binary,primitive["attributes"]["TEXCOORD_0"])
                c = _glb_accessor(spec,binary,primitive["attributes"]["COLOR_0"]) if "COLOR_0" in primitive["attributes"] else None
                faces.append(f + sum(len(x) for x in vertices))
                transformed = transform_points(v, pose).astype(np.float32)
                vertices.append(transformed)
                parameters = {"name":material.get("name",""),"baseColorFactor":pbr.get("baseColorFactor",[1,1,1,1]),
                    "roughness":pbr.get("roughnessFactor",1.),"metallic":pbr.get("metallicFactor",1.),"doubleSided":material.get("doubleSided",False),
                    "alphaMode":material.get("alphaMode","OPAQUE"),"alphaCutoff":material.get("alphaCutoff",.5),"emissiveFactor":material.get("emissiveFactor",[0,0,0]),
                    "colorEncoding":"linear","textureSampler":sampler}
                parts.append(MeshData(transformed,f.astype(np.uint32),c,uv,texture_bytes,mime,parameters))
        for child in node.get("children", []):
            visit(child, pose, ancestors | {index})
    for root in roots:
        visit(root, np.eye(4), set())
    if not vertices:
        raise PlatformError("glb_has_no_mesh")
    if len(parts) == 1:
        return parts[0]
    return MeshData(np.concatenate(vertices).astype(np.float32),np.concatenate(faces).astype(np.uint32),primitives=tuple(parts))


def _export_parts(mesh, override, *, apply_color_tint=False):
    parts = []
    for part in mesh.primitives or (mesh,):
        material = {**(part.material or {}),**override}
        tint = override.get("color")
        if apply_color_tint and isinstance(tint, (list, tuple)) and len(tint) >= 3:
            if any(type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1 for value in tint[:3]):
                raise PlatformError("invalid_material_color", 422)
            factor = material.get("baseColorFactor", [1.,1.,1.,1.])
            material["baseColorFactor"] = [factor[i]*tint[i] for i in range(3)] + [factor[3]]
        parts.append({"vertices":np.asarray(part.vertices,dtype=np.float32).tolist(),"faces":part.faces.tolist(),
            "colors":part.colors.tolist() if part.colors is not None else None,"uv":part.uv.tolist() if part.uv is not None else None,
            "texture":base64.b64encode(part.texture_bytes).decode() if part.texture_bytes else None,"textureMimeType":part.texture_mime_type,"material":material})
    return parts


def prepare_export(revision_id: str, document: dict[str, Any], resolve_asset: Callable[[str], bytes | MeshData]) -> dict[str, Any]:
    validate_document(document)
    document = json.loads(canonical(document))
    assets = {x["id"]: x for x in document["assets"]}
    frames = {x["id"] for x in document["coordinateFrames"]}
    objects, unresolved, excluded = [], [], []
    for entity in document["entities"]:
        for rep in entity.get("representations", []):
            if document['schemaVersion'] == 2 and rep['kind'] in ('generated_mesh', 'primitive') and rep['id'] != entity.get('activeModelRepresentationId'):
                excluded.append({'entityId':entity['id'], 'representationId':rep['id'], 'reason':'source_model_candidate'})
                continue
            if entity.get("sourceContext") is True and rep["kind"] == "point_cloud":
                excluded.append({"entityId": entity["id"], "representationId": rep["id"], "reason": "point_cloud_context"})
                continue
            if rep["placementState"] != "confirmed" or rep["kind"] == "point_cloud":
                unresolved.append({"entityId": entity["id"], "representationId": rep["id"], "reason": "placement_unconfirmed" if rep["placementState"] != "confirmed" else "not_a_mesh"})
                continue
            if rep["kind"] == "primitive":
                mesh = primitive_mesh(rep["primitive"])
            else:
                asset_id = rep.get("assetId")
                if asset_id not in assets:
                    raise PlatformError("unreferenced_export_asset", assetId=asset_id)
                mesh = mesh_from_asset(resolve_asset(asset_id), assets[asset_id])
            pose = entity.get("currentModelTransform") if rep["kind"] in ("generated_mesh", "primitive") else None
            pose = pose or rep["transform"]
            if pose["coordinateFrameId"] not in frames:
                raise PlatformError("invalid_coordinate_frame")
            matrix = transform_matrix(pose)
            material = (rep.get("material") or {}) if document["schemaVersion"] == 2 else entity.get("material", rep.get("material", {}))
            objects.append({"entityId": entity["id"], "id": rep["id"], "label": entity.get("label", ""), "kind": rep["kind"],
                            "coordinateFrameId": pose["coordinateFrameId"], "transform": pose, "matrix": matrix.tolist(),
                            "vertices": np.asarray(mesh.vertices,dtype=np.float32).tolist(), "faces": mesh.faces.tolist(),
                            "colors": mesh.colors.tolist() if mesh.colors is not None else None,
                            "uv":mesh.uv.tolist() if mesh.uv is not None else None,
                            "texture":base64.b64encode(mesh.texture_bytes).decode() if mesh.texture_bytes else None,"textureMimeType":mesh.texture_mime_type,
                            "material": material, "visible": entity.get("visible", True),
                            "parts":_export_parts(mesh,material,apply_color_tint=document["schemaVersion"] == 2 and rep["kind"] in ("generated_mesh", "primitive")),
                            "primitive": rep.get("primitive"), "sourceRefs": rep.get("sourceRefs", [])})
        if not entity.get("representations"):
            unresolved.append({"entityId": entity["id"], "reason": "no_representation"})
    cameras = [{**c, "blender": blender_camera_parameters(c)} for c in document["cameras"]]
    return {"schemaVersion": 1, "sceneRevisionId": revision_id, "documentSha256": digest(document), "document": document,
            "objects": objects, "cameras": cameras, "unplacedEntities": unresolved, "excludedRepresentations": excluded, "newModelCalls": 0}


def write_glb(prepared: dict[str, Any], path: Path) -> dict[str, Any]:
    """Preserve camera K in extras: glTF cameras cannot encode shifted principal points."""
    binary = bytearray()
    spec = {"asset": {"version": "2.0", "generator": "Panoptes revision export"}, "scene": 0, "scenes": [], "nodes": [], "meshes": [], "materials": [], "buffers": [], "bufferViews": [], "accessors": [], "images":[],"textures":[],
            "extras": {"sceneRevisionId": prepared["sceneRevisionId"], "documentSha256": prepared["documentSha256"], "sourceDocument": prepared["document"], "cameraProjection": "source K/cameraToWorld preserved in sourceDocument; no approximate glTF cameras"}}
    def accessor(array: np.ndarray, kind: str, component: int, bounds: bool = False):
        while len(binary)%4:
            binary.append(0)
        offset = len(binary)
        binary.extend(array.tobytes())
        spec["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": array.nbytes})
        record = {"bufferView": len(spec["bufferViews"])-1, "componentType": component, "count": len(array), "type": kind}
        if bounds:
            record.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        spec["accessors"].append(record)
        return len(spec["accessors"])-1
    frame_nodes = {x["id"]: [] for x in prepared["document"]["coordinateFrames"]}
    for obj in prepared["objects"]:
        primitives = []
        for part in obj["parts"]:
            v,f = np.asarray(part["vertices"],dtype="<f4"),np.asarray(part["faces"],dtype="<u4")
            normals = np.zeros_like(v)
            face_normals = np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
            for axis in range(3):
                np.add.at(normals,f[:,axis],face_normals)
            normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
            attrs = {"POSITION":accessor(v,"VEC3",5126,True),"NORMAL":accessor(normals,"VEC3",5126)}
            material = part["material"]
            if part["colors"] is not None:
                color = np.asarray(part["colors"],dtype="<f4")
                if material.get("colorEncoding","srgb") != "linear":
                    color[:,:3] = np.where(color[:,:3] <= .04045,color[:,:3]/12.92,((color[:,:3]+.055)/1.055)**2.4)
                attrs["COLOR_0"] = accessor(color,"VEC"+str(color.shape[1]),5126)
            spec["materials"].append({"name":material.get("name",""),"pbrMetallicRoughness":{"baseColorFactor":material.get("baseColorFactor",[1.,1.,1.,1.]),
                "roughnessFactor":material.get("roughness",.75),"metallicFactor":material.get("metallic",0.)},"doubleSided":material.get("doubleSided",True),
                "alphaMode":material.get("alphaMode","OPAQUE"),"alphaCutoff":material.get("alphaCutoff",.5),"emissiveFactor":material.get("emissiveFactor",[0,0,0])})
            if part.get("texture"):
                attrs["TEXCOORD_0"] = accessor(np.asarray(part["uv"],dtype="<f4"),"VEC2",5126)
                while len(binary)%4:
                    binary.append(0)
                data = base64.b64decode(part["texture"],validate=True)
                spec["bufferViews"].append({"buffer":0,"byteOffset":len(binary),"byteLength":len(data)})
                binary.extend(data)
                spec["images"].append({"bufferView":len(spec["bufferViews"])-1,"mimeType":part["textureMimeType"]})
                texture = {"source":len(spec["images"])-1}
                if material.get("textureSampler"):
                    spec.setdefault("samplers",[]).append(material["textureSampler"])
                    texture["sampler"] = len(spec["samplers"])-1
                spec["textures"].append(texture)
                spec["materials"][-1]["pbrMetallicRoughness"]["baseColorTexture"] = {"index":len(spec["textures"])-1}
            primitives.append({"attributes":attrs,"indices":accessor(f.reshape(-1),"SCALAR",5125),"material":len(spec["materials"])-1})
        spec["meshes"].append({"name":obj["id"],"primitives":primitives})
        spec["nodes"].append({"name": obj["id"], "mesh": len(spec["meshes"])-1, "matrix": np.asarray(obj["matrix"]).T.ravel().tolist(),
                              "extras": {k: obj[k] for k in ("entityId", "id", "kind", "primitive", "sourceRefs", "visible", "coordinateFrameId")}})
        if obj["visible"]:
            frame_nodes[obj["coordinateFrameId"]].append(len(spec["nodes"])-1)
    spec["scenes"] = [{"name": frame, "nodes": nodes} for frame,nodes in frame_nodes.items()] or [{"name": "Unplaced assets", "nodes": []}]
    spec["buffers"] = [{"byteLength": len(binary)}]
    js = canonical(spec)
    js += b" "*((-len(js))%4)
    binary.extend(b"\0"*((-len(binary))%4))
    payload = struct.pack("<4sII",b"glTF",2,12+8+len(js)+8+len(binary)) + struct.pack("<I4s",len(js),b"JSON") + js + struct.pack("<I4s",len(binary),b"BIN\0") + binary
    path.write_bytes(payload)
    reopened, data = _read_glb(path.read_bytes())
    if reopened["extras"]["sourceDocument"] != prepared["document"]:
        raise PlatformError("glb_reopen_provenance_mismatch")
    for obj,node in zip(prepared["objects"],reopened["nodes"],strict=True):
        if node["matrix"] != np.asarray(obj["matrix"]).T.ravel().tolist():
            raise PlatformError("glb_reopen_geometry_mismatch")
        for part,primitive in zip(obj["parts"],reopened["meshes"][node["mesh"]]["primitives"],strict=True):
            if not np.array_equal(_glb_accessor(reopened,data,primitive["attributes"]["POSITION"]),np.asarray(part["vertices"],dtype=np.float32)) or not np.array_equal(_glb_accessor(reopened,data,primitive["indices"]).ravel(),np.asarray(part["faces"],dtype=np.uint32).ravel()):
                raise PlatformError("glb_reopen_geometry_mismatch")
            if part["colors"] is not None:
                expected_color = np.asarray(part["colors"],dtype=np.float32)
                if part["material"].get("colorEncoding","srgb") != "linear":
                    expected_color[:,:3] = np.where(expected_color[:,:3] <= .04045,expected_color[:,:3]/12.92,((expected_color[:,:3]+.055)/1.055)**2.4)
                if not np.array_equal(_glb_accessor(reopened,data,primitive["attributes"]["COLOR_0"]),expected_color):
                    raise PlatformError("glb_reopen_color_mismatch")
            if part.get("texture"):
                mat = reopened["materials"][primitive["material"]]
                image = reopened["images"][reopened["textures"][mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]]
                view = reopened["bufferViews"][image["bufferView"]]
                if data[view["byteOffset"]:view["byteOffset"]+view["byteLength"]] != base64.b64decode(part["texture"]) or not np.array_equal(_glb_accessor(reopened,data,primitive["attributes"]["TEXCOORD_0"]),np.asarray(part["uv"],dtype=np.float32)):
                    raise PlatformError("glb_reopen_texture_mismatch")
    return {"status": "passed", "objects": len(prepared["objects"]), "sourceCameras": len(prepared["cameras"]), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


BLENDER_SCRIPT = r'''
import json, sys, base64
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
root = Path(sys.argv[sys.argv.index('--')+1])
spec = json.loads((root/'input.json').read_text())
bpy.ops.wm.read_factory_settings(use_empty=True)
first, scenes = bpy.context.scene, {}
for i, frame in enumerate(spec['document']['coordinateFrames']):
    scene = first if i == 0 else bpy.data.scenes.new(frame['id'])
    scene.name = frame['id']
    scene['coordinate_frame_json'] = json.dumps(frame)
    scene['scene_revision_id'] = spec['sceneRevisionId']
    scale = frame['scale']
    scene.unit_settings.system = 'METRIC' if scale['status'] == 'operator_anchored' else 'NONE'
    if scale['status'] == 'operator_anchored':
        scene.unit_settings.scale_length = scale['nativeToMeters']
    scenes[frame['id']] = scene
for item in spec['objects']:
    scene = scenes[item['coordinateFrameId']]
    mesh = bpy.data.meshes.new(item['id'])
    mesh.from_pydata(item['vertices'], [], item['faces'])
    mesh.update()
    obj = bpy.data.objects.new(item['id'],mesh)
    scene.collection.objects.link(obj)
    obj.matrix_world = Matrix(item['matrix'])
    obj['entity_id'], obj['representation_id'] = item['entityId'], item['id']
    obj['source_refs_json'] = json.dumps(item['sourceRefs'])
    obj['native_transform_json'] = json.dumps(item['transform'])
    obj.hide_render = obj.hide_viewport = not item['visible']
    rgba_parts = []
    for part in item['parts']:
        rgb = np.asarray(part['colors']) if part['colors'] is not None else np.ones((len(part['vertices']),4))
        if rgb.shape[1] == 3:
            rgb = np.column_stack((rgb,np.ones(len(rgb))))
        if part['material'].get('colorEncoding','srgb') != 'linear':
            rgb[:,:3] = np.where(rgb[:,:3] <= .04045,rgb[:,:3]/12.92,((rgb[:,:3]+.055)/1.055)**2.4)
        rgba_parts.append(rgb)
    colors = mesh.color_attributes.new(name='SourceColor',type='FLOAT_COLOR',domain='POINT')
    colors.data.foreach_set('color',np.concatenate(rgba_parts).ravel())
    uv_layer = mesh.uv_layers.new(name='UVMap') if any(p.get('texture') for p in item['parts']) else None
    face_start,vertex_start = 0,0
    for part_index,part in enumerate(item['parts']):
        material = part['material']
        mat = bpy.data.materials.new(item['id']+' material '+str(part_index))
        mat['source_material_json'] = json.dumps(material)
        mat.use_nodes = True
        mat.use_backface_culling = not material.get('doubleSided',True)
        shader = mat.node_tree.nodes.get('Principled BSDF')
        color = material.get('baseColorFactor',[1,1,1,1])
        shader.inputs['Roughness'].default_value = material.get('roughness',.75)
        shader.inputs['Metallic'].default_value = material.get('metallic',0)
        shader.inputs['Emission Color'].default_value = material.get('emissiveFactor',[0,0,0])+[1]
        shader.inputs['Emission Strength'].default_value = 1
        attr = mat.node_tree.nodes.new('ShaderNodeVertexColor')
        attr.layer_name = 'SourceColor'
        multiply = mat.node_tree.nodes.new('ShaderNodeMixRGB')
        multiply.blend_type = 'MULTIPLY'
        multiply.inputs[0].default_value = 1
        multiply.inputs[2].default_value = color
        mat.node_tree.links.new(attr.outputs['Color'],multiply.inputs[1])
        color_socket = multiply.outputs[0]
        alpha = mat.node_tree.nodes.new('ShaderNodeMath')
        alpha.operation = 'MULTIPLY'
        alpha.inputs[1].default_value = color[3]
        mat.node_tree.links.new(attr.outputs['Alpha'],alpha.inputs[0])
        alpha_socket = alpha.outputs[0]
        if part.get('texture'):
            path = root/(item['id']+'-'+str(part_index)+('.png' if part['textureMimeType'] == 'image/png' else '.jpg'))
            path.write_bytes(base64.b64decode(part['texture']))
            image = bpy.data.images.load(str(path))
            image.pack()
            uv = np.asarray(part['uv'])
            for poly in list(mesh.polygons)[face_start:face_start+len(part['faces'])]:
                for loop_index in poly.loop_indices:
                    index = mesh.loops[loop_index].vertex_index-vertex_start
                    uv_layer.data[loop_index].uv = (uv[index,0],1-uv[index,1])
            tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
            tex.image = image
            sampler = material.get('textureSampler') or {}
            tex.extension = {33071:'EXTEND',33648:'MIRROR',10497:'REPEAT'}.get(sampler.get('wrapS',10497),'REPEAT')
            tex.interpolation = 'Closest' if sampler.get('magFilter') == 9728 else 'Linear'
            mix = mat.node_tree.nodes.new('ShaderNodeMixRGB')
            mix.blend_type = 'MULTIPLY'
            mix.inputs[0].default_value = 1
            mat.node_tree.links.new(color_socket,mix.inputs[1])
            mat.node_tree.links.new(tex.outputs['Color'],mix.inputs[2])
            color_socket = mix.outputs[0]
            opacity = mat.node_tree.nodes.new('ShaderNodeMath')
            opacity.operation = 'MULTIPLY'
            mat.node_tree.links.new(alpha_socket,opacity.inputs[0])
            mat.node_tree.links.new(tex.outputs['Alpha'],opacity.inputs[1])
            alpha_socket = opacity.outputs[0]
        mat.node_tree.links.new(color_socket,shader.inputs['Base Color'])
        mode = material.get('alphaMode','OPAQUE')
        if mode != 'OPAQUE':
            mat.surface_render_method = 'DITHERED'
            if mode == 'MASK':
                cutoff = mat.node_tree.nodes.new('ShaderNodeMath')
                cutoff.operation = 'GREATER_THAN'
                cutoff.inputs[1].default_value = material.get('alphaCutoff',.5)
                mat.node_tree.links.new(alpha_socket,cutoff.inputs[0])
                alpha_socket = cutoff.outputs[0]
            mat.node_tree.links.new(alpha_socket,shader.inputs['Alpha'])
        mesh.materials.append(mat)
        for polygon in list(mesh.polygons)[face_start:face_start+len(part['faces'])]:
            polygon.material_index = part_index
        face_start += len(part['faces'])
        vertex_start += len(part['vertices'])
    primitive = item.get('primitive')
    if primitive:
        obj['primitive_json'] = json.dumps(primitive)
        kind = primitive.get('type',primitive.get('kind'))
        parameters = [('radiusNative',primitive['radius']),('heightNative',primitive['height'])] if kind == 'cylinder' else [('dimension'+str(a),primitive['dimensions'][a]) for a in range(3)]
        for key,value in parameters:
            obj[key] = value
        for axis in range(3):
            key,initial = parameters[0 if axis < 2 else 1] if kind == 'cylinder' else parameters[axis]
            base = item['transform']['scale'][axis]
            driver = obj.driver_add('scale',axis).driver
            driver.type = 'SCRIPTED'
            variable = driver.variables.new()
            variable.name,variable.type = 'dimension','SINGLE_PROP'
            variable.targets[0].id = obj
            variable.targets[0].data_path = '["'+key+'"]'
            driver.expression = str(base)+'*dimension/'+str(initial)
for camera in spec['cameras']:
    scene = scenes[camera['coordinateFrameId']]
    data = bpy.data.cameras.new(camera['id'])
    params = camera['blender']
    data.sensor_fit, data.sensor_width = 'HORIZONTAL',36
    data.lens, data.shift_x, data.shift_y = params['lens'],params['shiftX'],params['shiftY']
    data.clip_start, data.clip_end = .001,1000000
    obj = bpy.data.objects.new(camera['id'],data)
    scene.collection.objects.link(obj)
    obj.matrix_world = Matrix(np.asarray(camera['cameraToWorld']) @ np.diag([1,-1,-1,1]))
    obj['source_camera_json'] = json.dumps(camera)
    if scene.camera is None:
        scene.camera = obj
        scene.render.resolution_x,scene.render.resolution_y = camera['width'],camera['height']
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x,scene.render.pixel_aspect_y = params['pixelAspectX'],params['pixelAspectY']
text = bpy.data.texts.new('Panoptes source revision.json')
# Blender's Text editor inserts very long lines quadratically. Keep the full
# source document as readable JSON lines; reopen still compares every value.
text.write(json.dumps(spec['document'],ensure_ascii=False,indent=1))
bpy.ops.wm.save_as_mainfile(filepath=str(root/'scene.blend'))
bpy.ops.wm.open_mainfile(filepath=str(root/'scene.blend'))
records, camera_records, parameter_records = [],[],[]
for item in spec['objects']:
    obj = next(o for o in bpy.data.objects if o.get('representation_id') == item['id'])
    scene = bpy.data.scenes[item['coordinateFrameId']]
    bpy.context.window.scene = scene
    bpy.context.view_layer.update()
    actual = np.array([v.co[:] for v in obj.data.vertices])
    expected = np.asarray(item['vertices'])
    faces = np.array([p.vertices[:] for p in obj.data.polygons])
    assert np.array_equal(actual,expected),item['id']
    assert np.array_equal(faces,np.asarray(item['faces'])),item['id']
    a,b = np.asarray(obj.matrix_world),np.asarray(item['matrix'])
    residual = float(np.max(np.abs(actual@a[:3,:3].T+a[:3,3] - (expected@b[:3,:3].T+b[:3,3]))))
    assert residual < 5e-5,(item['id'],residual)
    assert obj['entity_id'] == item['entityId']
    assert obj.hide_viewport == (not item['visible'])
    assert len(obj.data.materials) == len(item['parts'])
    expected_colors = []
    for part in item['parts']:
        rgb = np.asarray(part['colors']) if part['colors'] is not None else np.ones((len(part['vertices']),4))
        if rgb.shape[1] == 3:
            rgb = np.column_stack((rgb,np.ones(len(rgb))))
        if part['material'].get('colorEncoding','srgb') != 'linear':
            rgb[:,:3] = np.where(rgb[:,:3] <= .04045,rgb[:,:3]/12.92,((rgb[:,:3]+.055)/1.055)**2.4)
        expected_colors.append(rgb)
    stored_colors = np.array([c.color[:] for c in obj.data.color_attributes['SourceColor'].data])
    color_error = float(np.max(np.abs(stored_colors-np.concatenate(expected_colors))))
    assert color_error < 1e-6,(item['id'],color_error)
    face_start,vertex_start = 0,0
    for part_index,part in enumerate(item['parts']):
        material = obj.data.materials[part_index]
        assert json.loads(material['source_material_json']) == part['material']
        shader = material.node_tree.nodes.get('Principled BSDF')
        assert abs(shader.inputs['Roughness'].default_value-part['material'].get('roughness',.75)) < 1e-6
        assert abs(shader.inputs['Metallic'].default_value-part['material'].get('metallic',0)) < 1e-6
        polys = list(obj.data.polygons)[face_start:face_start+len(part['faces'])]
        assert all(p.material_index == part_index for p in polys)
        if part.get('texture'):
            image = next(n.image for n in material.node_tree.nodes if n.type == 'TEX_IMAGE')
            assert image.packed_file is not None
            assert bytes(image.packed_file.data) == base64.b64decode(part['texture'])
            for polygon in polys:
                for loop_index in polygon.loop_indices:
                    index = obj.data.loops[loop_index].vertex_index-vertex_start
                    expected_uv = part['uv'][index]
                    assert np.allclose(obj.data.uv_layers.active.data[loop_index].uv,[expected_uv[0],1-expected_uv[1]],atol=1e-6)
        face_start += len(part['faces'])
        vertex_start += len(part['vertices'])
    records.append({'entityId':item['entityId'],'representationId':item['id'],'vertices':len(actual),'triangles':len(faces),'materials':len(item['parts']),'maxWorldErrorNative':residual,'maxLinearColorError':color_error})
    if item.get('primitive'):
        original_scale = np.array(obj.scale)
        key = 'radiusNative' if 'radiusNative' in obj else 'dimension0'
        original = obj[key]
        obj[key] = original*1.1
        obj.update_tag()
        bpy.context.view_layer.update()
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        assert abs(evaluated.scale[0]-original_scale[0]*1.1) < 1e-5
        obj[key] = original
        obj.update_tag()
        bpy.context.view_layer.update()
        parameter_records.append({'representationId':item['id'],'driverEdit':'passed'})
for camera in spec['cameras']:
    scene = bpy.data.scenes[camera['coordinateFrameId']]
    bpy.context.window.scene = scene
    obj = bpy.data.objects[camera['id']]
    scene.camera = obj
    scene.render.resolution_x,scene.render.resolution_y = camera['width'],camera['height']
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x,scene.render.pixel_aspect_y = camera['blender']['pixelAspectX'],camera['blender']['pixelAspectY']
    bpy.context.view_layer.update()
    k,c2w = np.asarray(camera['K']),np.asarray(camera['cameraToWorld'])
    residuals = []
    for p in [[-.2,-.25,1],[.25,.15,1],[0,0,2],[.4,-.3,3]]:
        p = np.asarray(p)
        world = c2w[:3,:3]@p+c2w[:3,3]
        ndc = world_to_camera_view(scene,obj,Vector(world))
        actual = np.array([ndc.x*camera['width']-.5,(1-ndc.y)*camera['height']-.5])
        expected = (k@p)[:2]/p[2]
        residuals.append(float(np.max(np.abs(actual-expected))))
    assert max(residuals) < camera['blender']['reopenTolerancePixels'],(camera['id'],residuals)
    camera_records.append({'cameraId':camera['id'],'maxConversionErrorPixels':max(residuals),
        'intrinsicConversionErrorPixels':camera['blender']['intrinsicConversionErrorPixels'],
        'reopenTolerancePixels':camera['blender']['reopenTolerancePixels'],
        'toleranceBasis':'16 float32 ULPs at source-image magnitude; minimum .001 px',
        'meaning':'Numerical export/reopen consistency, not physical reconstruction accuracy'})
assert json.loads(bpy.data.texts['Panoptes source revision.json'].as_string()) == spec['document']
(root/'blender-validation.json').write_text(json.dumps({'status':'passed','sceneRevisionId':spec['sceneRevisionId'],'objects':records,'cameras':camera_records,'parameters':parameter_records,'newModelCalls':0}))
'''


def export_scene_revision(revision_id: str, document: dict[str, Any], resolve_asset: Callable[[str], bytes | MeshData], output_dir: str | Path, blender_executable: str | Path | None = None) -> dict[str, Any]:
    prepared = prepare_export(revision_id, document, resolve_asset)
    executable = str(blender_executable) if blender_executable else shutil.which("blender")
    if not executable:
        raise PlatformError("blender_worker_unavailable", 503)
    destination = Path(output_dir)
    if destination.exists():
        raise PlatformError("export_destination_exists", 409)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".blender-export-",dir=destination.parent) as temporary:
        root = Path(temporary)
        (root/"input.json").write_bytes(canonical(prepared))
        (root/"export.py").write_text(BLENDER_SCRIPT)
        glb_validation = write_glb(prepared,root/"scene.glb")
        try:
            completed = subprocess.run([executable,"--background","--factory-startup","--python-exit-code","1","--python",str(root/"export.py"),"--",str(root)],capture_output=True,text=True,timeout=300,check=False)
        except subprocess.TimeoutExpired:
            raise PlatformError("blender_export_timeout", 504) from None
        except OSError:
            raise PlatformError("blender_worker_unavailable", 503) from None
        if completed.returncode or not (root/"blender-validation.json").exists():
            raise PlatformError("blender_export_validation_failed", 422)
        blend_validation = json.loads((root/"blender-validation.json").read_text())
        manifest = {"sceneRevisionId": revision_id,"documentSha256": prepared["documentSha256"],"status":"incomplete" if prepared["unplacedEntities"] else "succeeded", "newModelCalls":0,
                    "unplacedEntities":prepared["unplacedEntities"],"excludedRepresentations":prepared["excludedRepresentations"],"validation":{"blender":blend_validation,"glb":glb_validation},"files":[]}
        for name in ("scene.glb","scene.blend"):
            path = root/name
            manifest["files"].append({"name":name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size})
        (root/"manifest.json").write_bytes(canonical(manifest))
        (root/"input.json").unlink()
        (root/"export.py").unlink()
        for texture in root.glob("*.png"):
            texture.unlink()
        for texture in root.glob("*.jpg"):
            texture.unlink()
        backup = root/"scene.blend1"
        if backup.exists():
            backup.unlink()
        # Only fully verified outputs become visible; revision roots are immutable.
        root.rename(destination)
    return manifest
