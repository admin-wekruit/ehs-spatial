"""On-demand measurements of the actual posed model surfaces; no compliance verdicts."""
from __future__ import annotations

import heapq
import itertools
import json
import time
import numpy as np
from shapely import polygons, union_all, get_parts
from shapely.geometry import Polygon

from .contracts import PlatformError
from .spatial import transform_matrix, transform_points, primitive_mesh, affine
from .blender_export import mesh_from_asset


def fitted_plane(triangles):
    """Deterministic area-weighted dominant surface fit, including thin double-sided panels."""
    t = np.asarray(triangles, float)
    crosses = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    area = np.linalg.norm(crosses, axis=1)/2
    valid = area > 1e-15
    t, area, crosses = t[valid], area[valid], crosses[valid]
    if not len(t):
        raise PlatformError('measurement_no_surface', 422)
    centers = t.mean(1); normals = crosses/(2*area[:, None])
    span = float(np.linalg.norm(np.ptp(t.reshape(-1, 3), axis=0)))
    tolerance = max(span*.01, 1e-9)
    # ponytail: 64 deterministic area quantiles bound candidate cost; this selects
    # one dominant flat face, not every face of arbitrary curved machinery.
    samples = np.unique(np.searchsorted(np.cumsum(area), (np.arange(64)+.5)/64*area.sum()))
    best = None; score = 0
    for i in samples:
        support = (np.abs(np.einsum('ij,j->i', normals, normals[i])) >= np.cos(np.deg2rad(15))) & (np.abs(np.einsum('ij,j->i', centers-centers[i], normals[i])) <= tolerance)
        covered = area[support].sum()
        if covered > score:
            best, score = support, covered
    fraction = float(score/area.sum())
    if fraction < .2:
        raise PlatformError('measurement_no_stable_plane', 422)
    faces, weights = t[best], area[best]/score
    sums = faces.sum(1); center = (weights[:, None]*sums/3).sum(0)
    moment = (np.einsum('n,nki,nkj->ij', weights, faces, faces)+np.einsum('n,ni,nj->ij', weights, sums, sums))/12
    values, axes = np.linalg.eigh(moment-np.outer(center, center))
    if values[1] <= 1e-15 or values[1]/max(values[2], 1e-15) < .01 or max(values[0], 0)/values[1] > .02:
        raise PlatformError('measurement_no_stable_plane', 422)
    normal = axes[:, 0]; points = faces.reshape(-1, 3)
    local = (points-center)@axes[:, 1:]
    lo, hi = local.min(0), local.max(0)
    outline = [center+axes[:, 1]*x+axes[:, 2]*y for x, y in [(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1]),(lo[0],lo[1])]]
    return dict(center=center, normal=normal, outline=np.asarray(outline), areaFraction=fraction,
                residual=float(np.sqrt(max(values[0], 0))), span=span)


def _point_triangle(p, triangle):
    a,b,c = triangle; ab,ac = b-a,c-a; n = np.cross(ab,ac); nn = n@n
    candidates = []
    if nn > 1e-24:
        q = p-n*((p-a)@n/nn)
        signs = [np.cross(y-x,q-x)@n for x,y in [(a,b),(b,c),(c,a)]]
        if min(signs) >= -1e-12*nn:
            candidates.append(q)
    for x,y in [(a,b),(b,c),(c,a)]:
        v=y-x; candidates.append(x+v*np.clip((p-x)@v/max(v@v,1e-30),0,1))
    q = min(candidates,key=lambda q: (p-q)@(p-q))
    return p,q


def _segments(a,b,c,d):
    u,v,w=b-a,d-c,a-c; aa,bb,cc,dd,ee=u@u,u@v,v@v,u@w,v@w
    den=aa*cc-bb*bb
    s=np.clip((bb*ee-cc*dd)/den,0,1) if den>1e-24 else 0.
    t=np.clip((bb*s+ee)/max(cc,1e-30),0,1)
    s=np.clip((bb*t-dd)/max(aa,1e-30),0,1)
    t=np.clip((bb*s+ee)/max(cc,1e-30),0,1)
    return a+s*u,c+t*v


def _triangle_pair(a,b):
    # Edge/face crossings can occur without any vertex lying on the other face.
    for source,target in [(a,b),(b,a)]:
        n=np.cross(target[1]-target[0],target[2]-target[0])
        for x,y in zip(source,np.roll(source,-1,axis=0)):
            den=(y-x)@n
            if abs(den)>1e-20:
                t=(target[0]-x)@n/den
                if 0<=t<=1:
                    p=x+t*(y-x); _,q=_point_triangle(p,target)
                    if np.linalg.norm(p-q)<1e-10:
                        return p,p
    pairs=[_point_triangle(p,b) for p in a]+[(q,p) for p,q in (_point_triangle(p,a) for p in b)]
    pairs += [_segments(x,y,u,v) for x,y in zip(a,np.roll(a,-1,axis=0)) for u,v in zip(b,np.roll(b,-1,axis=0))]
    return min(pairs,key=lambda pair: np.sum((pair[0]-pair[1])**2))


def surface_distance(a,b, *, seconds=15):
    """Exact triangle distance with an AABB hierarchy; no vertex-only approximation."""
    started=time.monotonic()
    class Node:
        def __init__(self,t):
            self.t=t; self.lo=t.min((0,1)); self.hi=t.max((0,1)); self.children=None
        def split(self):
            if self.children is None:
                order=np.argsort(self.t.mean(1)[:,np.argmax(self.hi-self.lo)],kind='stable'); mid=len(order)//2
                self.children=[Node(self.t[order[:mid]]),Node(self.t[order[mid:]])]
            return self.children
    def lower(x,y):
        delta=np.maximum(0,np.maximum(x.lo-y.hi,y.lo-x.hi)); return float(delta@delta)
    x,y=Node(a),Node(b); pair=_triangle_pair(a[0],b[0]); best=float(np.sum((pair[0]-pair[1])**2))
    counter=itertools.count(); queue=[(lower(x,y),next(counter),x,y)]; visited=0
    while queue:
        bound,_,x,y=heapq.heappop(queue)
        if bound>=best: continue
        visited+=1
        if visited%128==0 and time.monotonic()-started>seconds:
            raise PlatformError('measurement_complexity_limit',422)
        if len(x.t)<=4 and len(y.t)<=4:
            for ta in x.t:
                for tb in y.t:
                    p,q=_triangle_pair(ta,tb); distance=float((p-q)@(p-q))
                    if distance<best: best,pair=distance,(p,q)
                    if best<=1e-20: return 0.,pair
        else:
            pairs=[(c,y) for c in x.split()] if len(x.t)>=len(y.t) else [(x,c) for c in y.split()]
            for nx,ny in pairs:
                distance=lower(nx,ny)
                if distance<best: heapq.heappush(queue,(distance,next(counter),nx,ny))
    return float(np.sqrt(best)),pair


def measure_scene(revision, kind, entity_a, entity_b, region, load_asset):
    if kind not in ('angle','distance','occupancy'):
        raise PlatformError('measurement_kind_invalid',422)
    doc=revision['document']; models=[]; refs=[]
    ids=[entity_a] if kind=='occupancy' else [entity_a,entity_b]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise PlatformError('measurement_choose_objects',422)
    for id in ids:
        e=next((e for e in doc['entities'] if e['id']==id and not e.get('sourceContext') and e.get('visible',True)),None)
        if not e: raise PlatformError('measurement_object_missing',422)
        r=next((r for r in e.get('representations',[]) if r['id']==e.get('activeModelRepresentationId') and r['kind'] in ('primitive','generated_mesh')),None)
        if not r or r.get('sourceValidity')=='stale': raise PlatformError('measurement_model_missing',422)
        pose=e.get('currentModelTransform') or r['transform']
        if pose['coordinateFrameId']!=r['coordinateFrameId']: raise PlatformError('measurement_frame_mismatch',422)
        asset=next((a for a in doc['assets'] if a['id']==r.get('assetId')),None)
        if r['kind']=='generated_mesh' and not asset: raise PlatformError('measurement_model_missing',422)
        mesh=primitive_mesh(r['primitive']) if r['kind']=='primitive' else mesh_from_asset(load_asset(r['assetId']),asset or {})
        triangles=transform_points(mesh.vertices,transform_matrix(pose))[mesh.faces]
        if len(triangles)>500_000: raise PlatformError('measurement_complexity_limit',422)
        models.append((triangles,pose['coordinateFrameId']))
        refs.append({'entityId':id,'representationId':r['id'],'assetId':r.get('assetId'),'assetSha256':(asset or {}).get('sha256'),'placementState':r.get('placementState'),'qualityStatus':(r.get('qualityEvidence') or {}).get('status')})
    frame_id=models[0][1]
    frame=next((f for f in doc['coordinateFrames'] if f['id']==frame_id),None)
    if frame is None: raise PlatformError('measurement_frame_mismatch',422)
    if any(frame!=frame_id for _,frame in models): raise PlatformError('measurement_frame_mismatch',422)
    result={'revisionId':revision['id'],'kind':kind,'coordinateFrameId':frame_id,'source':'model_inference','references':refs,'lines':[],'quality':{},'unit':'native','value':None}
    if kind=='angle':
        a,b=[fitted_plane(t) for t,_ in models]
        cosine=float(np.clip(abs(a['normal']@b['normal']),0,1)); angle=float(np.degrees(np.arccos(cosine)))
        result.update(value=angle,unit='deg',method='dominant-surface-plane-acute-angle-v1')
        result['quality']={'surfaceFits':[{'areaFraction':p['areaFraction'],'rmsResidualNative':p['residual']} for p in (a,b)]}
        result['lines']=[{'points':p['outline'].tolist(),'color':color} for p,color in [(a,'#e36b23'),(b,'#168bba')]]
        normal=b['normal'] if a['normal']@b['normal']>=0 else -b['normal']
        origin=(a['center']+b['center'])/2; size=min(a['span'],b['span'])*.25
        # The arc represents plane-normal separation (acute angle), not an
        # inferred hinge interior angle; UI names that convention explicitly.
        tangent=normal-a['normal']*cosine; length=np.linalg.norm(tangent)
        if length>1e-9:
            arc=[origin+size*(a['normal']*np.cos(t)+tangent/length*np.sin(t)) for t in np.linspace(0,np.deg2rad(angle),25)]
            result['lines'].append({'points':[origin.tolist(),arc[0].tolist(),*[p.tolist() for p in arc],origin.tolist()],'color':'#edbe38'})
        result['labelPoint']=origin.tolist()
    elif kind=='distance':
        distance,(a,b)=surface_distance(models[0][0],models[1][0])
        result.update(value=distance,method='mesh-triangle-surface-distance-v1',labelPoint=((a+b)/2).tolist(),lines=[{'points':[a.tolist(),b.tolist()],'color':'#e36b23'}])
    else:
        try:
            data=json.loads(region) if isinstance(region,str) else region
            points=np.asarray(data['points'],float); plane=affine(data['nativeToPlane'],rigid=True)
            if data['coordinateFrameId']!=frame_id or points.ndim!=2 or points.shape[1]!=2 or not 3<=len(points)<=64 or not np.isfinite(points).all(): raise ValueError()
            area=Polygon(points)
            if not area.is_valid or area.area<=1e-12: raise ValueError()
        except (KeyError,TypeError,ValueError,PlatformError):
            raise PlatformError('measurement_region_invalid',422)
        normal=np.asarray((frame.get('ground') or {}).get('normal',[]),float)
        if normal.shape!=(3,) or np.linalg.norm(normal)<1e-8 or abs(abs(normal@plane[2,:3]/np.linalg.norm(normal))-1)>1e-6:
            raise PlatformError('measurement_ground_missing',422)
        triangles=transform_points(models[0][0].reshape(-1,3),plane).reshape(-1,3,3)[:,:,:2]
        ab,ac=triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]
        mask=abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0])>1e-16
        footprint=union_all(polygons(triangles[mask])); overlap=footprint.intersection(area)
        result.update(value=float(overlap.area),unit='native2',method='mesh-ground-projection-region-overlap-v1')
        result['quality']={'regionAreaNative2':float(area.area),'regionFraction':float(overlap.area/area.area),'regionSource':'manual_temporary'}
        inverse=np.linalg.inv(plane)
        def world(ring): return transform_points(np.column_stack([np.asarray(ring),np.zeros(len(ring))]),inverse).tolist()
        result['lines']=[{'points':world(area.exterior.coords),'color':'#168bba'}]
        for part in get_parts(overlap):
            if part.geom_type=='Polygon':
                result['lines'].append({'points':world(part.exterior.coords),'color':'#e36b23'})
                result['lines'].extend({'points':world(r.coords),'color':'#e36b23'} for r in part.interiors)
        result['labelPoint']=world([area.centroid.coords[0]])[0]
    return result


def register_measurement_routes(app, get_revision, load_asset):
    from fastapi import Query
    @app.get('/api/revisions/{revision_id}/measurements')
    def measurement(revision_id: str, kind: str, entityA: str, entityB: str | None = None,
                    region: str | None = Query(default=None, max_length=12000)):
        revision=get_revision(revision_id)
        if revision is None: raise PlatformError('revision_not_found',404)
        return measure_scene(revision,kind,entityA,entityB,region,load_asset)
