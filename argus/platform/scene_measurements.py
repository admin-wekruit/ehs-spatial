"""On-demand measurements of the actual posed model surfaces; no compliance verdicts."""
from __future__ import annotations

import heapq
import itertools
import json
import time
import numpy as np
from shapely import polygons, union_all, get_parts
from shapely.geometry import Polygon

from argus.platform.contracts import DocumentDTO, PlatformError, digest
from argus.platform.spatial import transform_matrix, transform_points, primitive_mesh, affine
from argus.platform.blender_export import mesh_from_asset


def fitted_plane(triangles, *, narrow=False):
    """Deterministic area-weighted dominant surface fit, including thin double-sided panels."""
    t = np.asarray(triangles, float)
    crosses = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    area = np.linalg.norm(crosses, axis=1)/2
    valid = area > 1e-15
    t, area, crosses = t[valid], area[valid], crosses[valid]
    if not len(t):
        raise PlatformError('measurement_no_surface', 422)
    centers = t.mean(1); normals = crosses/(2*area[:, None])
    centroid = (centers*area[:, None]).sum(0)/area.sum()
    span = float(2*np.linalg.norm(t.reshape(-1, 3)-centroid, axis=1).max())
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
    if narrow:
        # A solid sheet has two parallel skins. Fit their common orientation
        # within each skin so material thickness is not counted as curvature.
        offsets = (faces.mean(1)-center)@axes[:, 0]
        means = np.array([offsets.min(), offsets.max()])
        for _ in range(16):
            layers = np.abs(offsets[:, None]-means).argmin(1)
            masses = np.array([weights[layers == j].sum() for j in (0, 1)])
            if masses.min() < .15: break
            means = np.array([np.average(offsets[layers == j], weights=weights[layers == j]) for j in (0, 1)])
        scatter = float(np.sum(weights*(offsets-means[layers])**2))
        if masses.min() >= .15 and abs(means[1]-means[0]) > 6*np.sqrt(scatter):
            covariances = []; skin_centers = []; skin_normals = []
            for j in (0, 1):
                skin = faces[layers == j]; w = weights[layers == j]/masses[j]
                s = skin.sum(1); c = (w[:, None]*s/3).sum(0)
                covariance = (np.einsum('n,nki,nkj->ij', w, skin, skin)+np.einsum('n,ni,nj->ij', w, s, s))/12-np.outer(c, c)
                covariances.append(covariance); skin_centers.append(c)
                skin_normals.append(np.linalg.eigh(covariance)[1][:, 0])
            # Distinct, parallel, thin skins only; splitting a curved surface
            # or unrelated faces into two clusters must not flatten them.
            if abs(skin_normals[0]@skin_normals[1]) > np.cos(np.deg2rad(5)) and abs(means[1]-means[0]) < span*.01:
                values, axes = np.linalg.eigh(sum(m*c for m, c in zip(masses, covariances)))
                center = np.mean(skin_centers, axis=0)
    if values[1] <= 1e-15 or values[1]/max(values[2], 1e-15) < (.002 if narrow else .01) or max(values[0], 0)/values[1] > (.08 if narrow else .02):
        raise PlatformError('measurement_no_stable_plane', 422)
    normal = axes[:, 0]; points = faces.reshape(-1, 3)
    local = (points-center)@axes[:, 1:]
    lo, hi = local.min(0), local.max(0)
    outline = [center+axes[:, 1]*x+axes[:, 2]*y for x, y in [(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1]),(lo[0],lo[1])]]
    return dict(center=center, normal=normal, outline=np.asarray(outline), areaFraction=fraction,
                residual=float(np.sqrt(max(values[0], 0))), span=span, points=points)


def fitted_bend(triangles):
    """Two supported sheet faces of ONE mesh, with an interior angle at their hinge."""
    t = np.asarray(triangles, float)
    try:
        a = fitted_plane(t, narrow=True)
        cross = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
        area = np.linalg.norm(cross, axis=1)/2
        normals = cross/np.maximum(2*area[:, None], 1e-30)
        # ponytail: this bounded two-plane fit covers a single visible fold with
        # at least 20 degrees of normal separation. Multiple/rounded folds need
        # explicit surface selection, not a forced two-plane answer.
        remaining = (area > 1e-15) & (np.abs(normals@a['normal']) < np.cos(np.deg2rad(20)))
        if not remaining.any(): raise ValueError()
        b = fitted_plane(t[remaining], narrow=True)
        b['areaFraction'] *= float(area[remaining].sum()/area.sum())
        if min(a['areaFraction'], b['areaFraction']) < .1 or a['areaFraction']+b['areaFraction'] < .6:
            raise ValueError()
        axis = np.cross(a['normal'], b['normal']); length = np.linalg.norm(axis)
        if length < np.sin(np.deg2rad(10)): raise ValueError()
        axis /= length
        center = (a['center']+b['center'])/2
        matrix = np.stack([a['normal'], b['normal'], axis])
        origin = center+np.linalg.solve(matrix, [(a['center']-center)@a['normal'], (b['center']-center)@b['normal'], 0])
        intervals=[]; directions=[]; widths=[]
        for surface in (a,b):
            points=surface['points']; along=(points-origin)@axis
            intervals.append((float(along.min()),float(along.max())))
            ray=surface['center']-origin; ray-=axis*(ray@axis)
            width=np.linalg.norm(ray)
            if width < max(surface['residual']*2.5, a['span']*.01): raise ValueError()
            ray/=width; across=(points-origin)@ray
            # The intersection must meet the supported edge of BOTH patches,
            # not merely intersect their infinite planes somewhere off-model.
            tolerance=max(a['span']*.03, surface['residual']*3)
            if across.min()>tolerance or across.min() < -tolerance: raise ValueError()
            directions.append(ray); widths.append(float(across.max()))
        lo=max(i[0] for i in intervals); hi=min(i[1] for i in intervals)
        if hi-lo < .25*min(i[1]-i[0] for i in intervals): raise ValueError()
        hinge=np.array([origin+axis*lo,origin+axis*hi])
        # Draw the cross-section near a shared edge so the bend is legible.
        vertex=hinge[0]+(hinge[1]-hinge[0])*.05
        cosine=float(np.clip(directions[0]@directions[1],-1,1))
        angle=float(np.arccos(cosine)); radius=min(widths)*.65
        tangent=directions[1]-cosine*directions[0]; tangent/=np.linalg.norm(tangent)
        arc=np.array([vertex+radius*(directions[0]*np.cos(v)+tangent*np.sin(v)) for v in np.linspace(0,angle,33)])
        return dict(value=float(np.degrees(angle)),surfaces=[a,b],hinge=hinge,
                    rays=np.array([vertex+directions[0]*radius,vertex,vertex+directions[1]*radius]),
                    arc=arc,labelPoint=arc[len(arc)//2])
    except (ValueError, np.linalg.LinAlgError, PlatformError):
        raise PlatformError('measurement_no_stable_bend',422) from None


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


def posed_model(doc, id, load_asset, *, triangle_limit=500_000):
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
    if triangle_limit is not None and len(triangles)>triangle_limit: raise PlatformError('measurement_complexity_limit',422)
    return (triangles,pose['coordinateFrameId']), {'entityId':id,'representationId':r['id'],'assetId':r.get('assetId'),'assetSha256':(asset or {}).get('sha256'),'placementState':r.get('placementState'),'qualityStatus':(r.get('qualityEvidence') or {}).get('status')}


def measure_scene(revision, kind, entity_a, entity_b, region, load_asset, *, triangle_limit=500_000):
    if kind not in ('angle','inclination','bend','distance','occupancy'):
        raise PlatformError('measurement_kind_invalid',422)
    doc=revision['document']; models=[]; refs=[]
    ids=[entity_a] if kind in ('occupancy','inclination','bend') else [entity_a,entity_b]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise PlatformError('measurement_choose_objects',422)
    for id in ids:
        model,ref=posed_model(doc,id,load_asset,triangle_limit=triangle_limit);models.append(model);refs.append(ref)
    frame_id=models[0][1]
    frame=next((f for f in doc['coordinateFrames'] if f['id']==frame_id),None)
    if frame is None: raise PlatformError('measurement_frame_mismatch',422)
    if any(frame!=frame_id for _,frame in models): raise PlatformError('measurement_frame_mismatch',422)
    result={'revisionId':revision['id'],'kind':kind,'coordinateFrameId':frame_id,'source':'model_inference','references':refs,'lines':[],'quality':{},'unit':'native','value':None}
    if kind=='bend':
        bend=fitted_bend(models[0][0])
        result.update(value=bend['value'],unit='deg',method=BEND_ALGORITHM,labelPoint=bend['labelPoint'].tolist())
        result['quality']={'surfaceFits':[{'areaFraction':p['areaFraction'],'rmsResidualNative':p['residual']} for p in bend['surfaces']]}
        result['lines']=[{'points':p['outline'].tolist(),'color':color} for p,color in zip(bend['surfaces'],['#e36b23','#168bba'])]
        result['lines'] += [{'points':bend[key].tolist(),'color':color} for key,color in [('hinge','#b56ce2'),('rays','#edbe38'),('arc','#86e342')]]
    elif kind in ('angle','inclination'):
        a=fitted_plane(models[0][0])
        if kind=='inclination':
            normal=np.asarray((frame.get('ground') or {}).get('normal',[]),float)
            if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-8:
                raise PlatformError('measurement_ground_missing',422)
            normal=normal/np.linalg.norm(normal)
            x=np.cross(normal,[1,0,0] if abs(normal[0])<.8 else [0,1,0]); x/=np.linalg.norm(x)
            y=np.cross(normal,x); half=a['span']*.3
            # This datum is parallel to ground through the panel center, not a
            # claim that the physical floor lies at the panel's elevation.
            outline=[a['center']+half*(u*x+v*y) for u,v in [(-1,-1),(1,-1),(1,1),(-1,1),(-1,-1)]]
            b={'normal':normal,'center':a['center'],'span':a['span'],'outline':np.asarray(outline)}
        else:
            b=fitted_plane(models[1][0])
        cosine=float(np.clip(abs(a['normal']@b['normal']),0,1)); angle=float(np.degrees(np.arccos(cosine)))
        result.update(value=angle,unit='deg',method='dominant-surface-plane-acute-angle-v1')
        result['quality']={'surfaceFits':[{'areaFraction':p['areaFraction'],'rmsResidualNative':p['residual']} for p in ((a,) if kind=='inclination' else (a,b))]}
        result['lines']=[{'points':p['outline'].tolist(),'color':color} for p,color in [(a,'#e36b23'),(b,'#168bba')]]
        normal=b['normal'] if a['normal']@b['normal']>=0 else -b['normal']
        origin=(a['center']+b['center'])/2; size=min(a['span'],b['span'])*.25
        # The arc represents plane-normal separation (acute angle), not an
        # inferred hinge interior angle; UI names that convention explicitly.
        tangent=normal-a['normal']*cosine; length=np.linalg.norm(tangent)
        if length>1e-9:
            arc=[origin+size*(a['normal']*np.cos(t)+tangent/length*np.sin(t)) for t in np.linspace(0,np.deg2rad(angle),25)]
            result['lines'].append({'points':[origin.tolist(),arc[0].tolist(),*[p.tolist() for p in arc],origin.tolist()],'color':'#edbe38'})
        if kind=='inclination':
            result['method']='dominant-surface-ground-inclination-v1'
            result['quality'].update(deviationFromVerticalDeg=90-angle, groundReference=frame['ground'])
            panel_normal=a['normal'] if a['normal']@b['normal']>=0 else -a['normal']
            horizontal=cosine*b['normal']-panel_normal
            horizontal=horizontal/np.linalg.norm(horizontal) if np.linalg.norm(horizontal)>1e-9 else x
            arc=[origin+size*(horizontal*np.cos(t)+b['normal']*np.sin(t)) for t in np.linspace(0,np.deg2rad(angle),25)]
            result['lines']=result['lines'][:2]+[{'points':[origin.tolist(),*[p.tolist() for p in arc],origin.tolist()],'color':'#edbe38'}]
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


BEND_ALGORITHM = "same-mesh-two-surface-interior-bend-v2"
BEND_ANALYSIS_ROUTE = "bend-analysis-v1"


def bend_input(document, entity):
    rep = next((r for r in entity.get('representations') or []
                if r['id'] == entity.get('activeModelRepresentationId')), None)
    asset = next((a for a in document['assets'] if rep and a['id'] == rep.get('assetId')), None)
    frame = next((f for f in document['coordinateFrames'] if rep and f['id'] == rep.get('coordinateFrameId')), None)
    return digest({'algorithm': BEND_ALGORITHM, 'entityId': entity['id'], 'representation': rep,
                   'asset': asset, 'pose': entity.get('currentModelTransform'), 'frame': frame})


def saved_bends(revision):
    """Read saved exact-input outcomes only; never load assets on report open."""
    items = []
    for entity in revision['document']['entities']:
        if entity.get('sourceContext') or entity.get('visible') is False:
            continue
        fingerprint = bend_input(revision['document'], entity)
        saved = entity.get('bendAnalysis') or {}
        row = dict(saved) if saved.get('inputSha256') == fingerprint else {'status': 'not_processed'}
        row.update(entityId=entity['id'], inputSha256=fingerprint)
        if row.get('result'):
            row['result'] = {**row['result'], 'revisionId': revision['id']}
        items.append(row)
    return {'revisionId': revision['id'], 'algorithm': BEND_ALGORITHM, 'items': items}


def analyze_bends(revision, load_asset, *, persist=False, cache=None):
    """Processing/publication stage. Cache binds geometry, pose and algorithm."""
    analysis = saved_bends(revision)
    entities = {e['id']: e for e in revision['document']['entities']}
    cache = {} if cache is None else cache
    for row in analysis['items']:
        entity = entities[row['entityId']]
        if row['status'] in ('not_processed', 'failed'):
            cached = cache.get(row['inputSha256'])
            if cached and cached['status'] != 'failed':
                row.update(cached)
            else:
                try:
                    result = measure_scene(revision, 'bend', entity['id'], None, None, load_asset, triangle_limit=None)
                    result.pop('revisionId', None)
                    row.pop('reason', None)
                    row.update(status='measured', result=result)
                except PlatformError as error:
                    status = ('unsupported' if error.code in ('measurement_no_stable_bend', 'measurement_no_surface')
                              else 'skipped' if error.code == 'measurement_model_missing' else 'failed')
                    row.update(status=status, reason=error.code)
                    row.pop('result', None)
                except (ValueError, TypeError, KeyError, OSError):
                    row.update(status='failed', reason='measurement_invalid_geometry')
                    row.pop('result', None)
        stored = {**row}
        if row.get('result'):
            stored['result'] = {k:v for k,v in row['result'].items() if k != 'revisionId'}
            row['result'] = {**stored['result'], 'revisionId': revision['id']}
        cache[row['inputSha256']] = stored
        if persist:
            entity['bendAnalysis'] = stored
    return analysis


def register_measurement_routes(app, get_revision, load_asset, get_analysis=None, get_inclinations=None):
    from fastapi import Query
    @app.get('/api/revisions/{revision_id}/' + BEND_ANALYSIS_ROUTE, response_model=DocumentDTO, response_model_exclude_unset=True)
    def bend_analysis(revision_id: str):
        saved = get_analysis(revision_id) if get_analysis else None
        if saved is not None: return saved
        revision = get_revision(revision_id)
        if revision is None: raise PlatformError('revision_not_found', 404)
        return saved_bends(revision)

    @app.get('/api/revisions/{revision_id}/inclination-analysis-v1', response_model=DocumentDTO, response_model_exclude_unset=True)
    def inclination_analysis(revision_id: str):
        saved=get_inclinations(revision_id) if get_inclinations else None
        if saved is not None:return saved
        revision=get_revision(revision_id)
        if revision is None:raise PlatformError('revision_not_found',404)
        return saved_inclinations(revision)

    @app.get('/api/revisions/{revision_id}/measurements', response_model=DocumentDTO, response_model_exclude_unset=True)
    def measurement(revision_id: str, kind: str, entityA: str, entityB: str | None = None,
                    region: str | None = Query(default=None, max_length=12000)):
        revision=get_revision(revision_id)
        if revision is None: raise PlatformError('revision_not_found',404)
        return measure_scene(revision,kind,entityA,entityB,region,load_asset)


INCLINATION_ANALYSIS_ROUTE = 'inclination-analysis-v1'


def saved_inclinations(revision, config=None):
    from argus.platform.planar_surfaces import VERSION, DEFAULT_CONFIG
    config = {**DEFAULT_CONFIG, **(config or {})}
    rows=[]
    for entity in revision['document']['entities']:
        if entity.get('sourceContext') or entity.get('visible') is False:continue
        fingerprint=digest({'algorithm':VERSION,'geometry':bend_input(revision['document'],entity),'config':config})
        saved=entity.get('inclinationAnalysis') or {}
        row=dict(saved) if saved.get('inputSha256')==fingerprint else {'status':'not_processed','surfaces':[]}
        row.update(entityId=entity['id'],inputSha256=fingerprint)
        row['surfaces']=[{**s,'result':{**s['result'],'revisionId':revision['id']}} for s in row.get('surfaces',[])]
        rows.append(row)
    return {'revisionId':revision['id'],'algorithm':VERSION,'configuration':config,'items':rows}


def analyze_inclinations(revision, load_asset, *, persist=False, cache=None, config=None):
    from argus.platform.planar_surfaces import extract_planar_surfaces, surface_inclinations, VERSION
    analysis=saved_inclinations(revision,config);doc=revision['document']
    entities={e['id']:e for e in doc['entities']};cache={} if cache is None else cache
    for row in analysis['items']:
        if row['status'] in ('not_processed','failed'):
            cached=cache.get(row['inputSha256'])
            if cached and cached['status']!='failed':row.update(cached)
            else:
                row.pop('reason',None)
                try:
                    (triangles,frame_id),ref=posed_model(doc,row['entityId'],load_asset,triangle_limit=None)
                    frame=next((f for f in doc['coordinateFrames'] if f['id']==frame_id),None)
                    if frame is None:raise PlatformError('measurement_frame_mismatch',422)
                    ground=frame.get('ground') or {}
                    surface_inclinations([],ground)  # Validate datum before detecting surfaces.
                    diagnostics={}
                    patches=extract_planar_surfaces(triangles,analysis['configuration'],diagnostics=diagnostics)
                    surfaces=surface_inclinations(patches,ground); saved=[]
                    for s in surfaces:
                        n=np.asarray(s['normal']);g=np.asarray(ground['normal'],float);g/=np.linalg.norm(g)
                        c=np.asarray(s['center']);cosine=float(np.clip(abs(n@g),0,1));g=g if n@g>=0 else -g
                        # Cross-section rays along the measured and horizontal planes.
                        axis=np.cross(n,g);length=np.linalg.norm(axis)
                        radius=np.sqrt(s['areaNative2'])*.25
                        lines=[{'points':b,'color':'#e36b23'} for b in s['boundary']]
                        if length>1e-8:
                            axis/=length;u=np.cross(axis,n);v=np.cross(axis,g)
                            tangent=v-u*cosine;tangent/=max(np.linalg.norm(tangent),1e-30)
                            arc=np.array([c+radius*(u*np.cos(a)+tangent*np.sin(a)) for a in np.linspace(0,np.deg2rad(s['inclinationDeg']),25)])
                            lines += [{'points':[c.tolist(),arc[0].tolist()],'color':'#e36b23'}, {'points':[c.tolist(),arc[-1].tolist()],'color':'#168bba'}, {'points':arc.tolist(),'color':'#409639'}]
                            label=arc[len(arc)//2].tolist()
                        else:label=c.tolist()
                        result={'revisionId':revision['id'],'kind':'inclination','coordinateFrameId':frame_id,'source':'model_inference','unit':'deg','method':VERSION,
                            'value':s['inclinationDeg'],'references':[ref],'lines':lines,'labelPoint':label,
                            'quality':{'deviationFromVerticalDeg':s['deviationFromVerticalDeg'],'groundReference':ground,'areaNative2':s['areaNative2'],
                                       'angularErrorDeg':s['angularErrorDeg'],'angularSpreadDeg':s['angularSpreadDeg'],'rmsResidualNative':s['rmsResidualNative'],
                                       'classification':s['classification'],'surfaceId':s['surfaceId']}}
                        # Full support indices are used during fitting; report needs boundaries/summary only.
                        saved.append({k:v for k,v in s.items() if k not in ('triangleIndices','boundary','normal','center')}|{'result':result})
                    row.update(status='partial' if diagnostics.get('limitReached') else 'measured' if saved else 'unsupported',surfaces=saved,diagnostics=diagnostics)
                    if diagnostics.get('limitReached'):row['reason']='measurement_complexity_limit'
                    elif not saved:row['reason']='measurement_no_stable_local_plane'
                except PlatformError as exc:
                    row.update(status='skipped' if exc.code=='measurement_model_missing' else 'failed',reason=exc.code,surfaces=[])
                except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
                    reason='measurement_ground_missing' if str(exc)=='measurement_ground_missing' else 'measurement_invalid_geometry'
                    row.update(status='skipped' if reason=='measurement_ground_missing' else 'failed',reason=reason,surfaces=[])
        stored={**row,'surfaces':[{**s,'result':{k:v for k,v in s['result'].items() if k!='revisionId'}} for s in row['surfaces']]}
        cache[row['inputSha256']]=stored
        row['surfaces']=[{**s,'result':{**s['result'],'revisionId':revision['id']}} for s in row['surfaces']]
        if persist:entities[row['entityId']]['inclinationAnalysis']=stored
    return analysis
