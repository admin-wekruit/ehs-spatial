"""Local planar patches from mesh support, with ground-relative model angles."""
from __future__ import annotations
import numpy as np
from shapely import polygons, union_all, get_parts
from shapely.geometry import Polygon

VERSION = 'local-planar-inclinations-v1'
# Native units, not metres. Persist these engineering settings with each result.
DEFAULT_CONFIG = {'minAreaNative2': .02, 'distanceToleranceNative': .01, 'normalToleranceDeg': 10.}


def extract_planar_surfaces(triangles, config, *, diagnostics=None):
    import open3d as o3d
    t=np.asarray(triangles,float)
    minimum=float(config['minAreaNative2']); tolerance=float(config['distanceToleranceNative']); degrees=float(config['normalToleranceDeg'])
    if t.ndim!=3 or t.shape[1:]!=(3,3) or not np.isfinite(t).all() or not all(np.isfinite([minimum,tolerance,degrees])) or minimum<=0 or tolerance<=0 or not 0<degrees<45:
        raise ValueError('invalid_planar_input')
    crosses=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); areas=np.linalg.norm(crosses,axis=1)/2
    valid=areas>1e-15; indices=np.flatnonzero(valid); t=t[valid];areas=areas[valid];crosses=crosses[valid]
    if areas.sum()<minimum:return []
    normals=crosses/(2*areas[:,None]); centers=t.mean(1)
    # ponytail: bounded area sampling limits detection cost; supports below this
    # resolution remain unmeasured, never a claim that no surface exists.
    # Increase sampling relative to minimum support area if finer coverage is needed.
    rng=np.random.default_rng(0); count=16000
    sampled=np.searchsorted(np.cumsum(areas),rng.random(count)*areas.sum())
    uv=rng.random((count,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    pts=t[sampled,0]+uv[:,0,None]*(t[sampled,1]-t[sampled,0])+uv[:,1,None]*(t[sampled,2]-t[sampled,0])
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts));cloud.normals=o3d.utility.Vector3dVector(normals[sampled])
    o3d.utility.random.seed(0)
    used=np.zeros(len(t),bool); results=[]
    limit_reached=True
    for _ in range(64):
        if len(cloud.points)<30:
            limit_reached=False;break
        plane,inliers=cloud.segment_plane(distance_threshold=tolerance/2,ransac_n=3,num_iterations=128,probability=.999)
        if len(inliers)<30:
            limit_reached=False;break
        cloud=cloud.select_by_index(inliers,invert=True)
        n=np.asarray(plane[:3]);offset=float(plane[3])
        support=(~used)&(abs(normals@n)>=np.cos(np.deg2rad(degrees)))&(np.max(abs(t@n+offset),axis=1)<=tolerance)
        if areas[support].sum()<minimum:continue
        f=t[support];w=areas[support];ids=np.flatnonzero(support)
        # Area-weighted normals ignore material thickness and triangle density.
        aligned=normals[support]*np.where(normals[support]@n<0,-1,1)[:,None]
        n=np.sum(w[:,None]*aligned,axis=0);n/=np.linalg.norm(n)
        if n[np.argmax(abs(n))]<0:n=-n
        origin=np.average(centers[support],axis=0,weights=w)
        u=np.cross(n,[1,0,0] if abs(n[0])<.8 else [0,1,0]);u/=np.linalg.norm(u);v=np.cross(n,u)
        coords=np.stack([(f-origin)@u,(f-origin)@v],axis=-1)
        # Exact supported projection: no convex hull/box area and no gap filling.
        footprints=polygons(coords); merged=union_all(footprints)
        for part in get_parts(merged):
            if part.geom_type!='Polygon' or part.area<minimum:continue
            members=np.array([part.intersects(p.representative_point()) for p in footprints])
            if not members.any():continue
            chosen=ids[members];face=t[chosen];weight=areas[chosen]
            center=np.average(centers[chosen],axis=0,weights=weight)
            # Parallel front/back skins share projected support and count once.
            deviation=np.degrees(np.arccos(np.clip(abs(normals[chosen]@n),0,1)))
            angular=float(np.sqrt(np.average(deviation**2,weights=weight)))
            offsets=(centers[chosen]-center)@n
            rings=[part.exterior,*part.interiors]
            boundary=[(origin+np.asarray(r.coords)[:,0,None]*u+np.asarray(r.coords)[:,1,None]*v).tolist() for r in rings]
            results.append({'center':center.tolist(),'normal':n.tolist(),'boundary':boundary,'areaNative2':float(part.area),
                'rmsResidualNative':float(np.sqrt(np.average(offsets**2,weights=weight))),
                'angularSpreadDeg':angular,'triangleIndices':indices[chosen].tolist()})
            used[chosen]=True
    # Merge parallel front/back patches only when their observed footprints
    # overlap. Proximity alone must never join separate coplanar pieces.
    merged_results=[]
    for row in sorted(results,key=lambda r:-r['areaNative2']):
        for other in merged_results:
            n=np.asarray(other['normal']); center=np.asarray(other['center'])
            if abs(n@row['normal'])<np.cos(np.deg2rad(5)) or abs((np.asarray(row['center'])-center)@n)>3*tolerance:continue
            u=np.cross(n,[1,0,0] if abs(n[0])<.8 else [0,1,0]);u/=np.linalg.norm(u);v=np.cross(n,u)
            def footprint(r):
                rings=[np.stack([(np.asarray(b)-center)@u,(np.asarray(b)-center)@v],axis=-1) for b in r['boundary']]
                return Polygon(rings[0],rings[1:])
            a,b=footprint(other),footprint(row)
            if a.intersection(b).area < .8*min(a.area,b.area):continue
            shape=a.union(b)
            if shape.geom_type!='Polygon':continue
            other['boundary']=[(center+np.asarray(r.coords)[:,0,None]*u+np.asarray(r.coords)[:,1,None]*v).tolist() for r in [shape.exterior,*shape.interiors]]
            other['areaNative2']=float(shape.area)
            other['triangleIndices']=sorted(set(other['triangleIndices'])|set(row['triangleIndices']))
            other['angularSpreadDeg']=max(other['angularSpreadDeg'],row['angularSpreadDeg'])
            other['rmsResidualNative']=max(other['rmsResidualNative'],row['rmsResidualNative'])
            break
        else:merged_results.append(row)
    results=merged_results
    results.sort(key=lambda row:(-round(row['areaNative2'],8),*np.round(row['center'],8)))
    if diagnostics is not None:
        diagnostics.update(limitReached=limit_reached, sampleCount=count, maxPlaneCandidates=64, supportedTriangleAreaFraction=float(areas[used].sum()/areas.sum()))
    return [{'surfaceId':str(i+1),**row} for i,row in enumerate(results)]


def surface_inclinations(surfaces, ground):
    n=np.asarray(ground.get('normal',[]),float)
    if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)<1e-12:raise ValueError('measurement_ground_missing')
    n/=np.linalg.norm(n)
    ground_error=ground.get('angularErrorDeg')
    if ground_error is not None and (not np.isfinite(ground_error) or ground_error<0):raise ValueError('invalid_ground_error')
    rows=[]
    for surface in surfaces:
        theta=float(np.degrees(np.arccos(np.clip(abs(np.dot(surface['normal'],n)),0,1))))
        delta=90-theta; spread=surface['angularSpreadDeg']
        error=None if ground_error is None else float(ground_error+spread)
        classification='direction_unverified' if error is None else 'non_vertical' if delta>max(error,1e-6) else 'vertical_within_error'
        rows.append({**surface,'inclinationDeg':theta,'deviationFromVerticalDeg':delta,'angularErrorDeg':error,'classification':classification})
    return rows
