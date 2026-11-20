"""Minimal binary COLMAP and user-data reader. SPDX-License-Identifier: Apache-2.0"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path, PurePosixPath
import struct
import json
import numpy as np
from PIL import Image


def rotation(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def read_exact(stream, fmt):
    size=struct.calcsize('<'+fmt);data=stream.read(size)
    if len(data)!=size:raise ValueError('Truncated COLMAP binary')
    return struct.unpack('<'+fmt,data)


def validate_normalization_resume(saved_center, saved_radius, center, radius):
    """Reject a changed scene while allowing cross-CPU double rounding."""
    saved_center = np.asarray(saved_center, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    saved_radius, radius = float(saved_radius), float(radius)
    if (saved_center.shape != (3,) or center.shape != (3,)
            or not np.isfinite(saved_center).all() or not np.isfinite(center).all()
            or not np.isfinite([saved_radius, radius]).all()
            or min(saved_radius, radius) <= 0):
        raise ValueError('Invalid checkpoint or input normalization')
    tolerance = np.finfo(np.float32).eps
    if (np.linalg.norm(saved_center-center)/saved_radius > tolerance
            or abs(radius/saved_radius-1) > tolerance):
        raise ValueError('Resume scene normalization mismatch')


def validate_resume_dataset_paths(previous, data_dir, labels, output):
    """Permit only the fixed data layout of a relocated, same-named job."""
    current = {'data_dir': data_dir, 'labels': labels}
    if all(previous.get(name) == value for name, value in current.items()):
        return
    old_output = PurePosixPath(str(previous.get('output', '')))
    new_output = PurePosixPath(str(output))
    old_root, new_root = old_output.parent.parent, new_output.parent.parent
    if (not old_output.is_absolute() or not new_output.is_absolute()
            or old_output.parts[-2:] != ('train', 'joint')
            or new_output.parts[-2:] != ('train', 'joint')
            or old_root.name != new_root.name):
        raise ValueError('Resume dataset mismatch outside the same production job')
    for name, relative in [('data_dir', 'data'), ('labels', 'data/regional_labels')]:
        if (previous.get(name) != str(old_root/relative)
                or current[name] != str(new_root/relative)):
            raise ValueError('Resume dataset layout mismatch: '+name)


class ColmapInputs:
    def __init__(self, root, *, labels=None, label_subject='subject3', factor=1, cache_views=12):
        if cache_views<1:raise ValueError('cache_views must be positive')
        if not isinstance(label_subject,str) or not label_subject or label_subject in ['.','..'] or any(c in label_subject for c in ['/','\\']):
            raise ValueError('label_subject must be one directory name')
        self.root=Path(root);self.factor=factor;self.cache=OrderedDict();self.cache_views=cache_views;self.label_root=Path(labels) if labels else None
        self.label_subject=label_subject;self.label_directory=None
        if self.label_root is not None:
            self.label_directory=self.label_root/'labels'/label_subject if (self.label_root/'labels').is_dir() else self.label_root
            if not self.label_directory.is_dir():raise ValueError('Selected label subject directory does not exist: '+str(self.label_directory))
        sparse=self.root/'sparse/0'
        if not sparse.exists():sparse=self.root/'sparse'
        cameras={}
        with (sparse/'cameras.bin').open('rb') as f:
            for _ in range(read_exact(f,'Q')[0]):
                cid,model,w,h=read_exact(f,'iiQQ')
                counts={0:3,1:4,2:4,3:5,4:8,5:8,6:12,7:5,8:4,9:5,10:12}
                pars=read_exact(f,'d'*counts[model])
                if model==0:fx,fy,cx,cy=pars[0],pars[0],pars[1],pars[2]
                elif model==1:fx,fy,cx,cy=pars
                else:raise ValueError('Use the existing undistorted PINHOLE inputs; distortion is not silently ignored')
                cameras[cid]=(int(w),int(h),np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32))
        rows=[]
        with (sparse/'images.bin').open('rb') as f:
            for _ in range(read_exact(f,'Q')[0]):
                record=read_exact(f,'i7di');iid=record[0];q=record[1:5];t=record[5:8];cid=record[8]
                name=bytearray()
                while True:
                    b=f.read(1)
                    if not b:raise ValueError('Truncated image name')
                    if b==b'\x00':break
                    name+=b
                count=read_exact(f,'Q')[0]
                observations=np.frombuffer(f.read(count*24),dtype=np.dtype([('x','<f8'),('y','<f8'),('point','<i8')])).copy()
                view=np.eye(4);view[:3,:3]=rotation(q);view[:3,3]=t
                w,h,k=cameras[cid]
                rows.append(dict(id=iid,name=name.decode(),width=w,height=h,K=k,c2w=np.linalg.inv(view),observations=observations))
        self.rows=sorted(rows,key=lambda r:r['name']);self.by_id={r['id']:i for i,r in enumerate(self.rows)}
        points=[];colors=[];point_ids=[];tracks=[]
        with (sparse/'points3D.bin').open('rb') as f:
            for _ in range(read_exact(f,'Q')[0]):
                row=read_exact(f,'Q3d3Bd');count=read_exact(f,'Q')[0]
                track=np.frombuffer(f.read(count*8),dtype=np.int32).reshape(-1,2).copy()
                point_ids.append(row[0]);points.append(row[1:4]);colors.append(row[4:7]);tracks.append(track)
        self.points=np.asarray(points,dtype=np.float32);self.colors=np.asarray(colors,dtype=np.float32)/255
        self.point_ids=np.asarray(point_ids,dtype=np.int64);self.tracks=tracks
        centers=np.stack([r['c2w'][:3,3] for r in self.rows]);self.center=centers.mean(0)
        self.radius=float(np.linalg.norm(centers-self.center,axis=1).max())
        self.points=(self.points-self.center)/self.radius
        for r in self.rows:r['c2w'][:3,3]=(r['c2w'][:3,3]-self.center)/self.radius
        self.train_indices=[i for i in range(len(self.rows)) if i%8!=0]
        self.test_indices=[i for i in range(len(self.rows)) if i%8==0]

    def _label_file(self, stem):
        if self.label_root is None:return None
        # Only data files are considered, never code, source traces or checkpoints.
        directory=self.label_directory/stem
        if (directory/'weight_mass.npy').is_file():return None
        root=self.label_directory
        candidates=[root/(stem+suffix) for suffix in ['.npz','.png']]
        candidates=[p for p in candidates if p.is_file()]
        return candidates[0] if candidates else None

    def load(self,index):
        if index in self.cache:
            self.cache.move_to_end(index);return self.cache[index]
        row=self.rows[index];name=row['name'];stem=Path(name).stem
        image_path=self.root/'images_1'/name
        if not image_path.exists():image_path=self.root/'images'/name
        with Image.open(image_path) as im:
            size=(row['width']//self.factor,row['height']//self.factor)
            rgb=np.array(im.convert('RGB').resize(size,Image.Resampling.LANCZOS),copy=True)
        mask_path=self.root/'masks_1'/(stem+'.png')
        if not mask_path.exists():mask_path=self.root/'masks'/(stem+'.png')
        with Image.open(mask_path) as im:foreground=np.asarray(im.convert('L').resize(size,Image.Resampling.NEAREST))>0
        hair=np.zeros(foreground.shape,dtype=np.float32);body=np.zeros_like(hair);known=np.zeros_like(foreground)
        detail=np.zeros_like(hair);unknown=np.zeros_like(foreground)
        directory=self.label_directory/stem if self.label_directory else None
        if directory is not None and (directory/'weight_mass.npy').is_file():
            # These are existing user segmentation data and review decisions.
            # Context appearance weights (.1) never become negative labels.
            mass=np.load(directory/'weight_mass.npy',allow_pickle=False).astype(np.float32)
            detail=np.load(directory/'weight_detail.npy',allow_pickle=False).astype(np.float32)
            unknown=np.load(directory/'ignore_effective.npy',allow_pickle=False).astype(bool)
            review=json.loads((directory/'counts.json').read_text())
            def resize_existing(a):
                if a.shape==(size[1],size[0]):return a.copy()
                if a.shape!=(row['height'],row['width']):raise ValueError('Segmentation/image coordinate mismatch')
                return np.asarray(Image.fromarray(a.astype(np.float32),mode='F').resize(size,Image.Resampling.NEAREST)).copy()
            mass,detail=resize_existing(mass),resize_existing(detail);unknown=resize_existing(unknown)>.5
            hair=np.maximum(mass,detail)
            negative_confidence=1.0 if review.get('mass_status')=='ACCEPT' else .25
            body=negative_confidence*foreground*(hair==0)*(~unknown)
            known=foreground & ~unknown
        path=self._label_file(stem)
        if path is not None and not (directory is not None and (directory/'weight_mass.npy').is_file()):
            if path.suffix=='.png':
                with Image.open(path) as im:hair=np.asarray(im.convert('L').resize(size,Image.Resampling.NEAREST),dtype=np.float32)/255
                known=foreground.copy();body=foreground.astype(np.float32)*(1-hair)
            else:
                with np.load(path,allow_pickle=False) as archive:
                    keys=set(archive.files)
                    for key in ['hair_confidence','hair_weight','masscore','hair','mass','positive']:
                        if key in keys:
                            hair=np.asarray(archive[key],dtype=np.float32).squeeze();break
                    for key in ['body_confidence','body_weight','body','negative']:
                        if key in keys:body=np.asarray(archive[key],dtype=np.float32).squeeze();break
                    if 'unknown' in keys:known=~np.asarray(archive['unknown'],dtype=bool).squeeze()
                    else:known=(hair>0)|(body>0)
                def resize(a):return np.asarray(Image.fromarray(a.astype(np.float32),mode='F').resize(size,Image.Resampling.NEAREST)).copy()
                hair,body=resize(hair),resize(body);known=resize(known.astype(np.float32))>.5
        k=row['K'].copy();k[:2]/=self.factor
        value=dict(rgb=rgb,foreground=foreground,hair=hair,body=body,detail=detail,unknown=unknown,known=known,K=k,c2w=row['c2w'].astype(np.float32),name=name)
        self.cache[index]=value
        if len(self.cache)>self.cache_views:self.cache.popitem(last=False)
        return value

    def point_partition(self):
        scores=np.zeros(len(self.points));votes=np.zeros(len(self.points));foreground_votes=np.zeros(len(self.points))
        lookup={int(pid):i for i,pid in enumerate(self.point_ids)}
        for index,row in enumerate(self.rows):
            item=self.load(index);obs=row['observations'];x=(obs['x']/self.factor).astype(int);y=(obs['y']/self.factor).astype(int)
            valid=(x>=0)&(x<item['foreground'].shape[1])&(y>=0)&(y<item['foreground'].shape[0])&(obs['point']>=0)
            for oi in np.flatnonzero(valid):
                pi=lookup.get(int(obs['point'][oi]))
                if pi is None:continue
                foreground_votes[pi]+=bool(item['foreground'][y[oi],x[oi]])
                hp=float(item['hair'][y[oi],x[oi]]);bp=float(item['body'][y[oi],x[oi]])
                if item['known'][y[oi],x[oi]] and hp+bp>0:
                    scores[pi]+=hp;votes[pi]+=hp+bp
        ratio=np.divide(scores,votes,out=np.zeros_like(scores),where=votes>0)
        hair=(votes>0)&(ratio>.5);unresolved=votes==0
        # Ownership of unobserved points is explicit; geometric interpolation
        # propagates a family identity, never a negative 2D supervision label.
        confident=(votes>0)&(foreground_votes>0)
        if confident.any() and unresolved.any():
            from scipy.spatial import cKDTree
            _,neighbors=cKDTree(self.points[confident]).query(self.points[unresolved],k=1)
            hair[unresolved]=hair[confident][neighbors]
        keep=foreground_votes>0
        return keep,hair,dict(raw_points=len(keep),foreground_points=int(keep.sum()),hair_points=int((hair&keep).sum()),unresolved_points=int((unresolved&keep).sum()))
