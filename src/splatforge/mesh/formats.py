"""Precision-preserving triangle PLY/OFF I/O for geometry processing."""
from pathlib import Path
import numpy as np
from .processing import TriangleMesh, _mesh


def read_mesh(path):
    path = Path(path)
    if path.suffix.lower() == '.off':
        def lines(handle):
            for line in handle:
                text = line.split('#', 1)[0].strip()
                if text: yield text
        with path.open(encoding='ascii') as handle:
            data = lines(handle)
            if next(data) != 'OFF': raise ValueError('expected OFF header')
            nv, nf, _ = map(int, next(data).split())
            vertices = [list(map(float, next(data).split())) for _ in range(nv)]
            faces = []
            for _ in range(nf):
                row = list(map(int, next(data).split()))
                if len(row) != 4 or row[0] != 3: raise ValueError('only triangle OFF supported')
                faces.append(row[1:])
        return _mesh(np.asarray(vertices).reshape(-1, 3), faces)
    if path.suffix.lower() != '.ply': raise ValueError('geometry I/O supports .ply and .off')
    scalar = {'char':'i1','int8':'i1','uchar':'u1','uint8':'u1','short':'i2','int16':'i2','ushort':'u2','uint16':'u2','int':'i4','int32':'i4','uint':'u4','uint32':'u4','float':'f4','float32':'f4','double':'f8','float64':'f8'}
    with path.open('rb') as handle:
        if handle.readline().strip() != b'ply': raise ValueError('expected PLY header')
        elements=[]; properties={}; encoding=None
        for _ in range(10000):
            line=handle.readline()
            if not line: raise ValueError('truncated PLY header')
            parts=line.decode('ascii').split()
            if not parts or parts[0] in ('comment','obj_info'): continue
            if parts[0]=='end_header': break
            if parts[0]=='format': encoding=parts[1]
            if parts[0]=='element':
                element=parts[1];elements.append((element,int(parts[2])));properties[element]=[]
            if parts[0]=='property': properties[element].append(parts[1:])
        else: raise ValueError('PLY header too long')
        if [e[0] for e in elements] != ['vertex','face']: raise ValueError('only vertex/face PLY layout supported')
        nv,nf=elements[0][1],elements[1][1]
        if nv<0 or nf<0: raise ValueError('negative PLY counts')
        vp=properties['vertex'];fp=properties['face']
        if any(len(p)!=2 or p[0] not in scalar for p in vp): raise ValueError('unsupported vertex property')
        names=[p[1] for p in vp]
        if not all(a in names for a in ('x','y','z')): raise ValueError('PLY needs xyz')
        if len(fp)!=1 or len(fp[0])!=4 or fp[0][0]!='list' or fp[0][3] not in ('vertex_indices','vertex_index'):
            raise ValueError('only a single triangle index list supported on faces')
        if encoding=='ascii':
            rows=[list(map(float,handle.readline().split())) for _ in range(nv)]
            if any(len(r)!=len(vp) for r in rows): raise ValueError('invalid PLY vertex record')
            v=np.asarray(rows).reshape(nv,len(vp))[:,[names.index(a) for a in ('x','y','z')]]
            f=[]
            for _ in range(nf):
                r=list(map(int,handle.readline().split()))
                if len(r)!=4 or r[0]!=3: raise ValueError('invalid triangle PLY record')
                f.append(r[1:])
        elif encoding in ('binary_little_endian','binary_big_endian'):
            endian='<' if encoding=='binary_little_endian' else '>'
            data=np.fromfile(handle,dtype=np.dtype([(p[1],endian+scalar[p[0]]) for p in vp]),count=nv)
            if len(data)!=nv: raise ValueError('truncated PLY vertices')
            v=np.column_stack([data[a] for a in ('x','y','z')])
            count_type,index_type=fp[0][1:3]
            if scalar.get(count_type) not in ('u1','i1') or scalar.get(index_type) not in ('i4','u4'):
                raise ValueError('binary face lists need uchar/char count and int32/uint32 index')
            records=np.fromfile(handle,dtype=[('count',scalar[count_type]),('indices',endian+scalar[index_type],(3,))],count=nf)
            if len(records)!=nf or np.any(records['count']!=3): raise ValueError('truncated or nontriangle PLY faces')
            f=records['indices']
        else: raise ValueError(f'unsupported PLY encoding {encoding}')
        return _mesh(v,f)


def write_mesh(path, mesh: TriangleMesh):
    """Write doubles, without the Studio viewer export's float32 conversion."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower()=='.off':
        with path.open('w',encoding='ascii',newline='\n') as handle:
            handle.write(f'OFF\n{len(mesh.vertices)} {len(mesh.faces)} 0\n')
            np.savetxt(handle,mesh.vertices,fmt='%.17g')
            np.savetxt(handle,np.c_[np.full(len(mesh.faces),3),mesh.faces],fmt='%d')
        return
    if path.suffix.lower()!='.ply': raise ValueError('geometry I/O supports .ply and .off')
    if mesh.faces.max(initial=0)>np.iinfo(np.int32).max: raise ValueError('PLY int32 face index overflow')
    with path.open('wb') as handle:
        header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(mesh.vertices)}\nproperty double x\nproperty double y\nproperty double z\nelement face {len(mesh.faces)}\nproperty list uchar int vertex_indices\nend_header\n'
        handle.write(header.encode('ascii'));np.asarray(mesh.vertices,dtype='<f8').tofile(handle)
        records=np.empty(len(mesh.faces),dtype=[('count','u1'),('indices','<i4',(3,))]);records['count']=3;records['indices']=mesh.faces
        records.tofile(handle)
