import csv,hashlib,io,json,os
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import numpy as np
from ahrs.filters import Madgwick

ROOT=Path(os.environ.get('FARMER_BENCHMARK_ROOT',Path(__file__).resolve().parents[1]))
ARCHIVE=ROOT/'empirical_source_data/certh_original_activity.zip'
OUT=Path(os.environ.get('FARMER_BENCHMARK_RUNS',ROOT/'empirical_runs'))
PROTOCOL=json.loads((ROOT/'empirical_experiment_tools/protocol.json').read_text())

def prepare():
    OUT.mkdir(exist_ok=True)
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==PROTOCOL['source_zip_sha256']
    records=[];saved={};counts=Counter()
    with ZipFile(ARCHIVE) as z:
        files=sorted(i for i in z.namelist() if i.endswith('_1530.csv'))
        for index,name in enumerate(files):
            payload=z.read(name)
            rows=list(csv.DictReader(io.StringIO(payload.decode('utf-8'))))
            times=np.array([float(r['time_s']) for r in rows])
            acc=np.array([[float(r[k]) for k in ('ax_m/s/s','ay_m/s/s','az_m/s/s')] for r in rows])
            gyro=np.array([[float(r[k]) for k in ('gx_deg/s','gy_deg/s','gz_deg/s')] for r in rows])
            labels=np.array([int(float(r['condition'])) for r in rows],dtype=np.int64)
            assert np.isfinite(acc).all() and np.isfinite(gyro).all()
            assert np.all(np.diff(times)>0) and np.allclose(np.diff(times),0.02,atol=1e-7)
            assert set(labels)==set(range(6)),name
            # Follow the author's README units, not the conflicting column suffix.
            quat=Madgwick(gyr=gyro,acc=acc,frequency=50.0,gain=0.033).Q
            quat/=np.linalg.norm(quat,axis=1,keepdims=True)
            if quat[0,np.flatnonzero(np.abs(quat[0])>1e-12)[0]]<0:quat[0]*=-1
            for t in range(1,len(quat)):
                if np.dot(quat[t-1],quat[t])<0:quat[t]*=-1
            assert np.isfinite(quat).all() and np.allclose(np.linalg.norm(quat,axis=1),1)
            stem=Path(name).stem;participant=stem[0]
            split=next(k for k in ('train','validation','test') if participant in PROTOCOL[k+'_subjects'])
            endpoints=np.arange(0,len(labels),PROTOCOL['evaluation_stride'],dtype=np.int64)
            next_labels=labels[np.minimum(endpoints+1,len(labels)-1)].copy()
            next_labels[endpoints+1>=len(labels)]=-1
            rec={'index':index,'clip_id':stem,'subject':participant,'split':split,
                 'source_member':name,'source_sha256':hashlib.sha256(payload).hexdigest(),
                 'native_frames':len(labels),'evaluation_frames':len(endpoints),
                 'duration_seconds':float(times[-1]-times[0]),
                 'class_counts_native':[int(np.sum(labels==k)) for k in range(6)],
                 'class_counts_evaluation':[int(np.sum(labels[endpoints]==k)) for k in range(6)]}
            records.append(rec);counts.update(int(x) for x in labels)
            saved[f'q_{index}']=quat.astype(np.float32)
            saved[f'y_{index}']=labels
            saved[f't_{index}']=(times-times[0]).astype(np.float64)
            saved[f'e_{index}']=endpoints
            saved[f'n_{index}']=next_labels
    np.savez_compressed(OUT/'recordings.npz',**saved)
    inventory={'recordings':records,'native_frames':sum(r['native_frames'] for r in records),
        'evaluation_frames':sum(r['evaluation_frames'] for r in records),
        'total_duration_seconds':sum(r['duration_seconds'] for r in records),
        'class_counts_native':[int(counts[k]) for k in range(6)],'subjects':sorted({r['subject'] for r in records}),
        'splits':{k:{'trials':sum(r['split']==k for r in records),
              'native_frames':sum(r['native_frames'] for r in records if r['split']==k),
              'evaluation_frames':sum(r['evaluation_frames'] for r in records if r['split']==k),
              'class_counts_evaluation':[sum(r['class_counts_evaluation'][c] for r in records if r['split']==k) for c in range(6)]}
              for k in ('train','validation','test')}}
    (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2))
    (OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    print(json.dumps({k:v for k,v in inventory.items() if k!='recordings'},indent=2),flush=True)

if __name__=='__main__':prepare()
