import json,platform,importlib.metadata,hashlib,os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix

ROOT=Path(os.environ.get('FARMER_BENCHMARK_ROOT',Path(__file__).resolve().parents[1]))
RUNS=Path(os.environ.get('FARMER_BENCHMARK_RUNS',ROOT/'empirical_runs'))
PROTOCOL=json.loads((ROOT/'empirical_experiment_tools/protocol.json').read_text())

def load_records():
    inventory=json.loads((RUNS/'inventory.json').read_text())
    records=[]
    with np.load(RUNS/'recordings.npz') as arrays:
        for item in inventory['recordings']:
            i=item['index']
            records.append({**item,'q':arrays[f'q_{i}'].copy(),'labels':arrays[f'y_{i}'].copy(),
                'times':arrays[f't_{i}'].copy(),'endpoints':arrays[f'e_{i}'].copy(),
                'next_labels':arrays[f'n_{i}'].copy()})
    return records

def select(records,split):return [r for r in records if r['split']==split]

def evaluate(records,probabilities):
    labels=np.concatenate([r['labels'][r['endpoints']] for r in records])
    p=np.asarray(probabilities,dtype=np.float64)
    assert p.shape==(len(labels),6) and np.isfinite(p).all()
    assert np.allclose(p.sum(1),1,atol=1e-5) and p.min()>=-1e-8
    predictions=p.argmax(1)
    precision,recall,f1,support=precision_recall_fscore_support(labels,predictions,labels=np.arange(6),zero_division=0)
    errors=p-np.eye(6)[labels]
    mse=float(np.square(errors).mean())
    offset=0;near_errors=[];false_switches=0;duration=0
    per_subject={}
    for r in records:
        n=len(r['endpoints']);t=r['times'][r['endpoints']]
        pred=predictions[offset:offset+n];y=labels[offset:offset+n]
        bounds=r['times'][np.flatnonzero(np.diff(r['labels'])!=0)+1]
        near=np.min(np.abs(t[:,None]-bounds[None,:]),axis=1)<=PROTOCOL['transition_window_seconds']
        near_errors.extend((pred[near]!=y[near]).tolist())
        false_switches+=int(np.sum((np.diff(pred)!=0)&(~near[1:])&(~near[:-1])))
        duration+=r['duration_seconds']
        per_subject.setdefault(r['subject'],{'labels':[],'predictions':[]})
        per_subject[r['subject']]['labels'].extend(y.tolist())
        per_subject[r['subject']]['predictions'].extend(pred.tolist())
        offset+=n
    for v in per_subject.values():
        pp,rr,ff,_=precision_recall_fscore_support(v['labels'],v['predictions'],labels=np.arange(6),zero_division=0)
        v.clear();v.update({'macro_f1':float(ff.mean())})
    return {'accuracy':float(accuracy_score(labels,predictions)),'macro_precision':float(precision.mean()),
        'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),
        'probability_mae':float(np.abs(errors).mean()),'probability_mse':mse,'probability_rmse':float(np.sqrt(mse)),
        'transition_error':float(np.mean(near_errors)),
        'false_switches_per_minute':float(false_switches/(duration/60)),
        'transition_evaluation_frames':len(near_errors),'false_switches':false_switches,
        'per_class':{'precision':precision.tolist(),'recall':recall.tolist(),'f1':f1.tolist(),'support':support.tolist()},
        'confusion_matrix':confusion_matrix(labels,predictions,labels=np.arange(6)).tolist(),
        'evaluation_frames':len(labels),'per_test_subject':per_subject}

def save_predictions(path,records,probabilities):
    labels=np.concatenate([r['labels'][r['endpoints']] for r in records])
    np.savez_compressed(path,probabilities=np.asarray(probabilities,dtype=np.float32),labels=labels,
        native_frame=np.concatenate([r['endpoints'] for r in records]),
        elapsed_seconds=np.concatenate([r['times'][r['endpoints']] for r in records]),
        clip_id=np.concatenate([np.repeat(r['clip_id'],len(r['endpoints'])) for r in records]),
        subject=np.concatenate([np.repeat(r['subject'],len(r['endpoints'])) for r in records]))

def runtime():
    packages={}
    for name in ['numpy','scipy','scikit-learn','torch','ahrs','htm']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    cpu='not recorded'
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu=line.split(':',1)[1].strip();break
    return {'python':platform.python_version(),'packages':packages,'os':platform.platform(),'cpu':cpu,
        'execution_device':'CPU','protocol_sha256':hashlib.sha256((ROOT/'empirical_experiment_tools/protocol.json').read_bytes()).hexdigest(),
        'target_workstation_benchmarked':False}

def write_json(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
