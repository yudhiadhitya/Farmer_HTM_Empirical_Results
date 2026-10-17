import copy,json,time,gc
from pathlib import Path
import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.encoders import RDSE,RDSE_Parameters
from htm.bindings.algorithms import SpatialPooler,TemporalMemory
from common import ROOT,RUNS,PROTOCOL,load_records,select,evaluate,save_predictions,runtime,write_json
from reference_model import probabilities,update_readout

CONFIG=json.loads((ROOT/'empirical_experiment_tools/reference_config.json').read_text())

def encoding_cache(records):
    file=RUNS/'encodings.npz'
    if file.exists():
        with np.load(file) as z:return {r['index']:z[f'e_{r["index"]}'].copy() for r in records}
    encoders=[]
    for seed in PROTOCOL['encoder_seeds']:
        p=RDSE_Parameters();p.size=512;p.activeBits=21;p.resolution=0.01;p.seed=seed
        encoders.append(RDSE(p))
    result={}
    for r in records:
        rows=np.full((len(r['q']),84),65535,dtype=np.uint16)
        for row_index,q in enumerate(r['q']):
            parts=[enc.encode(float(q[j])).sparse.copy().astype(np.uint32)+512*j for j,enc in enumerate(encoders)]
            bits=np.concatenate(parts);assert 0<len(bits)<=84
            rows[row_index,:len(bits)]=bits
        result[r['index']]=rows
    np.savez_compressed(file,**{f'e_{i}':a for i,a in result.items()})
    print('Cached deterministic orientation SDRs; labels were not used.',flush=True)
    return result

class Backbone:
    def __init__(self,seed,no_sp=False):
        sp=copy.deepcopy(CONFIG['sp']);tm=copy.deepcopy(CONFIG['tm'])
        sp['seed']=seed;tm['seed']=seed
        sp['inputDimensions']=tuple(sp['inputDimensions']);sp['columnDimensions']=tuple(sp['columnDimensions'])
        tm['columnDimensions']=tuple(tm['columnDimensions'])
        self.sp=None if no_sp else SpatialPooler(**sp)
        self.tm=TemporalMemory(**tm);self.input=SDR(2048);self.columns=SDR(2048)
        self.no_sp=no_sp;self.width=self.tm.numberOfCells()
    def reset(self):self.tm.reset()
    def step(self,bits,learn):
        bits=bits[bits<2048].astype(np.uint32)
        self.input.sparse=bits
        if self.no_sp:self.columns.sparse=bits
        else:self.sp.compute(self.input,learn,self.columns)
        self.tm.compute(self.columns,learn=learn)
        self.tm.activateDendrites(learn)
        return (self.tm.getActiveCells().sparse.copy().astype(np.int64),
                self.tm.getPredictiveCells().sparse.copy().astype(np.int64),
                self.columns.sparse.copy().astype(np.int64))
    def fit(self,records,encodings):
        started=time.perf_counter()
        for pass_index in range(3):
            for r in select(records,'train'):
                self.reset()
                for bits in encodings[r['index']]:self.step(bits,True)
            print(f'HTM no_sp={self.no_sp} finished training pass {pass_index+1}',flush=True)
        return time.perf_counter()-started
    def extract(self,records,encodings):
        started=time.perf_counter();result={}
        before=None if self.no_sp else self.sp.getIterationLearnNum()
        for r in records:
            self.reset();endpoints=set(int(x) for x in r['endpoints']);rows=[]
            for i,bits in enumerate(encodings[r['index']]):
                active,predictive,columns=self.step(bits,False)
                if i in endpoints:rows.append((active,predictive,columns))
            assert len(rows)==len(r['endpoints'])
            result[r['index']]=rows
        if not self.no_sp:assert before==self.sp.getIterationLearnNum()
        return result,time.perf_counter()-started

def fit_readout(records,features,seed,alpha,width,use_sp=False):
    rng=np.random.default_rng(seed);weights=rng.normal(0,0.001,(width,6));bias=np.zeros(6)
    train=select(records,'train');history=[];started=time.perf_counter()
    for epoch in range(20):
        total=0;count=0
        for i in rng.permutation(len(train)):
            r=train[i]
            for row,(active,predictive,columns) in enumerate(features[r['index']]):
                if use_sp:active=columns;predictive=np.array([],dtype=np.int64)
                following=int(r['next_labels'][row]);following=None if following<0 else following
                target=int(r['labels'][r['endpoints'][row]])
                total+=update_readout(weights,bias,active,target,predictive,following,alpha,0.01);count+=1
        history.append(total/count)
    return weights,bias,history,time.perf_counter()-started

def infer(records,features,weights,bias,use_sp=False):
    result=[]
    for r in records:
        for active,predictive,columns in features[r['index']]:
            result.append(probabilities(weights,bias,columns if use_sp else active))
    return np.asarray(result)

def save_model(folder,name,seed,alpha,records,features,fit,width,use_sp=False,extra=None):
    folder.mkdir(parents=True,exist_ok=True)
    weights,bias,history,elapsed=fit
    val=select(records,'validation');test=select(records,'test')
    vp=infer(val,features,weights,bias,use_sp);tp=infer(test,features,weights,bias,use_sp)
    result={'model':name,'seed':seed,'auxiliary_weight':alpha,'validation':evaluate(val,vp),'test':evaluate(test,tp),
        'readout_training_seconds':elapsed,'readout_loss_history':history,'readout_feature_dimension':width,
        'readout_trainable_parameters':width*6+6,'runtime':runtime(),'configuration':CONFIG,**(extra or {})}
    np.savez_compressed(folder/'readout.npz',weights=weights,bias=bias)
    save_predictions(folder/'test_predictions.npz',test,tp);save_predictions(folder/'validation_predictions.npz',val,vp)
    write_json(folder/'result.json',result)
    print(f'Completed {name} seed {seed}',flush=True)
    return result

def run_seed(seed,records,encodings):
    root=RUNS/'htm'/f'seed_{seed}';root.mkdir(parents=True,exist_ok=True)
    if (root/'completed.json').exists():return
    backbone=Backbone(seed);train_seconds=backbone.fit(records,encodings)
    features,extract_seconds=backbone.extract(records,encodings)
    options=[];validation=select(records,'validation')
    for alpha in PROTOCOL['auxiliary_weights']:
        fit=fit_readout(records,features,seed,alpha,65536)
        vp=infer(validation,features,fit[0],fit[1]);score=evaluate(validation,vp)['macro_f1']
        options.append((score,-alpha,fit))
        print(f'HTM seed {seed} alpha {alpha} validation_macro_f1={score:.5f}',flush=True)
    chosen=max(options,key=lambda a:(a[0],a[1]));selected_alpha=-chosen[1]
    selection=[{'alpha':-x[1],'validation_macro_f1':x[0]} for x in options]
    extra={'backbone_training_seconds':train_seconds,'frozen_feature_extraction_seconds':extract_seconds,
           'validation_auxiliary_selection':selection,'native_backbone_hz':50,'readout_and_evaluation_hz':10}
    for _,negative_alpha,fit in options:
        alpha=-negative_alpha
        save_model(root/f'aux_{alpha:g}',f'HTM auxiliary weight {alpha:g}',seed,alpha,records,features,fit,65536,extra=extra)
    save_model(root/'full','Full HTM (validation-selected weight)',seed,selected_alpha,records,features,chosen[2],65536,extra=extra)
    sp_fit=fit_readout(records,features,seed,0.0,2048,use_sp=True)
    save_model(root/'no_tm','Without Temporal Memory',seed,0.0,records,features,sp_fit,2048,use_sp=True,extra=extra)
    del backbone,features,options,chosen;gc.collect()
    backbone=Backbone(seed,no_sp=True);ns_train=backbone.fit(records,encodings)
    features,ns_extract=backbone.extract(records,encodings)
    fit=fit_readout(records,features,seed,selected_alpha,65536)
    save_model(root/'no_sp','Without Spatial Pooler',seed,selected_alpha,records,features,fit,65536,
        extra={'backbone_training_seconds':ns_train,'frozen_feature_extraction_seconds':ns_extract,
               'auxiliary_weight_source':'validation choice of full model with the same seed',
               'native_backbone_hz':50,'readout_and_evaluation_hz':10})
    write_json(root/'completed.json',{'seed':seed,'selected_auxiliary_weight':selected_alpha,'all_planned_runs_completed':True})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=PROTOCOL['model_seeds'])
    args=parser.parse_args()
    records=load_records();encodings=encoding_cache(records)
    for seed in ([args.seed] if args.seed is not None else PROTOCOL['model_seeds']):
        print(f'Starting HTM and ablation runs for seed {seed}',flush=True)
        run_seed(seed,records,encodings)
