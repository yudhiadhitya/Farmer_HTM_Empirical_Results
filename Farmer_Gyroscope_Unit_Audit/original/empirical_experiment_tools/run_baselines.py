import argparse,copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from common import PROTOCOL,RUNS,load_records,select,evaluate,save_predictions,runtime,write_json

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

def collate(records,mean,std):
    lengths=[len(r['q']) for r in records];maximum=max(lengths)
    x=np.zeros((len(records),maximum,4),np.float32)
    target=np.full((len(records),maximum),-100,np.int64)
    padding=np.ones((len(records),maximum),bool)
    for i,r in enumerate(records):
        n=len(r['q']);x[i,:n]=(r['q']-mean)/std;padding[i,:n]=False
        target[i,r['endpoints']]=r['labels'][r['endpoints']]
    return torch.from_numpy(x),torch.from_numpy(target),torch.from_numpy(padding)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([nn.Conv1d(4,32,5,dilation=1),nn.Conv1d(32,32,5,dilation=2),nn.Conv1d(32,32,5,dilation=4)])
        self.head=nn.Linear(32,6)
    def forward(self,x,padding=None):
        h=x.transpose(1,2)
        for layer in self.layers:
            h=torch.relu(layer(torch.nn.functional.pad(h,((layer.kernel_size[0]-1)*layer.dilation[0],0))))
        return self.head(h.transpose(1,2))

class LSTM(nn.Module):
    def __init__(self):
        super().__init__();self.lstm=nn.LSTM(4,32,batch_first=True);self.head=nn.Linear(32,6)
    def forward(self,x,padding=None):
        h,_=self.lstm(x);return self.head(h)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(4,32)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,2,64,0.1,batch_first=True),1,enable_nested_tensor=False)
        self.head=nn.Linear(32,6)
    def forward(self,x,padding=None):
        b,t,d=x.shape;positions=torch.arange(t,dtype=x.dtype,device=x.device)[:,None]
        factors=torch.exp(torch.arange(0,32,2,device=x.device,dtype=x.dtype)*(-math.log(10000.0)/32))
        encoding=torch.zeros((t,32),dtype=x.dtype,device=x.device)
        encoding[:,0::2]=torch.sin(positions*factors);encoding[:,1::2]=torch.cos(positions*factors)
        hidden=self.input(x)+encoding[None]
        mask=torch.triu(torch.ones((t,t),dtype=torch.bool,device=x.device),diagonal=1)
        return self.head(self.encoder(hidden,mask=mask,src_key_padding_mask=padding))

def causal_checks():
    torch.manual_seed(42)
    a=torch.randn(1,40,4);b=a.clone();b[:,25:]=torch.randn_like(b[:,25:])*10
    for cls in (CNN,LSTM,Transformer):
        model=cls().eval()
        with torch.no_grad():
            x=model(a)[:,:25];y=model(b)[:,:25]
        assert torch.allclose(x,y,atol=1e-5),cls.__name__
    print('Causality checks passed for CNN, LSTM, and Transformer.',flush=True)

def probabilities(model,records,mean,std):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(records),4):
            chunk=records[start:start+4]
            x,target,padding=collate(chunk,mean,std)
            p=torch.softmax(model(x,padding),dim=-1).numpy()
            result.extend(p[i,r['endpoints']] for i,r in enumerate(chunk))
    return np.concatenate(result)

def run_neural(kind,seed,records):
    folder=RUNS/'baselines'/kind/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    done=folder/'result.json'
    if done.exists():return
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    train=select(records,'train');validation=select(records,'validation');test=select(records,'test')
    all_train=np.concatenate([r['q'] for r in train]);mean=all_train.mean(0);std=all_train.std(0).clip(1e-6)
    cls={'CNN':CNN,'LSTM':LSTM,'Transformer':Transformer}[kind]
    model=cls();optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    loss=nn.CrossEntropyLoss(ignore_index=-100)
    best_score=-1;best_epoch=0;best_state=None;stale=0;history=[]
    started=time.perf_counter()
    for epoch in range(1,21):
        model.train();order=rng.permutation(len(train));total=0;iterations=0
        for start in range(0,len(order),4):
            chunk=[train[i] for i in order[start:start+4]]
            x,target,padding=collate(chunk,mean,std)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x,padding);value=loss(logits.reshape(-1,6),target.reshape(-1))
            assert torch.isfinite(value)
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=float(value.detach());iterations+=1
        vp=probabilities(model,validation,mean,std);vm=evaluate(validation,vp)
        score=vm['macro_f1'];history.append({'epoch':epoch,'training_loss':total/iterations,'validation_macro_f1':score})
        print(f'{kind} seed {seed} epoch {epoch} validation_macro_f1={score:.5f}',flush=True)
        if score>best_score+1e-12:
            best_score=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=5:break
    model.load_state_dict(best_state)
    elapsed=time.perf_counter()-started
    vp=probabilities(model,validation,mean,std);tp=probabilities(model,test,mean,std)
    result={'model':kind,'seed':seed,'validation':evaluate(validation,vp),'test':evaluate(test,tp),
        'best_epoch':best_epoch,'epochs_run':len(history),'history':history,'training_seconds':elapsed,
        'trainable_parameters':sum(p.numel() for p in model.parameters()),'runtime':runtime(),
        'configuration':{'hidden_width':32,'batch_trials':4,'optimizer':'Adam','learning_rate':0.001,
                         'gradient_clip_norm':1.0,'loss':'unweighted current-activity cross-entropy','causal':True}}
    torch.save({'state_dict':best_state,'mean':mean,'std':std,'model':kind,'seed':seed},folder/'checkpoint.pt')
    save_predictions(folder/'test_predictions.npz',test,tp);save_predictions(folder/'validation_predictions.npz',validation,vp)
    write_json(done,result)
    print(f'Completed {kind} seed {seed}.',flush=True)

def run_classical(records):
    train=select(records,'train');validation=select(records,'validation');test=select(records,'test')
    x=np.concatenate([r['q'][r['endpoints']] for r in train]);y=np.concatenate([r['labels'][r['endpoints']] for r in train])
    vx=np.concatenate([r['q'][r['endpoints']] for r in validation]);tx=np.concatenate([r['q'][r['endpoints']] for r in test])
    scaler=StandardScaler().fit(x);x=scaler.transform(x);vx=scaler.transform(vx);tx=scaler.transform(tx)
    for seed in PROTOCOL['model_seeds']:
        folder=RUNS/'baselines'/'LogisticRegression'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        if not (folder/'result.json').exists():
            options=[];started=time.perf_counter()
            for c in PROTOCOL['logistic_C_candidates']:
                model=LogisticRegression(C=c,max_iter=2000,random_state=seed).fit(x,y)
                vp=model.predict_proba(vx);score=evaluate(validation,vp)['macro_f1']
                options.append((score,-c,model))
            _,negc,model=max(options,key=lambda a:(a[0],a[1]))
            vp=model.predict_proba(vx);tp=model.predict_proba(tx)
            result={'model':'Logistic regression','seed':seed,'selected_C':-negc,
                'validation_C_scores':[{'C':-a[1],'macro_f1':a[0]} for a in options],
                'validation':evaluate(validation,vp),'test':evaluate(test,tp),'training_seconds':time.perf_counter()-started,'runtime':runtime()}
            save_predictions(folder/'test_predictions.npz',test,tp);write_json(folder/'result.json',result)
        folder=RUNS/'baselines'/'TimeOnly'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        if not (folder/'result.json').exists():
            clock=lambda rows:np.concatenate([r['times'][r['endpoints']] for r in rows])[:,None]
            started=time.perf_counter()
            model=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=30,n_jobs=2,random_state=seed).fit(clock(train),y)
            vp=model.predict_proba(clock(validation));tp=model.predict_proba(clock(test))
            result={'model':'Elapsed-time control','seed':seed,'validation':evaluate(validation,vp),'test':evaluate(test,tp),
                'training_seconds':time.perf_counter()-started,'runtime':runtime(),
                'configuration':PROTOCOL['time_control']}
            save_predictions(folder/'test_predictions.npz',test,tp);write_json(folder/'result.json',result)
    print('Completed logistic regression and elapsed-time controls.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['CNN','LSTM','Transformer','classical'],required=True)
    args=parser.parse_args();records=load_records()
    causal_checks()
    if args.model=='classical':run_classical(records)
    else:
        for seed in PROTOCOL['model_seeds']:run_neural(args.model,seed,records)
