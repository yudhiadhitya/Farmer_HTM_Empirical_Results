"""Recompute the reported metrics and summarize all prespecified experiments."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from common import ROOT,RUNS,PROTOCOL,load_records,select,evaluate,write_json

METRICS=['accuracy','macro_precision','macro_recall','macro_f1',
         'probability_mae','probability_mse','probability_rmse',
         'transition_error','false_switches_per_minute']
MODELS={
    'Full HTM':'htm/seed_{seed}/full',
    'Logistic regression':'baselines/LogisticRegression/seed_{seed}',
    'CNN':'baselines/CNN/seed_{seed}',
    'LSTM':'baselines/LSTM/seed_{seed}',
    'Transformer':'baselines/Transformer/seed_{seed}',
    'Elapsed time only':'baselines/TimeOnly/seed_{seed}',
    'HTM lambda 0':'htm/seed_{seed}/aux_0',
    'HTM lambda 0.1':'htm/seed_{seed}/aux_0.1',
    'HTM lambda 0.3':'htm/seed_{seed}/aux_0.3',
    'Without TM':'htm/seed_{seed}/no_tm',
    'Without SP':'htm/seed_{seed}/no_sp',
}

def main():
    records=load_records();test=select(records,'test')
    expected_labels=np.concatenate([r['labels'][r['endpoints']] for r in test])
    verified=[];all_results={};summary={}
    for name,pattern in MODELS.items():
        results=[]
        for seed in PROTOCOL['model_seeds']:
            folder=RUNS/pattern.format(seed=seed)
            result=json.loads((folder/'result.json').read_text())
            assert result['seed']==seed
            with np.load(folder/'test_predictions.npz') as z:
                assert np.array_equal(z['labels'],expected_labels)
                assert np.array_equal(z['subject'],np.concatenate([
                    np.repeat(r['subject'],len(r['endpoints'])) for r in test]))
                recomputed=evaluate(test,z['probabilities'])
            for key in METRICS:
                # Stored probabilities are float32; original scoring used float64.
                assert abs(recomputed[key]-result['test'][key])<2e-7,(name,seed,key)
            assert recomputed['confusion_matrix']==result['test']['confusion_matrix']
            assert result['runtime']['protocol_sha256']==hashlib.sha256(
                (ROOT/'empirical_experiment_tools/protocol.json').read_bytes()).hexdigest()
            results.append(result);verified.append({'model':name,'seed':seed,'passed':True})
        all_results[name]=results
        summary[name]={key:{'mean':float(np.mean([r['test'][key] for r in results])),
                            'sample_sd':float(np.std([r['test'][key] for r in results],ddof=1))}
                       for key in METRICS}
        summary[name]['selected_auxiliary_weights']=[r.get('auxiliary_weight') for r in results]
        summary[name]['per_class']={key:{'mean':np.mean([r['test']['per_class'][key] for r in results],axis=0).tolist(),
                                       'sample_sd':np.std([r['test']['per_class'][key] for r in results],axis=0,ddof=1).tolist()}
                                    for key in ['precision','recall','f1']}
        summary[name]['per_class']['support']=results[0]['test']['per_class']['support']
        cm=np.mean([r['test']['confusion_matrix'] for r in results],axis=0)
        summary[name]['mean_confusion_matrix']=cm.tolist()
        summary[name]['row_normalized_mean_confusion_matrix']=(cm/cm.sum(1,keepdims=True)).tolist()
        summary[name]['per_test_subject']={subject:{
            'mean_macro_f1':float(np.mean([r['test']['per_test_subject'][subject]['macro_f1'] for r in results])),
            'sample_sd':float(np.std([r['test']['per_test_subject'][subject]['macro_f1'] for r in results],ddof=1))}
            for subject in PROTOCOL['test_subjects']}
        full=all_results.get('Full HTM')
        if full is not None:
            diff=np.array([r['test']['macro_f1'] for r in results])-np.array([r['test']['macro_f1'] for r in full])
            summary[name]['paired_macro_f1_difference_from_full']={
                'mean_percentage_points':float(diff.mean()*100),
                'sample_sd_percentage_points':float(diff.std(ddof=1)*100)}
    output=RUNS/'summary';output.mkdir(exist_ok=True)
    write_json(output/'aggregate_metrics.json',summary)
    with (output/'aggregate_metrics.csv').open('w',newline='') as f:
        fields=['model']+[f'{key}_{stat}' for key in METRICS for stat in ['mean','sample_sd']]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,values in summary.items():
            writer.writerow({'model':name,**{f'{k}_{s}':values[k][s] for k in METRICS for s in ['mean','sample_sd']}})
    groups={s:{r['subject'] for r in select(records,s)} for s in ['train','validation','test']}
    assert not groups['train']&groups['validation'] and not groups['train']&groups['test'] and not groups['validation']&groups['test']
    hash_groups={}
    for r in records:hash_groups.setdefault(r['source_sha256'],[]).append(r)
    assert all(len({r['split'] for r in rows})==1 for rows in hash_groups.values())
    duplicate_groups=[[r['clip_id'] for r in rows] for rows in hash_groups.values() if len(rows)>1]
    counts=[]
    with np.load(RUNS/'encodings.npz') as z:
        for key in z.files:counts.extend(np.sum(z[key]<2048,axis=1).tolist())
    verification={'all_prespecified_runs':len(verified),'prediction_and_metric_checks':verified,
                  'participant_disjoint':True,'no_identical_recordings_across_splits':True,
                  'unique_recording_hashes':len(hash_groups),'within_split_duplicate_groups':duplicate_groups,
                  'test_evaluation_frames':len(expected_labels),
                  'encoder_unique_active_bits':{'minimum':min(counts),'maximum':max(counts),'mean':float(np.mean(counts))},
                  'statistical_scope':'Mean and sample SD across three model seeds on one fixed participant split; no significance claim.'}
    write_json(output/'verification.json',verification)
    make_figures(summary,output)
    print(json.dumps({'models':list(summary),'verified_runs':len(verified),'encoder':verification['encoder_unique_active_bits']},indent=2))

def make_figures(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=list(MODELS)[:6]
    fig,ax=plt.subplots(figsize=(6.7,3.5),layout='constrained')
    means=[summary[n]['macro_f1']['mean']*100 for n in names]
    sd=[summary[n]['macro_f1']['sample_sd']*100 for n in names]
    bars=ax.barh(np.arange(6),means,xerr=sd,color=['#555555']+['#909090']*4+['#CCCCCC'],capsize=3)
    bars[-1].set_hatch('///')
    ax.set_yticks(np.arange(6),['Full HTM','Logistic regression','CNN','LSTM','Transformer','Elapsed time only'])
    ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Test macro-F1 (%)')
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    for y,v in enumerate(means):ax.text(min(v+2,96),y,f'{v:.2f}',va='center',fontsize=9)
    fig.savefig(output/'baseline_macro_f1.png',dpi=300);fig.savefig(output/'baseline_macro_f1.pdf');plt.close(fig)
    labels=[str(i) for i in range(6)]
    fig,axes=plt.subplots(1,2,figsize=(6.7,3.5),layout='constrained')
    for ax,name in zip(axes,['Full HTM','Transformer']):
        cm=np.asarray(summary[name]['row_normalized_mean_confusion_matrix'])*100
        ax.imshow(cm,vmin=0,vmax=100,cmap='Greys')
        for i in range(6):
            for j in range(6):ax.text(j,i,f'{cm[i,j]:.0f}',ha='center',va='center',fontsize=10,color='white' if cm[i,j]>55 else 'black')
        ax.set_xticks(range(6),labels,fontsize=10);ax.set_yticks(range(6),labels,fontsize=10)
        ax.set_xlabel('Predicted activity');ax.set_ylabel('Annotated activity');ax.set_title(name,fontsize=11)
    fig.savefig(output/'class_confusion_matrices.png',dpi=300);fig.savefig(output/'class_confusion_matrices.pdf');plt.close(fig)

if __name__=='__main__':main()
