"""Fifteen paired training runs: exact tests and seed-bootstrap confidence intervals."""
from pathlib import Path
from itertools import product
import csv,hashlib,json
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parent
SEEDS=list(range(40,55));B=50000;ANALYSIS_SEED=20261026

def holm(p):
 p=np.array(p,dtype=float);order=np.argsort(p,kind='stable');out=np.empty_like(p)
 out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
 return out

def exact_paired(a,b):
 delta=np.asarray(a)-np.asarray(b);obs=delta.mean();signs=np.array(list(product([-1.,1.],repeat=len(delta))));null=(signs*delta).mean(1)
 p=float(np.mean(np.abs(null)>=abs(obs)-1e-12))
 reference=stats.permutation_test((a,b),lambda x,y,axis:np.mean(x-y,axis=axis),permutation_type='samples',n_resamples=np.inf,alternative='two-sided',vectorized=True,batch=2048)
 assert abs(p-reference.pvalue)<1e-12
 return p

def bootstrap_pairs(a,b,n_resamples=B):
 n=len(a);index=np.random.default_rng(ANALYSIS_SEED).integers(0,n,size=(n_resamples,n))
 am=np.asarray(a)[index].mean(1);bm=np.asarray(b)[index].mean(1);gain=(np.asarray(a)-np.asarray(b))[index].mean(1)
 reference=stats.bootstrap((a,b),lambda x,y,axis:np.mean(x-y,axis=axis),paired=True,n_resamples=n_resamples,batch=n_resamples,vectorized=True,confidence_level=.95,method='percentile',random_state=np.random.default_rng(ANALYSIS_SEED))
 np.testing.assert_allclose(reference.bootstrap_distribution,gain,rtol=0,atol=1e-12)
 np.testing.assert_allclose([reference.confidence_interval.low,reference.confidence_interval.high],np.quantile(gain,[.025,.975]),rtol=0,atol=1e-12)
 return am,bm,gain

def main():
 report=ROOT/'report';runs=[];summaries=[];tests=[];scores={};gpus=set();count=0
 for ds,n in [('kvasir',100),('prostate',117)]:
  ref_ids=None
  for variant in ['full','no_gaze']:
   means=[];fingerprints=[]
   for seed in SEEDS:
    folder=report/variant/ds/f'seed{seed}';cfg=json.loads((folder/'settings.json').read_text());tr=json.loads((folder/'train_summary.json').read_text());te=json.loads((folder/'test_summary.json').read_text());ver=json.loads((folder/'verification.json').read_text())
    records=list(csv.DictReader((folder/'test_per_image.csv').open()));ids=[r['id'] for r in records]
    if ref_ids is None:ref_ids=ids
    assert ids==ref_ids and len(ids)==n
    assert cfg['seed']==seed and not cfg['segmentation_checkpoint_loaded'] and cfg['sigma_px']==(32 if ds=='kvasir' else 8)
    assert ver['fresh_training'] and ver['validation_only_checkpoint_selection'] and ver['test_masks_verified']==n
    assert ver['checkpoint_sha256']==tr['checkpoint_sha256']
    kind='prior3' if seed<=42 else ('prior5' if seed<=44 else 'current');source=report/'source_snapshots'/kind/variant/'code'
    for name,digest in cfg['sources_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,(variant,seed,name)
    d=np.array([float(x['dice']) for x in records]);value=100*float(d.mean());case_sd=float(d.std(ddof=1));assert abs(value-te['dice_mean'])<1e-9 and abs(100*case_sd-te['dice_case_sd'])<1e-9
    gpus.add(cfg['gpu']);means.append(value);fingerprints.append(cfg['initial_model_fingerprint']);count+=n
    runs.append({'dataset':ds,'variant':variant,'training_seed':seed,'n_test':n,'mean_dice_percent':value,'case_sd_coefficient':case_sd,'best_epoch':tr['best_epoch'],'epochs_completed':tr['epochs_completed'],'initial_fingerprint':fingerprints[-1],'checkpoint_sha256':tr['checkpoint_sha256'],'gpu':cfg['gpu']})
   assert len(set(fingerprints))==15
   scores[ds,variant]=np.array(means)
  a=scores[ds,'full'];b=scores[ds,'no_gaze'];delta=a-b;am,bm,gm=bootstrap_pairs(a,b)
  for variant,values,boot in [('full',a,am),('no_gaze',b,bm)]:
   low,high=np.quantile(boot,[.025,.975]);summaries.append({'dataset':ds,'variant':variant,'training_runs':15,'mean_dice_percent':float(values.mean()),'run_sd_pp':float(values.std(ddof=1)),'bootstrap_mean_se_pp':float(boot.std(ddof=1)),'mean_ci95_lower_percent':float(low),'mean_ci95_upper_percent':float(high)})
  t=stats.ttest_rel(a,b);tci=t.confidence_interval(.95);lo,hi=np.quantile(gm,[.025,.975]);p=exact_paired(a,b)
  tests.append({'dataset':ds,'paired_runs':15,'mean_gain_pp':float(delta.mean()),'gain_sd_pp':float(delta.std(ddof=1)),'gain_bootstrap_se_pp':float(gm.std(ddof=1)),'gain_ci95_lower_pp':float(lo),'gain_ci95_upper_pp':float(hi),'wins':int(np.sum(delta>0)),'ties':int(np.sum(delta==0)),'losses':int(np.sum(delta<0)),'paired_cohens_dz':float(delta.mean()/delta.std(ddof=1)),'exact_assignments':32768,'exact_p_two_sided':p,'paired_t_stat_sensitivity':float(t.statistic),'paired_t_p_sensitivity':float(t.pvalue),'paired_t_ci95_lower_sensitivity':float(tci.low),'paired_t_ci95_upper_sensitivity':float(tci.high)})
 assert len(runs)==60 and count==6510 and len(gpus)==1 and 'A100' in next(iter(gpus))
 for field,out in [('exact_p_two_sided','holm_exact_p'),('paired_t_p_sensitivity','holm_t_p_sensitivity')]:
  for row,value in zip(tests,holm([x[field] for x in tests])):row[out]=float(value)
 for row in tests:row['significant_primary_alpha05']=row['holm_exact_p']<.05
 for ds in ['kvasir','prostate']:
  for seed in SEEDS:
   group=[x for x in runs if x['dataset']==ds and x['training_seed']==seed];assert len(group)==2
 for variant in ['full','no_gaze']:
  for seed in SEEDS:
   group=[x for x in runs if x['variant']==variant and x['training_seed']==seed];assert len(group)==2 and group[0]['initial_fingerprint']==group[1]['initial_fingerprint']
 for name,rows in [('per_seed_results.csv',runs),('summary_with_ci95.csv',summaries),('paired_tests_with_ci95.csv',tests)]:
  with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 a={'seeds':SEEDS,'total_models_verified':60,'new_training_runs':40,'retained_verified_runs':20,'native_scores_verified':6510,'hardware':list(gpus),'primary_test':'Two-sided exact paired sign-flip permutation of 15 run-level mean differences; Holm correction for two datasets','confidence_intervals':'Pointwise 95% percentile paired-seed bootstrap, 50000 draws; conditional on saved train/validation/test cohorts','analysis_rng_seed':ANALYSIS_SEED,'bootstrap_replicates':B,'summary':summaries,'tests':tests}
 (ROOT/'analysis.json').write_text(json.dumps(a,indent=2),encoding='utf-8');print(json.dumps(a,indent=2))

if __name__=='__main__':main()
