"""
MERGED EXPERIMENT (Track H rigor x Track R data) — element-specific bipartite
interface persistent homology on Platinum drug-resistance.

Why this is the one experiment both cancelled sessions left unrun:
  * Track R (resistance) built a *single all-atom* interface PH on Platinum and
    scored AUROC 0.425 (below random). That is exactly the *vanilla* formulation
    Track H (hERG) showed is weak: vanilla ligand PH = 0.71, while ELEMENT-SPECIFIC
    PH (Cang & Wei channels) lifted it to 0.85. Track R never tried element channels.
  * Track H proved the element-channel + fixed-range vectorizer machinery but never
    had the both-structure data to test the *interface* (track C) construct.
This script ports Track H's element-channel idea + fixed-range rigorous vectorizer
onto Track R's bipartite opposition-distance interface, and judges it with a
powered, paired-bootstrap gate against the SAME cheap mCSM-style baseline that
scored 0.62, on identical group-CV rows.

Decision rule (pre-registered, before looking at test scores):
  PASS iff element-specific interface PH (T), or T concatenated with the cheap
  features (T+S), beats the cheap baseline (S) on AUROC with a paired-bootstrap
  95% CI on the delta that excludes 0. Also report MCC. If it merely matches 0.42-
  0.50 -> the 0.425 was not an implementation artifact; topology is dead on
  resistance too, and the thesis is buried on both targets.
"""
import warnings; warnings.filterwarnings("ignore")
import os, re, json, urllib.request
import numpy as np, pandas as pd
from ripser import ripser
from persim import PersistenceImager
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# ----- geometry / filtration constants -----
CUT   = 8.0      # interface = protein atoms within CUT A of any ligand atom
BIG   = 1e6      # within-affiliation distance (bipartite: only cross links exist)
THRESH= 16.0     # ripser filtration cap on the opposition-distance complex
SKIP  = {"HOH","WAT","NA","CL","K","MG","ZN","CA","SO4","PO4","GOL","EDO","ACT",
         "DMS","BME","MES","TRS","IOD","FMT","PEG","PG4","NO3","CO3"}

# ----- Track-H-style fixed-range vectorizer (no train-fit leakage) -----
_BIRTH_RANGE=(0.0,16.0); _PERS_RANGE=(0.0,16.0); _PIXEL=4.0   # -> 4x4 = 16 px
_H0_BINS=8; _H0_MAX=16.0
_IMGR=PersistenceImager(birth_range=_BIRTH_RANGE,pers_range=_PERS_RANGE,pixel_size=_PIXEL)
_IMG_LEN=int(round((_BIRTH_RANGE[1]-_BIRTH_RANGE[0])/_PIXEL))*int(round((_PERS_RANGE[1]-_PERS_RANGE[0])/_PIXEL))

def _summary(dgm):
    if dgm.shape[0]==0: return np.zeros(4)
    p=dgm[:,1]-dgm[:,0]
    return np.array([len(p),p.max(),p.sum(),p.mean()])

def _h0hist(dgm):
    if dgm.shape[0]==0: return np.zeros(_H0_BINS)
    h,_=np.histogram(np.clip(dgm[:,1],0,_H0_MAX),bins=_H0_BINS,range=(0,_H0_MAX))
    return h.astype(float)

def _img(dgm):
    if dgm.shape[0]==0: return np.zeros(_IMG_LEN)
    return np.asarray(_IMGR.transform(dgm,skew=True),float).ravel()

def vectorize_dm(M):
    """Opposition distance matrix -> fixed-length PH vector (H0 hist+stats, H1 image+stats)."""
    if M is None or M.shape[0]<2:
        return np.zeros(_H0_BINS+4+_IMG_LEN+4)
    dgms=ripser(M,distance_matrix=True,maxdim=1,thresh=THRESH)["dgms"]
    d0=dgms[0]; d0=d0[np.isfinite(d0).all(1)] if d0.size else d0.reshape(-1,2)
    d1=dgms[1] if len(dgms)>1 else np.empty((0,2)); d1=d1[np.isfinite(d1).all(1)] if d1.size else d1.reshape(-1,2)
    return np.concatenate([_h0hist(d0),_summary(d0),_img(d1),_summary(d1)])

_CHAN_LEN=_H0_BINS+4+_IMG_LEN+4

# ----- element-specific bipartite channels (the Track-H upgrade Track R skipped) -----
# (ligand element subset, protein element subset); "ALL" = no restriction.
# ("ALL","ALL") reproduces Track R's vanilla interface as a strict sub-block, so any
# lift over 0.425 is attributable to the element channels.
CHANNELS=[("ALL","ALL"),("ALL","C"),("ALL","N"),("ALL","O"),
          (("N","O"),"ALL"),(("N","O"),("N","O"))]

def opp_dm(lig,inter):
    nl=len(lig); n=nl+len(inter); M=np.full((n,n),BIG)
    cross=np.sqrt(((lig[:,None,:]-inter[None,:,:])**2).sum(-1))
    M[:nl,nl:]=cross; M[nl:,:nl]=cross.T; np.fill_diagonal(M,0.0)
    return M

def _sel(coords,elems,which):
    if which=="ALL": return coords
    keep=np.isin(elems,list(which) if isinstance(which,tuple) else [which])
    return coords[keep]

def esph_interface_vector(lig,lig_e,inter,inter_e):
    parts=[]
    for le,pe in CHANNELS:
        L=_sel(lig,lig_e,le); P=_sel(inter,inter_e,pe)
        if len(L)<2 or len(P)<2: parts.append(np.zeros(_CHAN_LEN)); continue
        parts.append(vectorize_dm(opp_dm(L,P)))
    return np.concatenate(parts)

# ----- PDB parsing WITH element symbols -----
def _elem(line):
    e=line[76:78].strip()
    if e and e.isalpha(): return e.capitalize()
    nm=line[12:16].strip()
    return re.sub(r'[^A-Za-z]','',nm)[:1].upper() or "C"

def fetch_pdb(pid):
    p=f"structures/{pid}.pdb"
    if os.path.exists(p) and os.path.getsize(p)>0: return p
    try: urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pid}.pdb",p); return p
    except Exception: return None

def parse(path):
    prot=[]; prot_e=[]; het={}
    for L in open(path):
        rec=L[:6].strip()
        if rec not in ("ATOM","HETATM"): continue
        try: x,y,z=float(L[30:38]),float(L[38:46]),float(L[46:54])
        except ValueError: continue
        e=_elem(L)
        if e=="H": continue
        res=L[17:20].strip(); ch=L[21]
        if rec=="ATOM": prot.append((x,y,z)); prot_e.append(e)
        elif res not in SKIP: het.setdefault(res,[]).append((ch,(x,y,z),e))
    return np.array(prot,float), np.array(prot_e), het

def interface(path,lig_code,chain):
    prot,prot_e,het=parse(path)
    if len(prot)==0 or not het: return None
    atoms=None
    if lig_code and lig_code.upper() in het: atoms=het[lig_code.upper()]
    if atoms is None: atoms=max(het.values(),key=len)
    sub=[(c,e) for ch,c,e in atoms if (not chain or ch==chain)] or [(c,e) for ch,c,e in atoms]
    lig=np.array([c for c,e in sub],float); lig_e=np.array([e for c,e in sub])
    if len(lig)<4: return None
    d=np.sqrt(((prot[:,None,:]-lig[None,:,:])**2).sum(-1))
    m=d.min(1)<=CUT; inter=prot[m]; inter_e=prot_e[m]
    if len(inter)<4: return None
    return lig,lig_e,inter,inter_e

# ----- cheap mCSM-style baseline (reconstructs Track R's 0.62 feature set) -----
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
    'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
VOL={'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,
     'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
CHG={'D':-1,'E':-1,'K':1,'R':1,'H':0.5}
def aa_deltas(mut):
    m=re.match(r'^([A-Z])(\d+)([A-Z])',str(mut).strip())
    if not m: return 0.0,0.0,0.0
    w,_,t=m.groups()
    return (KD.get(t,0)-KD.get(w,0), VOL.get(t,0)-VOL.get(w,0), CHG.get(t,0)-CHG.get(w,0))

def simple_features(r):
    dist=pd.to_numeric(r.get("mut.distance_to_lig"),errors="coerce")
    duet=pd.to_numeric(r.get("mut.duet_stability"),errors="coerce")
    mw  =pd.to_numeric(r.get("lig.mol_weight"),errors="coerce")
    inbs=1.0 if str(r.get("mut.in_binding_site")).upper()=="YES" else 0.0
    dh,dv,dc=aa_deltas(r.get("mutation"))
    return np.nan_to_num(np.array([dist,inbs,duet,mw,dh,dv,dc],float),nan=0.0)

# ----- powered evaluation: grouped OOF predictions + paired bootstrap gate -----
def oof_proba(X,y,groups,n_splits):
    gkf=GroupKFold(n_splits=n_splits); pr=np.zeros(len(y))
    for tr,te in gkf.split(X,y,groups):
        clf=RandomForestClassifier(n_estimators=600,n_jobs=-1,random_state=0,class_weight="balanced")
        clf.fit(X[tr],y[tr]); pr[te]=clf.predict_proba(X[te])[:,1]
    return pr

def paired_bootstrap(y,pa,pb,n=2000):
    """delta = AUROC(b) - AUROC(a); returns (delta, ci_lo, ci_hi, p(b>a))."""
    rng=np.random.RandomState(0); idx=np.arange(len(y)); deltas=[]
    for _ in range(n):
        s=rng.choice(idx,len(idx),replace=True)
        if len(np.unique(y[s]))<2: continue
        deltas.append(roc_auc_score(y[s],pb[s])-roc_auc_score(y[s],pa[s]))
    deltas=np.array(deltas)
    return (roc_auc_score(y,pb)-roc_auc_score(y,pa),
            float(np.percentile(deltas,2.5)),float(np.percentile(deltas,97.5)),
            float((deltas>0).mean()))

def main():
    df=pd.read_csv("data/platinum.csv")
    kw=pd.to_numeric(df["affin.k_wt"],errors="coerce"); km=pd.to_numeric(df["affin.k_mt"],errors="coerce")
    df=df[(kw>0)&(km>0)].copy(); df["resist"]=((km/kw)>=10).astype(int)
    pat=re.compile(r'^[0-9A-Za-z]{4}$')
    both=df[df["mut.wt_pdb"].astype(str).str.match(pat)&df["mut.mt_pdb"].astype(str).str.match(pat)].copy()
    print(f"mutations with BOTH structures: {len(both)}",flush=True)
    ligcol="lig.pdbe" if "lig.pdbe" in both else "affin.lig_id"

    XT,XS,y,groups=[],[],[],[]
    for i,(_,r) in enumerate(both.iterrows()):
        wt=fetch_pdb(str(r["mut.wt_pdb"])); mt=fetch_pdb(str(r["mut.mt_pdb"]))
        if not wt or not mt: continue
        cw=interface(wt,str(r.get(ligcol,"")),str(r["affin.chain"]))
        cm=interface(mt,str(r.get(ligcol,"")),str(r["affin.chain"]))
        if cw is None or cm is None: continue
        vw=esph_interface_vector(*cw); vm=esph_interface_vector(*cm)
        XT.append(np.concatenate([vw,vm,vm-vw]))
        XS.append(simple_features(r)); y.append(int(r["resist"]))
        groups.append(str(r["mut.uniprot"]) if pd.notna(r["mut.uniprot"]) else str(r["affin.pdb_id"]))
        if (i+1)%25==0: print(f"  {i+1}/{len(both)} processed, {len(XT)} usable",flush=True)
    XT=np.nan_to_num(np.array(XT)); XS=np.array(XS); y=np.array(y); groups=np.array(groups)
    XTS=np.concatenate([XS,XT],axis=1)
    ng=len(np.unique(groups)); nsp=min(10,ng)
    print(f"\nusable: {len(y)} | positives: {int(y.sum())} | groups: {ng} | dims T={XT.shape[1]} S={XS.shape[1]} | folds={nsp}",flush=True)

    pS =oof_proba(XS ,y,groups,nsp)
    pT =oof_proba(XT ,y,groups,nsp)
    pTS=oof_proba(XTS,y,groups,nsp)
    def m(p):
        return roc_auc_score(y,p), matthews_corrcoef(y,(p>=0.5).astype(int))
    aS,cS=m(pS); aT,cT=m(pT); aTS,cTS=m(pTS)
    # in-data existing predictor (duet stability), single feature, sign-agnostic
    duet=XS[:,2]; aD=max(roc_auc_score(y,duet),roc_auc_score(y,-duet))

    gTvsS =paired_bootstrap(y,pS,pT)
    gTSvsS=paired_bootstrap(y,pS,pTS)

    res={
      "n":int(len(y)),"positives":int(y.sum()),"groups":int(ng),"folds":int(nsp),
      "dims":{"T":int(XT.shape[1]),"S":int(XS.shape[1])},
      "channels":[f"{le}|{pe}" for le,pe in CHANNELS],
      "auroc":{"S_cheap":aS,"T_esph_interface":aT,"TS_concat":aTS,"duet_single":aD},
      "mcc":{"S_cheap":cS,"T_esph_interface":cT,"TS_concat":cTS},
      "gate_T_vs_S":{"delta":gTvsS[0],"ci_lo":gTvsS[1],"ci_hi":gTvsS[2],"p_T_gt_S":gTvsS[3],"passes":gTvsS[1]>0},
      "gate_TS_vs_S":{"delta":gTSvsS[0],"ci_lo":gTSvsS[1],"ci_hi":gTSvsS[2],"p_TS_gt_S":gTSvsS[3],"passes":gTSvsS[1]>0},
      "reference":{"track_R_vanilla_interface":0.425,"simple_features_bar":0.62,"mcsm_lig_published":0.70},
    }
    res["verdict_pass"]=bool(res["gate_T_vs_S"]["passes"] or res["gate_TS_vs_S"]["passes"])
    os.makedirs("results",exist_ok=True)
    json.dump(res,open("results/esph_interface_gate.json","w"),indent=2)
    print("="*64,flush=True)
    print(f"AUROC  cheap-S={aS:.3f}  ESPH-interface-T={aT:.3f}  T+S={aTS:.3f}  (duet alone {aD:.3f})",flush=True)
    print(f"MCC    cheap-S={cS:.3f}  ESPH-interface-T={cT:.3f}  T+S={cTS:.3f}",flush=True)
    print(f"GATE T vs S : delta={gTvsS[0]:+.3f}  95%CI[{gTvsS[1]:+.3f},{gTvsS[2]:+.3f}]  pass={res['gate_T_vs_S']['passes']}",flush=True)
    print(f"GATE T+S vs S: delta={gTSvsS[0]:+.3f}  95%CI[{gTSvsS[1]:+.3f},{gTSvsS[2]:+.3f}]  pass={res['gate_TS_vs_S']['passes']}",flush=True)
    print(f"reference: Track-R vanilla 0.425 | bar 0.62 | mCSM-lig 0.70",flush=True)
    print(f"VERDICT_PASS = {res['verdict_pass']}",flush=True)
    print("wrote results/esph_interface_gate.json",flush=True)

if __name__=="__main__":
    main()
