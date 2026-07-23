"""
Version B modeling core (imported by the notebook, and runnable standalone to validate).

Question: can we FORECAST a domain's FINAL forgetting from cheap static features + a short
WARM-UP of fine-tuning dynamics -- for a domain the models have never seen?

- Row = one (domain, learning_rate) fine-tuning run.
- Static features  : semantic_dist, clip_embed_dist, spectral_dist, frechet_dist, log10(lr)
- Warm-up features : from the first K epochs -> forgetting@K, train_loss@K, train_acc@K,
                     param_drift@K, emb_drift@K, and the slope of forgetting over 1..K
- Target           : forgetting at the final epoch
- Evaluation       : LEAVE-ONE-DOMAIN-OUT (all LRs of the held-out domain are unseen in training)
- Ablation         : static-only vs static+warmup  (does watching training help?)
"""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

STATIC=["semantic_dist","clip_embed_dist","spectral_dist","frechet_dist","log_lr"]

def load(out="outputs/dynamics"):
    cur=pd.read_csv(f"{out}/dynamics_curves.csv"); feat=pd.read_csv(f"{out}/features.csv")
    return cur, feat

def build_table(cur, feat, K=5):
    """one row per (domain,lr); warm-up summarised over first K epochs; target = final forgetting."""
    final_ep=cur.epoch.max()
    rows=[]
    for (dom,lr),g in cur.groupby(["domain","lr"]):
        g=g.sort_values("epoch")
        if g.epoch.max()<final_ep: continue                 # skip incomplete runs
        w=g[g.epoch<=K]
        fslope=np.polyfit(w.epoch, w.forgetting,1)[0] if len(w)>=2 else 0.0
        fr=feat[feat.domain==dom].iloc[0]
        rows.append(dict(domain=dom, family=fr["family"], lr=lr, log_lr=np.log10(lr),
            semantic_dist=fr.semantic_dist, clip_embed_dist=fr.clip_embed_dist,
            spectral_dist=fr.spectral_dist, frechet_dist=fr.frechet_dist,
            f_warm=w.forgetting.iloc[-1], loss_warm=w.train_loss.iloc[-1], acc_warm=w.train_acc.iloc[-1],
            pdrift_warm=w.param_drift.iloc[-1], edrift_warm=w.emb_drift.iloc[-1], fslope_warm=fslope,
            target=g[g.epoch==final_ep].forgetting.iloc[0]))
    return pd.DataFrame(rows)

WARMUP=["f_warm","loss_warm","acc_warm","pdrift_warm","edrift_warm","fslope_warm"]

def models():
    return {
        "Ridge (linear)": lambda: RidgeCV(alphas=np.logspace(-3,3,13)),
        "Random Forest":  lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0),
        "Grad Boosting":  lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, random_state=0),
        "MLP (neural net)": lambda: MLPRegressor(hidden_layer_sizes=(32,16), max_iter=4000,
                                                 early_stopping=False, random_state=0),
    }

def lodo(df, cols, make):
    """leave-one-domain-out: predict each domain's runs from all OTHER domains. returns preds aligned to df."""
    pred=np.full(len(df), np.nan)
    for dom in df.domain.unique():
        tr=df[df.domain!=dom]; te=df[df.domain==dom]
        sc=StandardScaler().fit(tr[cols])
        m=make(); m.fit(sc.transform(tr[cols]), tr.target)
        pred[te.index]=m.predict(sc.transform(te[cols]))
    return pred

def evaluate(df, cols):
    out={}
    for name,make in models().items():
        p=lodo(df.reset_index(drop=True), cols, make)
        t=df.target.values
        mae=np.mean(np.abs(p-t)); r=spearmanr(p,t)[0]
        ss=1-np.sum((p-t)**2)/np.sum((t-t.mean())**2)
        out[name]=dict(MAE=float(mae), Spearman=float(r), R2=float(ss))
    return out

def report(df):
    print(f"rows={len(df)}  domains={df.domain.nunique()}  (LODO folds)")
    for label,cols in [("STATIC only",STATIC), ("STATIC + WARM-UP",STATIC+WARMUP)]:
        print(f"\n### {label} ###   {'model':18s}{'MAE':>8s}{'Spearman':>10s}{'R2':>8s}")
        for name,m in evaluate(df,cols).items():
            print(f"    {name:18s}{m['MAE']:8.4f}{m['Spearman']:10.3f}{m['R2']:8.3f}")

def warmup_curve(cur, feat):
    """held-out forecasting skill (Ridge) as a function of warm-up length K -> how early can we tell?"""
    res=[]
    for K in range(1, cur.epoch.max()):
        df=build_table(cur,feat,K=K)
        if df.domain.nunique()<4: continue
        p=lodo(df.reset_index(drop=True), STATIC+WARMUP, models()["Grad Boosting"])
        t=df.target.values; res.append((K, 1-np.sum((p-t)**2)/np.sum((t-t.mean())**2)))
    return pd.DataFrame(res, columns=["K","held_out_R2"])

SEQFEATS=["forgetting","train_loss","train_acc","param_drift","emb_drift"]

def build_sequences(cur, feat, K=5):
    """for the LSTM: per-run sequence of the first K epochs (5 dynamics signals) + static feats -> final forgetting."""
    final_ep=cur.epoch.max(); X,Xs,y,doms=[],[],[],[]
    for (dom,lr),g in cur.groupby(["domain","lr"]):
        g=g.sort_values("epoch")
        if g.epoch.max()<final_ep: continue
        w=g[g.epoch<=K]
        if len(w)<K: continue
        fr=feat[feat.domain==dom].iloc[0]
        X.append(w[SEQFEATS].values.astype("float32"))
        Xs.append(np.array([fr.semantic_dist,fr.clip_embed_dist,fr.spectral_dist,fr.frechet_dist,np.log10(lr)],dtype="float32"))
        y.append(float(g[g.epoch==final_ep].forgetting.iloc[0])); doms.append(dom)
    return np.array(X), np.array(Xs), np.array(y,dtype="float32"), np.array(doms)

def lstm_lodo(cur, feat, K=5, epochs=300):
    """leave-one-domain-out forecasting with a small LSTM over the warm-up sequence + static feats."""
    import torch, torch.nn as nn
    X,Xs,y,doms=build_sequences(cur,feat,K)
    if len(set(doms))<4: return None
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.lstm=nn.LSTM(X.shape[-1],16,batch_first=True)
            s.head=nn.Sequential(nn.Linear(16+Xs.shape[-1],16),nn.ReLU(),nn.Linear(16,1))
        def forward(s,seq,stat):
            _,(h,_)=s.lstm(seq); return s.head(torch.cat([h[-1],stat],1)).squeeze(-1)
    pred=np.full(len(y),np.nan)
    for dom in set(doms):
        tr=doms!=dom; te=doms==dom
        # Fit every normalization statistic on the training domains only. Computing these
        # values globally would let the held-out domain influence its own evaluation.
        xtr_flat=X[tr].reshape(-1,X.shape[-1])
        xm,xsd=xtr_flat.mean(0),xtr_flat.std(0)+1e-6
        sm,ssd=Xs[tr].mean(0),Xs[tr].std(0)+1e-6
        Xn=(X-xm)/xsd; Xsn=(Xs-sm)/ssd
        torch.manual_seed(0)
        net=Net(); opt=torch.optim.Adam(net.parameters(),lr=5e-3,weight_decay=1e-4); lossf=nn.MSELoss()
        seq=torch.tensor(Xn[tr]); st=torch.tensor(Xsn[tr]); tt=torch.tensor(y[tr])
        for _ in range(epochs):
            opt.zero_grad(); out=net(seq,st); loss=lossf(out,tt); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pred[te]=net(torch.tensor(Xn[te]),torch.tensor(Xsn[te])).numpy()
    t=y; mae=np.mean(np.abs(pred-t)); r=spearmanr(pred,t)[0]; ss=1-np.sum((pred-t)**2)/np.sum((t-t.mean())**2)
    return dict(MAE=float(mae),Spearman=float(r),R2=float(ss)), pred, y, doms

if __name__=="__main__":
    cur,feat=load()
    print("epochs done per run (min):", cur.groupby(['domain','lr']).epoch.max().min(),
          "| complete runs:", (cur.groupby(['domain','lr']).epoch.max()==cur.epoch.max()).sum())
    df=build_table(cur,feat,K=5)
    if df.domain.nunique()>=4:
        report(df)
        print("\nwarm-up length vs held-out R2:"); print(warmup_curve(cur,feat).to_string(index=False))
    else:
        print("not enough complete domains yet:", df.domain.nunique())
