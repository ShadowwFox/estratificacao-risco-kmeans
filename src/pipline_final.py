#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIPELINE FINAL — TCC MBA em IA e Big Data (USP/ICMC)
Estratificação de risco e fenotipagem exploratória (K-Means) com indicadores de
regulação emocional (DERS, EQR, LESS-II) e validação externa pela DASS-21.

Fluxo: dados brutos (itens) -> higienização -> escoragem (chaves publicadas) ->
z-score (22 dimensões; DASS-21 FORA do agrupamento) -> K-Means (K=2 e K=3) ->
estabilidade (bootstrap, GMM, MICE) -> testes não paramétricos -> PCA -> RF/SHAP.

Uso:  DATA_DIR=<pasta com os CSV>  OUT_DIR=<pasta de saída>  python pipeline_final.py
Arquivos de entrada esperados em DATA_DIR:
  BANCO_DE_DADOS_REFORMADO.csv   (exportação de 242 registros com resposta)
  USP.CSV                        (exportação de 263 registros; contém os 23 desistentes iniciais)
Semente global = 42; K-Means: init='k-means++', n_init=10.
"""
import os, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DATA = os.environ.get("DATA_DIR", "/mnt/project")
OUT = os.environ.get("OUT_DIR", "/mnt/user-data/outputs/pipeline_saida")
FIG, TAB = os.path.join(OUT, "figuras"), os.path.join(OUT, "tabelas")
for d in (OUT, FIG, TAB):
    os.makedirs(d, exist_ok=True)
SEED, N_INIT, N_BOOT, N_IMP = 42, 10, 100, 5
R = {}  # resultados numéricos (vão para results.json)

# ------------------------------------------------------------------ utilidades
def read(path):
    d = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    d.columns = [c.strip() for c in d.columns]
    return d.loc[:, ~d.columns.str.startswith("Unnamed")]

def num(s):
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")

def br(x, d=2, thousands=False):
    s = f"{x:,.{d}f}" if thousands else f"{x:.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def pfmt(p):
    return "< 0,001" if p < 0.001 else br(p, 3)

def eff_label(r):
    a = abs(r)
    return "irrelevante" if a < .1 else "pequeno" if a < .3 else "médio" if a < .5 else "grande"

DERS_ITEMS = [f"ders_{i}" for i in range(1, 37)]
EQR_ITEMS = [f"eqr_{i}" for i in range(1, 11)]
LESS_ITEMS = [f"less_{i}" for i in range(1, 29)]
DASS_ITEMS = [f"dass_{i}" for i in range(1, 22)]
ER74 = DERS_ITEMS + EQR_ITEMS + LESS_ITEMS
ALL95 = ER74 + DASS_ITEMS
RANGE = {"ders": (1, 5), "eqr": (1, 7), "less": (1, 6), "dass": (0, 3)}
LO = np.array([RANGE[c.split("_")[0]][0] for c in ER74], float)
HI = np.array([RANGE[c.split("_")[0]][1] for c in ER74], float)

# CHAVES DE ESCORAGEM (Apêndice A) -------------------------------------------------
DERS_REV = {1, 2, 6, 7, 8, 10, 17, 20, 22, 24, 34}          # Gratz & Roemer (2004); confirmado nos dados
DERS_MAP = {"NonAcceptance": [11, 12, 21, 23, 25, 29], "Goals": [13, 18, 20, 26, 33],
            "Impulse": [3, 14, 19, 24, 27, 32], "Awareness": [2, 6, 8, 10, 17, 34],
            "Strategies": [15, 16, 22, 28, 30, 31, 35, 36], "Clarity": [1, 4, 5, 7, 9]}
LESS_REV = {4, 6, 14, 15, 19, 24, 25, 26}
LESS_MAP = {"acusacao": [8, 21], "baixa_expressao": [4, 15], "baixo_consenso": [1, 25],
            "culpa_vergonha": [2, 10], "desconexao_valores": [14, 26], "descontrole": [5, 17],
            "duracao": [9, 19], "entorpecimento": [11, 20], "incompreensibilidade": [3, 7],
            "invalidacao": [6, 12], "nao_aceitacao": [24, 18], "racionalidade": [13, 27],
            "visao_simplista": [23, 28], "ruminacao": [22, 16]}
EQR_MAP = {"reavaliacao": [1, 3, 5, 7, 8, 10], "supressao": [2, 4, 6, 9]}
DASS_MAP = {"depressao": [3, 5, 10, 13, 16, 17, 21], "ansiedade": [2, 4, 7, 9, 15, 19, 20],
            "estresse": [1, 6, 8, 11, 12, 14, 18]}

LABEL = {"supressao": "Supressão (EQR)", "reavaliacao": "Reavaliação (EQR)",
         "acusacao": "Acusação (LESS-II)", "baixa_expressao": "Baixa Expressão (LESS-II)",
         "baixo_consenso": "Baixo Consenso (LESS-II)", "culpa_vergonha": "Culpa/Vergonha (LESS-II)",
         "desconexao_valores": "Desconexão de Valores (LESS-II)", "descontrole": "Descontrole (LESS-II)",
         "duracao": "Duração (LESS-II)", "entorpecimento": "Entorpecimento (LESS-II)",
         "incompreensibilidade": "Incompreensibilidade (LESS-II)", "invalidacao": "Invalidação (LESS-II)",
         "nao_aceitacao": "Não Aceitação dos Sentimentos (LESS-II)", "racionalidade": "Racionalidade (LESS-II)",
         "visao_simplista": "Visão Simplista (LESS-II)", "ruminacao": "Ruminação (LESS-II)",
         "NonAcceptance": "Não Aceitação (DERS)", "Goals": "Metas (DERS)", "Impulse": "Impulsos (DERS)",
         "Awareness": "Consciência (DERS)", "Strategies": "Estratégias (DERS)", "Clarity": "Clareza (DERS)"}
SHORT = {"supressao": "Supressão", "reavaliacao": "Reavaliação", "acusacao": "Acusação",
         "baixa_expressao": "Baixa expr.", "baixo_consenso": "Baixo consenso", "culpa_vergonha": "Culpa/Vergonha",
         "desconexao_valores": "Desconex. valores", "descontrole": "Descontrole", "duracao": "Duração",
         "entorpecimento": "Entorpecim.", "incompreensibilidade": "Incompreens.", "invalidacao": "Invalidação",
         "nao_aceitacao": "Não aceit. sent.", "racionalidade": "Racionalidade", "visao_simplista": "Visão simplista",
         "ruminacao": "Ruminação", "NonAcceptance": "Não aceit. (DERS)", "Goals": "Metas (DERS)",
         "Impulse": "Impulsos (DERS)", "Awareness": "Consciência (DERS)", "Strategies": "Estratégias (DERS)",
         "Clarity": "Clareza (DERS)"}
DIMS22 = ["supressao", "reavaliacao", "acusacao", "baixa_expressao", "baixo_consenso", "culpa_vergonha",
          "desconexao_valores", "descontrole", "duracao", "entorpecimento", "incompreensibilidade",
          "invalidacao", "nao_aceitacao", "racionalidade", "visao_simplista", "ruminacao",
          "NonAcceptance", "Goals", "Impulse", "Awareness", "Strategies", "Clarity"]
DERS6 = list(DERS_MAP)
DASS3 = ["depressao", "ansiedade", "estresse"]
DASS_LAB = {"depressao": "Depressão", "ansiedade": "Ansiedade", "estresse": "Estresse"}

def keyed(items):
    """Recodifica itens reversos: escore alto = mais desregulação/sofrimento (DERS 6-x; LESS 7-x)."""
    k = items.copy()
    for i in DERS_REV:
        if f"ders_{i}" in k.columns:
            k[f"ders_{i}"] = 6 - k[f"ders_{i}"]
    for i in LESS_REV:
        if f"less_{i}" in k.columns:
            k[f"less_{i}"] = 7 - k[f"less_{i}"]
    return k

def score22(k):
    o = pd.DataFrame(index=k.index)
    for n, it in EQR_MAP.items():
        o[n] = k[[f"eqr_{i}" for i in it]].mean(axis=1)
    for n, it in LESS_MAP.items():
        o[n] = k[[f"less_{i}" for i in it]].mean(axis=1)
    for n, it in DERS_MAP.items():
        o[n] = k[[f"ders_{i}" for i in it]].sum(axis=1)
    return o[DIMS22]

def scoredass(k):
    return pd.DataFrame({n: k[[f"dass_{i}" for i in it]].sum(axis=1) * 2 for n, it in DASS_MAP.items()})

def relabel(lab, Zdf, k):
    """Rótulos ordenados pela média de z das 6 dimensões da DERS (0 = menor desregulação)."""
    m = pd.Series(Zdf[DERS6].mean(axis=1).values).groupby(lab).mean().sort_values()
    mp = {old: new for new, old in enumerate(m.index)}
    return np.array([mp[l] for l in lab])

def kmeans(Z, k, seed=SEED):
    return KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=seed).fit(Z)

# ------------------------------------------------------------------ 1. FLUXO DO BANCO
ref = read(os.path.join(DATA, "BANCO_DE_DADOS_REFORMADO.csv"))
usp = read(os.path.join(DATA, "USP.CSV"))
for d in (ref, usp):
    d["id"] = num(d["id_registro"])
n_linhas = len(ref)
ref = ref[ref["id"].notna()].copy()
usp = usp[usp["id"].notna()].copy()
gen_ref = num(ref["genero"])
invalid = ref[~gen_ref.isin([0, 1])]                       # ID 424: e-mail no campo gênero / colunas desalinhadas
base = ref[gen_ref.isin([0, 1])].copy().set_index("id")
items_base = pd.DataFrame({c: num(base[c]) for c in ALL95})
complete = items_base.notna().all(axis=1)
excl_missing = items_base.index[~complete].tolist()
ids238 = items_base.index[complete]
X_items = items_base.loc[ids238].copy()
# faixa de resposta
for c in ALL95:
    lo, hi = RANGE[c.split("_")[0]]
    assert X_items[c].between(lo, hi).all(), f"valor fora da faixa em {c}"
usp = usp.set_index("id")
usp_items = pd.DataFrame({c: num(usp[c]) for c in ALL95})
only_usp = sorted(set(usp.index) - set(ref["id"]))
only_ref = sorted(set(ref["id"]) - set(usp.index))
common = sorted(set(usp.index) & set(ref["id"]))
diff_common = int((usp_items.loc[common].fillna(-9).values != pd.DataFrame({c: num(ref.set_index("id").loc[common, c]) for c in ALL95}).fillna(-9).values).sum())
R["fluxo"] = {
    "linhas_arquivo_exportado": n_linhas, "ids_REFORMADO_com_resposta": int(len(ref)), "ids_USP_com_resposta": int(len(usp)),
    "uniao_ids": int(len(set(usp.index) | set(ref["id"]))), "ids_comuns": len(common),
    "diferencas_de_valores_nos_ids_comuns": diff_common,
    "somente_USP(desistentes)": only_usp, "somente_REFORMADO": only_ref,
    "excluido_erro_exportacao": invalid["id"].astype(int).tolist(), "base_valida": int(len(base)),
    "excluidos_itens_ausentes": [int(i) for i in excl_missing], "amostra_analitica": int(len(ids238)),
    "itens_respondidos_desistentes": {int(i): int(usp_items.loc[i].notna().sum()) for i in only_usp},
}

# ------------------------------------------------------------------ 2. ESCORAGEM E z-SCORE
K95 = keyed(X_items)
D22 = score22(K95)
DASS = scoredass(K95)
Z = pd.DataFrame(StandardScaler().fit_transform(D22), columns=DIMS22, index=D22.index)
n = len(Z)
R["n"] = n

# Alfa de Cronbach (itens já recodificados)
def cronbach(df):
    k = df.shape[1]
    return k / (k - 1) * (1 - df.var(ddof=1).sum() / df.sum(axis=1).var(ddof=1))
alpha_rows = []
for nme, it in DASS_MAP.items():
    alpha_rows.append(("DASS-21", DASS_LAB[nme], len(it), cronbach(K95[[f"dass_{i}" for i in it]])))
for nme, it in EQR_MAP.items():
    alpha_rows.append(("EQR", LABEL[nme].replace(" (EQR)", ""), len(it), cronbach(K95[[f"eqr_{i}" for i in it]])))
for nme, it in DERS_MAP.items():
    alpha_rows.append(("DERS", nme, len(it), cronbach(K95[[f"ders_{i}" for i in it]])))
for nme, it in LESS_MAP.items():
    alpha_rows.append(("LESS-II", LABEL[nme].replace(" (LESS-II)", ""), len(it), cronbach(K95[[f"less_{i}" for i in it]])))
ALFA = pd.DataFrame(alpha_rows, columns=["Instrumento", "Subescala", "Nº de itens", "alfa"])
ALFA_ORD = ALFA.copy()

# Sociodemografia (dados disponíveis nos CSV)
age = num(base.loc[ids238, "idade"]); gcod = num(base.loc[ids238, "genero"])
R["sociodemografia"] = {"idade_media": age.mean(), "idade_dp": age.std(ddof=1), "idade_mediana": age.median(),
                        "idade_min": age.min(), "idade_max": age.max(), "amplitude": age.max() - age.min(),
                        "genero_codigo1": int((gcod == 1).sum()), "genero_codigo0": int((gcod == 0).sum()),
                        "n_idade_ge40": int((age >= 40).sum())}

# ------------------------------------------------------------------ 3. K-MEANS (K=1..10), K=2 e K=3
inertia = {k: (KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=SEED).fit(Z.values).inertia_ if k > 1
               else float(((Z.values - Z.values.mean(0)) ** 2).sum())) for k in range(1, 11)}
sil = {k: silhouette_score(Z.values, kmeans(Z.values, k).labels_) for k in range(2, 11)}
R["inercia"] = inertia; R["silhueta"] = sil
km2, km3 = kmeans(Z.values, 2), kmeans(Z.values, 3)
L2, L3 = relabel(km2.labels_, Z, 2), relabel(km3.labels_, Z, 3)
N2 = ["Adaptativo", "Alto Risco"]; N3 = ["Resiliente", "Moderado", "Severo"]
R["k2"] = {"n": np.bincount(L2).tolist(), "silhueta": sil[2]}
R["k3"] = {"n": np.bincount(L3).tolist(), "silhueta": sil[3]}
R["k2_vs_k3_ARI"] = adjusted_rand_score(L2, L3)

# ------------------------------------------------------------------ 4. ESTABILIDADE: BOOTSTRAP + GMM
def boot(Zv, lab, k):
    a = []
    for i in range(N_BOOT):
        idx = resample(np.arange(len(Zv)), random_state=i)
        bl = KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=SEED).fit_predict(Zv[idx])
        a.append(adjusted_rand_score(lab[idx], bl))
    return np.array(a)
B2, B3 = boot(Z.values, L2, 2), boot(Z.values, L3, 3)
def gmm(Zv, lab, k, cov="spherical"):
    """GMM com covariância esférica = equivalente probabilístico do K-Means (escolha teórica, pré-declarada);
    as demais estruturas de covariância entram como análise de sensibilidade (Apêndice E)."""
    g = GaussianMixture(n_components=k, covariance_type=cov, n_init=N_INIT, random_state=SEED).fit(Zv)
    gl = g.predict(Zv)
    ct = pd.crosstab(lab, gl).reindex(columns=range(k), fill_value=0).values
    r_, c_ = linear_sum_assignment(-ct)                      # alinha colunas do GMM às linhas do K-Means
    return adjusted_rand_score(lab, gl), ct[:, list(c_)]
G2, CT2 = gmm(Z.values, L2, 2); G3, CT3 = gmm(Z.values, L3, 3)
GMM_SENS = {cov: {"K2": gmm(Z.values, L2, 2, cov)[0], "K3": gmm(Z.values, L3, 3, cov)[0]} for cov in ("spherical", "diag", "full", "tied")}
BIC = {cov: [GaussianMixture(n_components=k, covariance_type=cov, n_init=N_INIT, random_state=SEED).fit(Z.values).bic(Z.values) for k in range(1, 7)] for cov in ("spherical", "full")}
R["bootstrap"] = {"K2": {"media": B2.mean(), "dp": B2.std(), "min": B2.min(), "max": B2.max(), "prop_gt_085": float((B2 > .85).mean()), "mediana": float(np.median(B2))},
                  "K3": {"media": B3.mean(), "dp": B3.std(), "min": B3.min(), "max": B3.max(), "prop_gt_085": float((B3 > .85).mean()), "mediana": float(np.median(B3))}}
R["gmm"] = {"covariancia_principal": "esférica", "K2_ARI": G2, "K3_ARI": G3, "K2_matriz": CT2.tolist(), "K3_matriz": CT3.tolist(), "sensibilidade_covariancia": GMM_SENS, "BIC_K1a6": BIC}

# ------------------------------------------------------------------ 5. MICE (sensibilidade): todos que iniciaram
sub = usp_items.loc[only_usp]
sub = sub[sub[ER74].notna().sum(axis=1) > 0]                 # remove ID 22 (0 itens)
allstart = pd.concat([items_base[ER74], sub[ER74]])
allstart = allstart[~allstart.index.duplicated()]
R["mice_base"] = {"n_base": int(len(allstart)), "n_itens_ausentes_pct": float(allstart.isna().mean().mean() * 100)}
def run_mice(m):
    imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=10, min_value=LO, max_value=HI,
                           random_state=m, initial_strategy="mean")
    arr = imp.fit_transform(allstart.values)
    it = pd.DataFrame(arr, columns=ER74, index=allstart.index)
    k = keyed(it.assign(**{c: 0 for c in DASS_ITEMS}))       # DASS não entra
    d = score22(k)
    Zm = pd.DataFrame(StandardScaler().fit_transform(d), columns=DIMS22, index=d.index)
    out = {}
    for kk, ref_lab in ((2, L2), (3, L3)):
        lab = relabel(kmeans(Zm.values, kk).labels_, Zm, kk)
        s = pd.Series(lab, index=Zm.index).loc[Z.index].values
        out[kk] = (adjusted_rand_score(ref_lab, s), s)
    return out
mice = [run_mice(m) for m in range(N_IMP)]
M2 = np.array([m[2][0] for m in mice]); M3 = np.array([m[3][0] for m in mice])
CTM = pd.crosstab(L2, mice[-1][2][1]).values
R["mice"] = {"K2": {"media": M2.mean(), "dp": M2.std(ddof=1), "min": M2.min(), "max": M2.max(), "por_imputacao": M2.tolist()},
             "K3": {"media": M3.mean(), "dp": M3.std(ddof=1), "min": M3.min(), "max": M3.max(), "por_imputacao": M3.tolist()},
             "n_comum": int(len(Z)), "matriz_K2_ultima": CTM.tolist(), "n_base": int(len(allstart))}

# ------------------------------------------------------------------ 6. PCA
pca = PCA().fit(Z.values)
ev = pca.explained_variance_ratio_
load = pd.DataFrame(pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2]), index=DIMS22, columns=["PC1", "PC2"])
PC = pca.transform(Z.values)[:, :2]
R["pca"] = {"pc1": ev[0] * 100, "pc2": ev[1] * 100, "soma": (ev[0] + ev[1]) * 100,
            "top_pc1": load["PC1"].reindex(load["PC1"].abs().sort_values(ascending=False).index).head(6).round(2).to_dict(),
            "top_pc2": load["PC2"].reindex(load["PC2"].abs().sort_values(ascending=False).index).head(6).round(2).to_dict()}

# ------------------------------------------------------------------ 7. TESTES (K=2 e K=3)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
shap_p = {v: stats.shapiro(D22[v])[1] for v in DIMS22}; shap_p.update({v: stats.shapiro(DASS[v])[1] for v in DASS3})
R["shapiro"] = {"n_dim_p<0.05_de_22": int(sum(shap_p[v] < .05 for v in DIMS22)),
                "dass_p<0.05": {v: bool(shap_p[v] < .05) for v in DASS3},
                "dims_nao_rejeitadas": [v for v in DIMS22 if shap_p[v] >= .05]}
def mw(x1, x0):
    u1, p = stats.mannwhitneyu(x1, x0, alternative="two-sided")
    return u1, p, 2 * u1 / (len(x1) * len(x0)) - 1
rows = []
g0, g1 = L2 == 0, L2 == 1
for v in DIMS22 + DASS3:
    src = D22 if v in DIMS22 else DASS
    a, b = src.loc[g1, v], src.loc[g0, v]
    u, p, r = mw(a, b)
    rows.append({"var": v, "rotulo": LABEL.get(v, DASS_LAB.get(v)), "M0": b.mean(), "DP0": b.std(), "M1": a.mean(), "DP1": a.std(),
                 "U": u, "p": p, "r_rb": r})
T2 = pd.DataFrame(rows)
T2.loc[T2["var"].isin(DIMS22), "p_holm"] = holm(T2.loc[T2["var"].isin(DIMS22), "p"].values)
R["k2_tabela"] = T2.round(5).to_dict("records")
rows = []
for v in DIMS22 + DASS3:
    src = D22 if v in DIMS22 else DASS
    grp = [src.loc[L3 == c, v] for c in range(3)]
    H, p = stats.kruskal(*grp)
    pairs = {}
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        u, pp = stats.mannwhitneyu(grp[j], grp[i], alternative="two-sided")
        pairs[(i, j)] = (pp, 2 * u / (len(grp[i]) * len(grp[j])) - 1)
    ns = [f"{N3[j]} ≈ {N3[i]}" for (i, j), (pp, rr) in pairs.items() if pp >= 0.05 / 3]
    rows.append({"var": v, "rotulo": LABEL.get(v, DASS_LAB.get(v)), **{f"M{c}": grp[c].mean() for c in range(3)},
                 **{f"DP{c}": grp[c].std() for c in range(3)}, "H": H, "p": p, "eps2": H / (n - 1),
                 "pares_ns": "; ".join(ns) if ns else "Nenhum",
                 **{f"p_{i}{j}": pairs[(i, j)][0] for (i, j) in pairs}, **{f"r_{i}{j}": pairs[(i, j)][1] for (i, j) in pairs}})
T3 = pd.DataFrame(rows)
R["k3_tabela"] = T3.round(5).to_dict("records")

# ------------------------------------------------------------------ 8. ATRITO (desistentes x amostra analítica)
att = usp_items.loc[only_usp].copy()
att["idade"] = num(usp.loc[only_usp, "idade"]); att["genero"] = num(usp.loc[only_usp, "genero"])
ders_ok = att[DERS_ITEMS].notna().all(axis=1)
kd = keyed(att[DERS_ITEMS + EQR_ITEMS].copy())
att["ders_total"] = kd[DERS_ITEMS].sum(axis=1).where(ders_ok)
att["reav"] = kd[[f"eqr_{i}" for i in EQR_MAP["reavaliacao"]]].mean(axis=1)
att["supr"] = kd[[f"eqr_{i}" for i in EQR_MAP["supressao"]]].mean(axis=1)
ders_tot_238 = K95[DERS_ITEMS].sum(axis=1)
res_att = {"n_desistentes": len(only_usp), "n_com_DERS_completa": int(ders_ok.sum())}
def cmp(x, y):
    x, y = x.dropna(), y.dropna()
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {"n_desist": int(len(x)), "M_desist": x.mean(), "M_analitica": y.mean(), "p": p}
res_att["idade"] = cmp(att["idade"], age)
res_att["ders_total"] = cmp(att["ders_total"], ders_tot_238)
res_att["reavaliacao"] = cmp(att["reav"], D22["reavaliacao"]); res_att["supressao"] = cmp(att["supr"], D22["supressao"])
res_att["genero_prop_cod1"] = {"desist": float((att["genero"] == 1).sum() / att["genero"].notna().sum()), "analitica": float((gcod == 1).mean())}
# 3 excluídos por itens ausentes: posição nos percentis da amostra analítica
exc = items_base.loc[excl_missing]
ke = keyed(exc[DERS_ITEMS + EQR_ITEMS])
res_att["excluidos_itens_ausentes"] = {int(i): {"itens_respondidos": int(exc.loc[i].notna().sum()),
    "idade": float(num(base.loc[[i], "idade"]).iloc[0]),
    "ders_total": float(ke.loc[i, DERS_ITEMS].sum()) if ke.loc[i, DERS_ITEMS].notna().all() else None,
    "percentil_ders": float(stats.percentileofscore(ders_tot_238, ke.loc[i, DERS_ITEMS].sum())) if ke.loc[i, DERS_ITEMS].notna().all() else None} for i in excl_missing}
R["atrito"] = res_att

# ------------------------------------------------------------------ 9. RANDOM FOREST + SHAP (substituto interpretável; DASS fora)
import shap
Xrf = X_items[ER74].copy()
rf = RandomForestClassifier(n_estimators=500, random_state=SEED, class_weight="balanced", n_jobs=-1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
proba = cross_val_predict(rf, Xrf, L2, cv=skf, method="predict_proba")[:, 1]
pred = (proba >= .5).astype(int)
accs = []
for tr, te in skf.split(Xrf, L2):
    accs.append((RandomForestClassifier(n_estimators=500, random_state=SEED, class_weight="balanced", n_jobs=-1).fit(Xrf.iloc[tr], L2[tr]).predict(Xrf.iloc[te]) == L2[te]).mean())
R["rf"] = {"acc_cv_media": float(np.mean(accs)), "acc_cv_dp": float(np.std(accs)), "acc_folds": [float(a) for a in accs],
           "auc_cv": float(roc_auc_score(L2, proba)), "acc_global_cv": float((pred == L2).mean())}
rf.fit(Xrf, L2)
gini = pd.Series(rf.feature_importances_, index=ER74).sort_values(ascending=False)
sv = shap.TreeExplainer(rf).shap_values(Xrf)
sv1 = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
msh = pd.Series(np.abs(sv1).mean(0), index=ER74).sort_values(ascending=False)
corr = {c: float(np.corrcoef(Xrf[c], sv1[:, ER74.index(c)])[0, 1]) for c in msh.index[:20]}
R["rf"].update({"gini_top10": {k: round(float(v) * 100, 2) for k, v in gini.head(10).items()},
                "shap_top20": {k: round(float(v), 4) for k, v in msh.head(20).items()},
                "shap_corr_top20": {k: round(v, 2) for k, v in corr.items()}})

# ------------------------------------------------------------------ 10. AUDITORIA DE EXTREMOS
sev = (DASS.sum(axis=1)).sort_values(ascending=False)
R["extremos"] = {"mais_grave_id": int(sev.index[0]), "n_dass_zero_zero_zero": int((DASS.sum(axis=1) == 0).sum()),
                 "ids_dass_zero": [int(i) for i in DASS.index[DASS.sum(axis=1) == 0]]}
R["extremos"]["dass_zero_clusters"] = {int(i): N2[L2[list(Z.index).index(i)]] for i in DASS.index[DASS.sum(axis=1) == 0]}
lessd = ["culpa_vergonha", "descontrole", "incompreensibilidade", "invalidacao"]
R["extremos"]["ids_LESS4_piso1"] = [int(i) for i in D22.index[(D22[lessd] == 1).all(axis=1)]]
R["extremos"]["ids_LESS4_teto6"] = [int(i) for i in D22.index[(D22[lessd] == 6).all(axis=1)]]
R["extremos"]["ranking_reavaliacao_menores"] = {int(i): float(v) for i, v in D22["reavaliacao"].sort_values().head(4).items()}
for pid in (354, 263):
    if pid in Z.index:
        j = list(Z.index).index(pid)
        R["extremos"][str(pid)] = {"cluster_K2": N2[L2[j]], "cluster_K3": N3[L3[j]],
            "dass": DASS.loc[pid].to_dict(), "reavaliacao": float(D22.loc[pid, "reavaliacao"]),
            "menor_reavaliacao_da_amostra": bool(D22.loc[pid, "reavaliacao"] == D22["reavaliacao"].min()),
            "LESS": {v: float(D22.loc[pid, v]) for v in ["culpa_vergonha", "descontrole", "incompreensibilidade", "invalidacao"]}}
R["extremos"]["reavaliacao_media"] = float(D22["reavaliacao"].mean())

# ------------------------------------------------------------------ 11. FAIXAS DASS-21 (Lovibond & Lovibond, 1995) das médias dos perfis
def band(v, cuts, names):
    for c, nme in zip(cuts, names):
        if v <= c:
            return nme
    return names[-1]
CUT = {"depressao": ([9, 13, 20, 27], ["normal", "leve", "moderada", "grave", "extremamente grave"]),
       "ansiedade": ([7, 9, 14, 19], ["normal", "leve", "moderada", "grave", "extremamente grave"]),
       "estresse": ([14, 18, 25, 33], ["normal", "leve", "moderado", "grave", "extremamente grave"])}
R["faixas_dass"] = {N2[c]: {v: band(DASS.loc[L2 == c, v].mean(), *CUT[v]) for v in DASS3} for c in (0, 1)}

# ------------------------------------------------------------------ 12. TABELAS PARA O TEXTO (formatadas) + EXCEL
def t5_format():
    rows = []
    for c in (0, 1):
        r = {"Perfil": f"{N2[c]}", "n": int((L2 == c).sum())}
        for v in DASS3:
            r[DASS_LAB[v]] = f"{br(DASS.loc[L2 == c, v].mean())} ({br(DASS.loc[L2 == c, v].std())})"
        rows.append(r)
    for lab, key in (("U", "U"), ("Valor-p", "p"), ("r_rb", "r_rb")):
        r = {"Perfil": lab, "n": ""}
        for v in DASS3:
            t = T2[T2["var"] == v].iloc[0]
            r[DASS_LAB[v]] = br(t["U"], 1, True) if key == "U" else pfmt(t["p"]) if key == "p" else br(t["r_rb"])
        rows.append(r)
    return pd.DataFrame(rows)
T5 = t5_format()
def t6_format():
    o = []
    for v in DIMS22:
        t = T2[T2["var"] == v].iloc[0]
        d = 2 if v not in DERS6 else 2
        o.append({"Variável psicométrica": LABEL[v], f"Adaptativo (n = {(L2==0).sum()}) M (DP)": f"{br(t.M0)} ({br(t.DP0)})",
                  f"Alto Risco (n = {(L2==1).sum()}) M (DP)": f"{br(t.M1)} ({br(t.DP1)})", "U": br(t.U, 1, True),
                  "Valor-p": pfmt(t.p), "p ajustado (Holm)": pfmt(t.p_holm), "r_rb": br(t.r_rb), "Efeito": eff_label(t.r_rb)})
    return pd.DataFrame(o)
T6 = t6_format()
def t7_format():
    rows = []
    for c in (2, 1, 0):
        r = {"Perfil": N3[c], "n": int((L3 == c).sum())}
        for v in DASS3:
            r[DASS_LAB[v]] = f"{br(DASS.loc[L3 == c, v].mean())} ({br(DASS.loc[L3 == c, v].std())})"
        rows.append(r)
    r = {"Perfil": "Kruskal-Wallis H (p)", "n": ""}
    for v in DASS3:
        t = T3[T3["var"] == v].iloc[0]
        r[DASS_LAB[v]] = f"{br(t.H)} ({pfmt(t.p)})"
    rows.append(r)
    r = {"Perfil": "Pares n.s. (Bonferroni)", "n": ""}
    for v in DASS3:
        r[DASS_LAB[v]] = T3[T3["var"] == v].iloc[0]["pares_ns"]
    rows.append(r)
    return pd.DataFrame(rows)
T7 = t7_format()
def t8_format():
    o = []
    for v in DIMS22:
        t = T3[T3["var"] == v].iloc[0]
        o.append({"Variável": LABEL[v], f"Severo (n={(L3==2).sum()})": f"{br(t.M2)} ({br(t.DP2)})",
                  f"Moderado (n={(L3==1).sum()})": f"{br(t.M1)} ({br(t.DP1)})", f"Resiliente (n={(L3==0).sum()})": f"{br(t.M0)} ({br(t.DP0)})",
                  "H": br(t.H), "p (KW)": pfmt(t.p), "Pares n.s. (Bonferroni)": t.pares_ns})
    return pd.DataFrame(o)
T8 = t8_format()
def tab2_format():
    o = ALFA.copy()
    o["α"] = o["alfa"].map(lambda x: br(x, 3))
    return o[["Instrumento", "Subescala", "Nº de itens", "α"]]
T2A = tab2_format()
mice_tab = pd.DataFrame({"Métrica": ["ARI médio (5 imputações)", "ARI (desvio-padrão)", "ARI (mínimo – máximo)"],
    "K = 2": [br(M2.mean(), 3), br(M2.std(ddof=1), 3), f"{br(M2.min(),3)} – {br(M2.max(),3)}"],
    "K = 3": [br(M3.mean(), 3), br(M3.std(ddof=1), 3), f"{br(M3.min(),3)} – {br(M3.max(),3)}"]})
def ari_lab(a): return "Excelente" if a >= .9 else "Boa" if a >= .8 else "Moderada" if a >= .65 else "Fraca"
est_tab = pd.DataFrame({"Métrica": ["Bootstrap ARI (média)", "Bootstrap ARI (desvio-padrão)", "Bootstrap ARI (mínimo – máximo)", "Interpretação (bootstrap)", "GMM vs. K-Means (ARI)", "Interpretação (GMM)"],
    "K = 2": [br(B2.mean(), 3), br(B2.std(), 3), f"{br(B2.min(),3)} – {br(B2.max(),3)}", ari_lab(B2.mean()), br(G2, 3), ari_lab(G2)],
    "K = 3": [br(B3.mean(), 3), br(B3.std(), 3), f"{br(B3.min(),3)} – {br(B3.max(),3)}", ari_lab(B3.mean()), br(G3, 3), ari_lab(G3)]})
mice_tab_interp = ari_lab(M2.mean()), ari_lab(M3.mean())
R["interpretacao_ari"] = {"mice_K2": mice_tab_interp[0], "mice_K3": mice_tab_interp[1], "boot_K2": ari_lab(B2.mean()), "boot_K3": ari_lab(B3.mean()), "gmm_K2": ari_lab(G2), "gmm_K3": ari_lab(G3)}
flux = pd.DataFrame({"Etapa": ["Registros (linhas) no arquivo exportado", "Identificadores com alguma resposta (união das duas exportações)", "Desistentes iniciais (só DERS/EQR; ausentes do banco curado)", "Registro com erro de exportação (ID 424)", "Base válida (banco curado sem o registro corrompido)", "Excluídos por itens ausentes (IDs " + ", ".join(map(str, R['fluxo']['excluidos_itens_ausentes'])) + ")", "Amostra analítica final"],
    "n": [n_linhas, R["fluxo"]["uniao_ids"], len(only_usp), len(invalid), len(base), len(excl_missing), len(ids238)]})
appA = pd.DataFrame([{"Instrumento": "DERS", "Dimensão": nme, "Itens": ", ".join(map(str, it)), "Itens reversos": ", ".join(str(i) for i in it if i in DERS_REV) or "—", "Cálculo": "soma"} for nme, it in DERS_MAP.items()]
    + [{"Instrumento": "EQR", "Dimensão": LABEL[nme], "Itens": ", ".join(map(str, it)), "Itens reversos": "—", "Cálculo": "média"} for nme, it in EQR_MAP.items()]
    + [{"Instrumento": "LESS-II", "Dimensão": LABEL[nme], "Itens": ", ".join(map(str, it)), "Itens reversos": ", ".join(str(i) for i in it if i in LESS_REV) or "—", "Cálculo": "média"} for nme, it in LESS_MAP.items()]
    + [{"Instrumento": "DASS-21", "Dimensão": DASS_LAB[nme], "Itens": ", ".join(map(str, it)), "Itens reversos": "—", "Cálculo": "soma × 2"} for nme, it in DASS_MAP.items()])
params = pd.DataFrame({"Parâmetro": ["Semente global (random_state)", "K-Means", "n_init", "Bootstrap (reamostragens)", "GMM", "MICE", "Random Forest", "Validação cruzada", "Padronização", "Escala das reversões", "Versão scikit-learn"],
    "Valor": [SEED, "init='k-means++'", N_INIT, N_BOOT, "covariance_type='spherical' (principal); diag/full/tied em sensibilidade; n_init=10", f"IterativeImputer + BayesianRidge, sample_posterior=True, {N_IMP} imputações, itens da DASS fora", "500 árvores, class_weight='balanced'", "StratifiedKFold(5, shuffle=True)", "z-score (StandardScaler, ddof=0) sobre as 22 dimensões", "DERS: 6−x; LESS-II: 7−x", __import__('sklearn').__version__]})
with pd.ExcelWriter(os.path.join(OUT, "RESULTADOS_FINAIS.xlsx")) as xw:
    flux.to_excel(xw, sheet_name="Fluxo_amostral", index=False)
    T2A.to_excel(xw, sheet_name="Tab2_Alfa", index=False)
    mice_tab.to_excel(xw, sheet_name="Tab3_MICE", index=False)
    est_tab.to_excel(xw, sheet_name="Tab4_Estabilidade", index=False)
    T5.to_excel(xw, sheet_name="Tab5_DASS_K2", index=False)
    T6.to_excel(xw, sheet_name="Tab6_K2", index=False)
    T7.to_excel(xw, sheet_name="Tab7_DASS_K3", index=False)
    T8.to_excel(xw, sheet_name="Tab8_K3", index=False)
    appA.to_excel(xw, sheet_name="ApendiceA_Chaves", index=False)
    T2.to_excel(xw, sheet_name="ApendiceB_K2_completo", index=False)
    T3.to_excel(xw, sheet_name="ApendiceC_K3_completo", index=False)
    pd.DataFrame({"item": msh.index[:30], "SHAP_medio_abs": msh.values[:30], "Gini_%": [gini[i] * 100 for i in msh.index[:30]]}).to_excel(xw, sheet_name="ApendiceD_RF_SHAP", index=False)
    pd.DataFrame({"id": Z.index, "cluster_K2": [N2[i] for i in L2], "cluster_K3": [N3[i] for i in L3]}).join(D22.reset_index(drop=True)).join(DASS.reset_index(drop=True)).to_excel(xw, sheet_name="Base_final_238", index=False)
    pd.DataFrame([{"Covariância do GMM": c, "ARI K=2": v["K2"], "ARI K=3": v["K3"]} for c, v in GMM_SENS.items()]).to_excel(xw, sheet_name="ApendiceE_GMM", index=False)
    pd.DataFrame({"K": range(1, 7), **{f"BIC ({c})": v for c, v in BIC.items()}}).to_excel(xw, sheet_name="ApendiceE_BIC", index=False)
    params.to_excel(xw, sheet_name="Parametros", index=False)
for nm_, df_ in (("Tab2_alfa", T2A), ("Tab3_mice", mice_tab), ("Tab4_estabilidade", est_tab), ("Tab5_dass_k2", T5), ("Tab6_k2", T6), ("Tab7_dass_k3", T7), ("Tab8_k3", T8)):
    df_.to_csv(os.path.join(TAB, nm_ + ".csv"), index=False, sep=";", encoding="utf-8-sig")
pd.DataFrame({"id": Z.index, "cluster_K2": L2, "cluster_K3": L3}).to_csv(os.path.join(TAB, "clusters_238.csv"), index=False, sep=";")

# ------------------------------------------------------------------ 13. FIGURAS
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})
C2 = {0: "#1f77b4", 1: "#d62728"}; C3 = {0: "#21918c", 1: "#e6b800", 2: "#440154"}
def save(fig, nme):
    fig.savefig(os.path.join(FIG, nme), dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
# Fig 1: MICE
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for a, (arr, k, col) in zip(ax, ((M2, 2, "#4c72b0"), (M3, 3, "#dd8452"))):
    a.bar(range(1, N_IMP + 1), arr, color=col); a.axhline(arr.mean(), color="k", ls="--", lw=1, label=f"Média = {br(arr.mean(), 3)}")
    a.axhline(0.80, color="gray", ls=":", lw=1, label="Referência (0,80)"); a.set_ylim(0, 1.02)
    a.set_title(f"Concordância listwise × MICE (K={k})"); a.set_xlabel("Imputação MICE (m)"); a.set_ylabel("ARI vs. solução listwise"); a.legend(loc="lower left", fontsize=7)
save(fig, "FIGURA_1.png")
# Fig 2: contingência MICE
fig, a = plt.subplots(figsize=(4.6, 4.2))
a.imshow(CTM, cmap="Greens"); 
for i in range(2):
    for j in range(2):
        a.text(j, i, int(CTM[i, j]), ha="center", va="center", color="white" if CTM[i, j] > CTM.max() / 2 else "black", fontsize=13)
a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels([f"MICE: {N2[0]}", f"MICE: {N2[1]}"]); a.set_yticklabels([f"Listwise: {N2[0]}", f"Listwise: {N2[1]}"])
a.set_title(f"Listwise × MICE (K=2, imputação final)\nARI = {br(mice[-1][2][0], 3)} | n comum = {len(Z)}")
save(fig, "FIGURA_2.png")
# Fig 3: cotovelo e silhueta
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].plot(range(1, 11), [inertia[k] for k in range(1, 11)], "o-", color="blue"); ax[0].set_title("Método do Cotovelo (Elbow Method)")
ax[0].set_xlabel("Número de clusters (K)"); ax[0].set_ylabel("Soma dos quadrados intra-cluster (inércia)"); ax[0].set_xticks(range(1, 11)); ax[0].grid(alpha=.3, ls="--")
ax[1].plot(range(2, 11), [sil[k] for k in range(2, 11)], "s-", color="red"); ax[1].set_title("Análise de Silhueta (Silhouette Score)")
ax[1].set_xlabel("Número de clusters (K)"); ax[1].set_ylabel("Coeficiente de silhueta (maior = melhor)"); ax[1].set_xticks(range(2, 11)); ax[1].grid(alpha=.3, ls="--")
save(fig, "FIGURA_3.png")
# Fig 4: bootstrap
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for a, (arr, k, col) in zip(ax, ((B2, 2, "#4c72b0"), (B3, 3, "#dd8452"))):
    a.hist(arr, bins=np.arange(0.3, 1.025, 0.025), color=col, edgecolor="white", alpha=.9)
    a.axvline(arr.mean(), color="k", ls="--", lw=1, label=f"Média = {br(arr.mean(), 3)}"); a.axvline(0.80, color="gray", ls=":", lw=1, label="Referência (0,80)")
    a.set_xlim(0.3, 1.0); a.set_title(f"Distribuição do ARI — bootstrap\n(K={k}, {N_BOOT} reamostragens)"); a.set_xlabel("Índice de Rand Ajustado (ARI)"); a.set_ylabel("Frequência"); a.legend(loc="upper left", fontsize=7)
save(fig, "FIGURA_4.png")
# Fig 5: GMM
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.9))
for a, (ct, k, nm_) in zip(ax, ((CT2, 2, N2), (CT3, 3, N3))):
    a.imshow(ct, cmap="Blues")
    for i in range(k):
        for j in range(k):
            a.text(j, i, int(ct[i, j]), ha="center", va="center", color="white" if ct[i, j] > ct.max() / 2 else "black", fontsize=11)
    a.set_xticks(range(k)); a.set_yticks(range(k)); a.set_xticklabels([f"GMM {j}" for j in range(k)]); a.set_yticklabels([f"K-Means: {nm_[i]}" for i in range(k)])
    a.set_title(f"K-Means × GMM (K={k}) | ARI = {br(G2 if k == 2 else G3, 3)}")
save(fig, "FIGURA_5.png")
# Fig 6: PCA
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.9))
for a, (lab, cmap, nm_, k) in zip(ax, ((L2, C2, N2, 2), (L3, C3, N3, 3))):
    for c in range(k):
        a.scatter(PC[lab == c, 0], PC[lab == c, 1], s=22, alpha=.75, color=cmap[c], label=f"{nm_[c]} (n={int((lab == c).sum())})")
    a.set_xlabel(f"Componente Principal 1 ({br(ev[0]*100,1)}%)"); a.set_ylabel(f"Componente Principal 2 ({br(ev[1]*100,1)}%)")
    a.set_title(f"Projeção PCA dos clusters (K={k})"); a.legend(fontsize=7); a.grid(alpha=.3, ls="--")
save(fig, "FIGURA_6.png")
# Fig 7: SHAP
plt.figure(figsize=(7.4, 6.0))
shap.summary_plot(sv1, Xrf, feature_names=[c.replace("ders_", "DERS_").replace("less_", "LESS_").replace("eqr_", "EQR_") for c in ER74], max_display=20, show=False)
fg = plt.gcf(); fg.axes[0].set_title("Impacto direcional dos itens nos agrupamentos (valores SHAP; classe Alto Risco)", fontsize=9)
fg.axes[0].set_xlabel("Valor SHAP (impacto na saída do modelo)")
if len(fg.axes) > 1:
    fg.axes[-1].set_ylabel("Valor do item")
    fg.axes[-1].set_yticklabels(["Baixo", "Alto"])
save(fg, "FIGURA_7.png")
# Fig 8: radar (22 dimensões, z-scores dos centróides)
cent = np.array([Z.values[L2 == c].mean(0) for c in (0, 1)])
ang = np.linspace(0, 2 * np.pi, len(DIMS22), endpoint=False); ang_c = np.concatenate([ang, ang[:1]])
fig = plt.figure(figsize=(8.2, 8.2)); a = plt.subplot(111, polar=True)
for c in (0, 1):
    v = np.concatenate([cent[c], cent[c][:1]]); a.plot(ang_c, v, color=C2[c], lw=1.8, label=f"{N2[c]} (n={int((L2 == c).sum())})"); a.fill(ang_c, v, color=C2[c], alpha=.18)
a.set_xticks(ang); a.set_xticklabels([SHORT[v] for v in DIMS22], fontsize=7.5); a.set_ylim(-1.2, 1.2); a.set_yticks([-1, -.5, 0, .5, 1]); a.set_yticklabels(["-1,0", "-0,5", "0,0", "0,5", "1,0"], fontsize=7)
a.set_rlabel_position(97)
a.set_title("Perfis de regulação emocional (centróides em z-score)", fontsize=10, pad=18); a.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
save(fig, "FIGURA_8.png")
R["centroides_z_K2"] = {v: {"Adaptativo": float(cent[0][i]), "Alto Risco": float(cent[1][i])} for i, v in enumerate(DIMS22)}
dz = pd.Series({v: cent[1][i] - cent[0][i] for i, v in enumerate(DIMS22)}).sort_values(ascending=False)
R["maiores_diferencas_z"] = {"top5": dz.head(5).round(2).to_dict(), "menores": dz.tail(3).round(2).to_dict()}

# ------------------------------------------------------------------ 14. SAÍDAS
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
R["alfa"] = {f"{r.Instrumento}|{r.Subescala}": round(float(r.alfa), 3) for r in ALFA.itertuples()}
json.dump(clean(R), open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=1)
print("OK — saída em", OUT)
