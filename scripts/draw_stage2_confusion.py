#!/usr/bin/env python3
"""Estagio 2 — MATRIZ DE CONFUSAO por CATEGORIA (taxonomia FINA, 6 classes) + metricas por grupo.

Le `artifacts/reports/evaluation_report.json` (held-out, gerado pelo run_experiment) e produz:

  • confusion_matrix_stage2_fina.png / .pdf      (PT)  — 2 paineis: contagens + recall (linha-norm.)
  • confusion_matrix_stage2_fina_en.png / .pdf   (EN)
  • metricas_stage2_por_grupo.png / .pdf          (PT)  — tabela: n, acertos, PRECISAO, recall, F1, ACURACIA
  • stage2_per_group_metrics_en.png / .pdf        (EN)
  • STAGE2_MATRIZ_CONFUSAO.md                            — relatorio pronto p/ apresentacao

As 6 categorias finas sao: black_bars · disordered_layout · distortion · empty_space ·
orientation · overlay. O Estagio 2 atribui categoria SO a imagens de ERRO, pelo PROTOTIPO de
categoria mais proximo no espaco SupCon (metodo canonico). Reportamos o modo ORACULO (todos os
89 erros do teste) e, em apendice, o CONDICIONAL ao gate E1 (so erros que o E1 sinalizou = producao).

DEFINICOES por classe c (one-vs-rest sobre a matriz de confusao):
  TP=diagonal · FN=linha-TP · FP=coluna-TP · TN=N-TP-FN-FP
  precisao(c) = TP/(TP+FP)                 -> dos que o modelo CHAMOU de c, quantos eram c
  recall(c)   = TP/(TP+FN)  [= taxa de acerto da classe / diagonal normalizada por linha]
  acuracia(c) = (TP+TN)/N   [one-vs-rest]  -> ATENCAO: inflada por TN em classes raras; ler junto do suporte
  F1(c)       = 2*prec*rec/(prec+rec)
  acuracia GLOBAL = soma(diagonal)/N       -> fracao de TODOS os erros com categoria correta

Todas as metricas sao RECALCULADAS da matriz crua e CONFERIDAS contra o JSON (assert) — se algo
divergir, o script falha. Nenhuma metrica e' inventada aqui; a fonte da verdade e' a CM do held-out.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

REP = Path(__file__).resolve().parent.parent / "artifacts" / "reports"

# rotulos amigaveis (multilinha p/ caber) — ordem definida pelo JSON (sorted by id)
PRETTY = {
    "black_bars": "black_bars",
    "disordered_layout": "disordered_layout",
    "distortion": "distortion",
    "empty_space": "empty_space",
    "orientation": "orientation",
    "overlay": "overlay",
}

I18N = {
    "pt": {
        "title_cm": "Estágio 2 — Matriz de confusão por categoria (FINA, 6 classes) — held-out",
        "sub_cm": "ORÁCULO: classifica todos os {n} erros do teste · acurácia global = {acc:.0%} · F1-macro = {f1:.2f} [IC95 {lo:.2f}–{hi:.2f}]",
        "panel_counts": "Contagens (linhas = verdadeiro, colunas = previsto)",
        "panel_norm": "Recall por classe (% da linha) — diagonal (borda) = acerto",
        "xlabel": "categoria prevista (protótipo)",
        "ylabel": "categoria verdadeira",
        "tbl_title": "Estágio 2 — métricas por grupo (taxonomia fina, held-out, oráculo n={n})",
        "tbl_note": ("Precisão = dos que o modelo chamou de X, quantos eram X (— = nunca previu X: 0/0 indefinida).  "
                     "Recall (taxa de acerto) = dos X reais, quantos foram acertados.  "
                     "Acurácia¹ = (acerto+rejeição correta)/total (one-vs-resto).\n"
                     "¹ inflada por verdadeiros-negativos em classes raras — leia SEMPRE com o suporte (n). "
                     "⚠ = suporte < 5 (métrica instável; leia a fração crua acertos/n). "
                     "Linha GLOBAL = micro (precisão = recall = F1 = acurácia = {acc:.0%}, identidade multiclasse); F1-macro = {f1:.2f} (no título)."),
        "cols": ["grupo", "n", "acertos", "precisão", "recall\n(acerto)", "F1", "acurácia¹\n(1-vs-resto)"],
        "global_row": "GLOBAL (micro)",
    },
    "en": {
        "title_cm": "Stage 2 — per-category confusion matrix (FINE, 6 classes) — held-out",
        "sub_cm": "ORACLE: classifies all {n} test errors · global accuracy = {acc:.0%} · macro-F1 = {f1:.2f} [CI95 {lo:.2f}–{hi:.2f}]",
        "panel_counts": "Counts (rows = actual, cols = predicted)",
        "panel_norm": "Per-class recall (% of row) — diagonal (outlined) = hit",
        "xlabel": "predicted category (prototype)",
        "ylabel": "actual category",
        "tbl_title": "Stage 2 — per-group metrics (fine taxonomy, held-out, oracle n={n})",
        "tbl_note": ("Precision = of those the model called X, how many were X (— = never predicted X: 0/0 undefined).  "
                     "Recall (hit rate) = of the real X, how many were caught.  "
                     "Accuracy¹ = (hit+correct reject)/total (one-vs-rest).\n"
                     "¹ inflated by true-negatives on rare classes — ALWAYS read with the support (n). "
                     "⚠ = support < 5 (unstable; read the raw fraction hits/n). "
                     "GLOBAL row = micro (precision = recall = F1 = accuracy = {acc:.0%}, multiclass identity); macro-F1 = {f1:.2f} (in title)."),
        "cols": ["group", "n", "hits", "precision", "recall\n(hit)", "F1", "accuracy¹\n(1-vs-rest)"],
        "global_row": "GLOBAL (micro)",
    },
}


def per_class_from_cm(cm: np.ndarray):
    """Recalcula precisao/recall/f1/acuracia-OvR/suporte da matriz crua (linhas=verdadeiro)."""
    n = cm.sum()
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)          # por linha (verdadeiro)
    pred = cm.sum(axis=0).astype(float)             # por coluna (previsto)
    fn = support - tp
    fp = pred - tp
    tn = n - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred > 0, tp / pred, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    accuracy_ovr = (tp + tn) / n
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": support,
        "precision": precision, "recall": recall, "f1": f1,
        "precision_defined": pred > 0,        # False => modelo nunca previu a classe (precisao = 0/0)
        "accuracy_ovr": accuracy_ovr,
        "n": int(n),
        "global_accuracy": float(tp.sum() / n),
    }


def _pct(v, defined=True):
    """% formatado; '—' quando indefinido (precisao 0/0)."""
    return f"{v:.0%}" if defined else "—"


def verify(node, cm, classes):
    """Confere as metricas recalculadas contra as armazenadas no JSON. Falha (assert) se divergir."""
    m = per_class_from_cm(cm)
    jp = node["precisao_por_classe"]; jr = node["recall_por_classe"]
    jf = node["f1_por_classe"]; js = node["suporte_por_classe"]
    for i, c in enumerate(classes):
        assert abs(m["precision"][i] - jp[c]) < 1e-6, f"precisao {c}: {m['precision'][i]} != {jp[c]}"
        assert abs(m["recall"][i] - jr[c]) < 1e-6, f"recall {c}: {m['recall'][i]} != {jr[c]}"
        assert abs(m["f1"][i] - jf[c]) < 1e-6, f"f1 {c}: {m['f1'][i]} != {jf[c]}"
        assert int(m["support"][i]) == js[c], f"suporte {c}: {m['support'][i]} != {js[c]}"
    assert abs(m["global_accuracy"] - node["accuracy"]) < 1e-6, \
        f"acuracia global: {m['global_accuracy']} != {node['accuracy']}"
    return m


# --------------------------------------------------------------------------- figuras
BLUE = LinearSegmentedColormap.from_list("b", ["#f7fbff", "#08306b"])


def draw_confusion(cm, classes, m, f1_macro, ci, n, lang, out_stub):
    from matplotlib.patches import Rectangle
    T = I18N[lang]
    k = len(classes)
    names = [PRETTY.get(c, c) for c in classes]
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row_sum == 0, 1, row_sum)) * 100.0
    lo, hi = ci

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 7.2))

    # painel 1: contagens, cor por fracao da linha (comparavel entre classes de tamanhos diferentes)
    axL.imshow(norm / 100.0, cmap=BLUE, vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            v = cm[i, j]
            txt = f"{v}" if v else "·"
            axL.text(j, i, txt, ha="center", va="center",
                     fontsize=15, weight="bold" if i == j else "normal",
                     color="white" if norm[i, j] >= 50 else ("#111" if v else "#bbb"))
    axL.set_title(T["panel_counts"], fontsize=11, weight="bold")

    # painel 2: recall por linha (%)
    axR.imshow(norm / 100.0, cmap=BLUE, vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            p = norm[i, j]
            txt = f"{p:.0f}%" if p >= 0.5 else "·"
            axR.text(j, i, txt, ha="center", va="center",
                     fontsize=12, weight="bold" if i == j else "normal",
                     color="white" if p >= 50 else ("#111" if p >= 0.5 else "#bbb"))
    axR.set_title(T["panel_norm"], fontsize=11, weight="bold")

    for ax in (axL, axR):
        ax.set_xticks(range(k)); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(k)); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(T["xlabel"], fontsize=10)
        ax.set_ylabel(T["ylabel"], fontsize=10)
        ax.set_xticks(np.arange(-.5, k, 1), minor=True)
        ax.set_yticks(np.arange(-.5, k, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", length=0)
        # contorna a DIAGONAL (acerto) p/ que ela nunca se confunda com um off-diagonal de mesmo valor
        for i in range(k):
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="#d62728", linewidth=2.2, zorder=5))

    # recall por classe anotado a' direita do painel 2
    for i in range(k):
        axR.text(k - 0.35, i, f"  {m['recall'][i]*100:4.0f}%", ha="left", va="center",
                 fontsize=9, color="#08306b", weight="bold", clip_on=False)

    fig.suptitle(T["title_cm"] + "\n"
                 + T["sub_cm"].format(n=n, acc=m["global_accuracy"], f1=f1_macro, lo=lo, hi=hi),
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"{out_stub}.{ext}", dpi=140)
    plt.close(fig)


def draw_table(cm, classes, m, f1_macro, n, lang, out_stub):
    T = I18N[lang]
    k = len(classes)
    names = [PRETTY.get(c, c) for c in classes]
    # ordena por suporte desc p/ leitura (classes robustas no topo)
    order = np.argsort(-m["support"])
    fig, ax = plt.subplots(figsize=(11.5, 0.5 * (k + 2) + 1.4))
    ax.axis("off")

    WARN_BG = "#fdeedd"   # bege: classe rara (n<5) — NAO destacar como boa/ruim

    def shade(val):
        x = float(val)
        return (0.86 - 0.5 * x, 0.92 - 0.18 * x, 0.86 - 0.5 * x)  # verde mais forte = melhor

    cell_text, cell_colors = [], []
    for i in order:
        warn = m["support"][i] < 5
        cell_text.append([
            ("⚠ " if warn else "") + names[i],
            f"{int(m['support'][i])}",
            f"{int(m['tp'][i])}",
            _pct(m["precision"][i], m["precision_defined"][i]),
            _pct(m["recall"][i]),
            f"{m['f1'][i]:.2f}",
            _pct(m["accuracy_ovr"][i]),
        ])
        # classes raras: TUDO em bege neutro (um 100% com n=2 nao pode parecer o melhor da tabela)
        if warn:
            cell_colors.append([WARN_BG] * 7)
        else:
            cell_colors.append([
                "#eef1f6", "#eef1f6", "#eef1f6",
                shade(m["precision"][i]) if m["precision_defined"][i] else "#eef1f6",
                shade(m["recall"][i]),
                shade(m["f1"][i]),
                "#eaeef4",   # acuracia OvR neutra (nao destacar: enganosa)
            ])

    # linha GLOBAL — MICRO (single-label multiclasse): precisao = recall = F1 = acuracia = acuracia global.
    # (F1-macro 0.36 fica no titulo; aqui tudo e' micro p/ a linha ser internamente derivavel.)
    ga = m["global_accuracy"]
    cell_text.append([
        T["global_row"], f"{int(n)}", f"{int(m['tp'].sum())}",
        f"{ga:.0%}", f"{ga:.0%}", f"{ga:.2f}", f"{ga:.0%}",
    ])
    cell_colors.append(["#d7dde6"] * 7)

    tbl = ax.table(cellText=cell_text, colLabels=T["cols"],
                   cellColours=cell_colors, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
    base_h = tbl[(1, 0)].get_height()
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white"); cell.set_linewidth(1.5)
        if r == 0:
            cell.set_facecolor("#08306b"); cell.set_text_props(color="white", weight="bold")
            cell.set_height(base_h * 1.5)          # cabeçalho de 2 linhas precisa de mais altura
        elif r == len(cell_text):
            cell.set_text_props(weight="bold")
        if c == 0:
            cell.set_text_props(ha="left")
    # destaca colunas pedidas: precisao (3) e acuracia (6)
    for r in range(1, len(cell_text) + 1):
        for c in (3, 6):
            tbl[(r, c)].set_text_props(weight="bold")

    ax.set_title(T["tbl_title"].format(n=n), fontsize=13, weight="bold", pad=18)
    fig.text(0.5, 0.015, T["tbl_note"].format(acc=m["global_accuracy"], f1=f1_macro),
             ha="center", va="bottom", fontsize=8.3, color="#333", wrap=True)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(REP / f"{out_stub}.{ext}", dpi=140)
    plt.close(fig)


def md_table(classes, m, order):
    names = [PRETTY.get(c, c) for c in classes]
    rows = ""
    for i in order:
        warn = " ⚠" if m["support"][i] < 5 else ""
        prec = f"**{m['precision'][i]:.0%}**" if m["precision_defined"][i] else "— *(0/0)*"
        rows += (f"| `{names[i]}`{warn} | {int(m['support'][i])} | {int(m['tp'][i])} | "
                 f"{prec} | {m['recall'][i]:.0%} | {m['f1'][i]:.2f} | "
                 f"**{m['accuracy_ovr'][i]:.0%}** |\n")
    return rows


def top_confusions(cm, classes, k=4):
    """Maiores confusoes fora da diagonal (verdadeiro -> previsto), como % da linha."""
    names = [PRETTY.get(c, c) for c in classes]
    cm = np.asarray(cm)
    items = []
    for i in range(len(classes)):
        rs = cm[i].sum()
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                items.append((cm[i, j], cm[i, j] / rs if rs else 0.0, names[i], names[j]))
    items.sort(key=lambda t: (-t[0], -t[1]))
    return items[:k]


def main():
    rep_json = REP / "evaluation_report.json"
    if not rep_json.exists():
        sys.exit(f"[erro] {rep_json} nao existe — rode antes: python scripts/run_experiment.py")
    report = json.loads(rep_json.read_text())
    e2 = report.get("estagio2_categoria")
    if not e2:
        sys.exit("[erro] 'estagio2_categoria' ausente no relatorio (treine com train.multiclass: true).")

    out = {"verificacao": "ok", "modos": {}}
    for modo, stub_cm, stub_tb in (
        ("oraculo", "confusion_matrix_stage2_fina", "metricas_stage2_por_grupo"),
        ("condicional_ao_gate", "confusion_matrix_stage2_fina_gate", "metricas_stage2_por_grupo_gate"),
    ):
        node = e2.get(modo)
        if not node:
            continue
        fina = node["fina"]["por_prototipo"]
        classes = fina["classes"]
        cm = np.array(fina["confusion_matrix"])
        n = int(node["n_erro"])
        f1_macro = float(fina["f1_macro"])
        ci = fina.get("ci95_f1_macro", [float("nan"), float("nan")])
        m = verify(fina, cm, classes)          # <-- assert contra JSON
        order = list(np.argsort(-m["support"]))

        # PT
        draw_confusion(cm, classes, m, f1_macro, ci, n, "pt", stub_cm)
        draw_table(cm, classes, m, f1_macro, n, "pt", stub_tb)
        # EN (so' o oraculo precisa de versao EN p/ apresentacao)
        if modo == "oraculo":
            draw_confusion(cm, classes, m, f1_macro, ci, n, "en", stub_cm + "_en")
            draw_table(cm, classes, m, f1_macro, n, "en", "stage2_per_group_metrics_en")

        out["modos"][modo] = {
            "n_erro": n, "acuracia_global": m["global_accuracy"], "f1_macro": f1_macro,
            "classes": [PRETTY.get(c, c) for c in classes],
            "por_grupo": {
                PRETTY.get(c, c): {
                    "suporte": int(m["support"][i]), "acertos": int(m["tp"][i]),
                    "precisao": float(m["precision"][i]), "recall": float(m["recall"][i]),
                    "f1": float(m["f1"][i]), "acuracia_ovr": float(m["accuracy_ovr"][i]),
                } for i, c in enumerate(classes)
            },
            "confusion_matrix": cm.tolist(),
            "_order": order,
        }

    # --------- relatorio markdown ---------
    orc = out["modos"]["oraculo"]
    classes = e2["oraculo"]["fina"]["por_prototipo"]["classes"]
    fina_node = e2["oraculo"]["fina"]["por_prototipo"]
    m_orc = verify(fina_node, np.array(fina_node["confusion_matrix"]), classes)
    order = list(np.argsort(-m_orc["support"]))
    coarse = e2["oraculo"]["grossa"]
    cm_orc = np.array(fina_node["confusion_matrix"])
    names = [PRETTY.get(c, c) for c in classes]

    f_lo, f_hi = fina_node.get("ci95_f1_macro", [float("nan")] * 2)
    c_lo, c_hi = coarse.get("ci95_f1_macro", [float("nan")] * 2)

    # callout de classes raras DERIVADO dos dados (nunca contradiz os ⚠ da tabela)
    rare = [(names[i], int(m_orc["support"][i])) for i in range(len(classes)) if m_orc["support"][i] < 5]
    rare_str = ", ".join(f"`{nm}` n={s}" for nm, s in rare) or "nenhuma"
    n_small = int((m_orc["support"] <= 16).sum())
    # precisao indefinida (modelo nunca previu a classe)
    undef = [names[i] for i in range(len(classes)) if not m_orc["precision_defined"][i]]
    undef_str = (", ".join(f"`{u}`" for u in undef)) if undef else "—"

    # maiores confusoes (verdadeiro -> previsto)
    confs = top_confusions(cm_orc, classes, k=4)
    conf_lines = "".join(
        f"  - **{a} → {b}**: {int(cnt)} casos ({pct:.0%} da linha de `{a}`)\n"
        for cnt, pct, a, b in confs)

    # acesso por nome (robusto a mudanca de ordem)
    idx = {PRETTY.get(c, c): i for i, c in enumerate(classes)}
    def g(name, key):
        return m_orc[key][idx[name]]
    def cmcell(true_name, pred_name):
        return int(cm_orc[idx[true_name], idx[pred_name]])

    md = f"""# Estágio 2 — Matriz de confusão por categoria (taxonomia FINA, 6 grupos)

> **Held-out** (teste processado 1×). O Estágio 2 atribui a **categoria do erro** apenas a imagens
> de **erro**, pelo **protótipo de categoria mais próximo** no espaço SupCon (método canônico).
> Modo **ORÁCULO** = classifica os **{orc['n_erro']} erros reais** do teste (isola a qualidade do
> Estágio 2, independente do gate). Apêndice = **condicional ao gate** (produção).
>
> ⚠️ **Escala/fragilidade:** todas as métricas finas vêm de **um único held-out de {orc['n_erro']} erros**;
> **{n_small} das 6 classes têm n≤16** (raras: {rare_str}). Tratar como **indicativo**, não definitivo.

## Resumo (oráculo, n={orc['n_erro']})

- **Acurácia global (multiclasse):** **{orc['acuracia_global']:.1%}** — fração dos {orc['n_erro']} erros com a categoria correta (acertos na diagonal: **{int(m_orc['tp'].sum())}/{orc['n_erro']}**).
- **F1-macro:** **{orc['f1_macro']:.2f}** [IC95 {f_lo:.2f}–{f_hi:.2f}] — intervalo largo (n pequeno): leia como ordem de grandeza.
- Taxonomia **grossa** (3 superclasses): acurácia **{coarse['accuracy']:.1%}**, F1-macro **{coarse['f1_macro']:.2f}** [IC95 {c_lo:.2f}–{c_hi:.2f}].
  > **Por que a grossa parece melhor?** O salto fina→grossa é, em boa parte, **agregação 6→3**: o mapa
  > funde justamente os vizinhos mais confundidos (`overlay`↔`disordered_layout`, `black_bars`↔`empty_space`).
  > É a **mesma representação** medida numa tarefa **mais fácil** (3 classes), não um modelo melhor — e os
  > dois IC95 ({f_lo:.2f}–{f_hi:.2f} vs {c_lo:.2f}–{c_hi:.2f}) **se sobrepõem**: o ganho é **sugestivo, não estatístico**.

## Métricas por grupo (precisão e acurácia em destaque)

| Grupo | n | acertos | **Precisão** | Recall (acerto) | F1 | **Acurácia¹** (1-vs-resto) |
|---|---|---|---|---|---|---|
{md_table(classes, m_orc, order)}
> ¹ **Acurácia (1-vs-resto)** = (acertos + rejeições corretas)/total. **Cuidado:** ela fica *alta*
> para classes **raras** ({rare_str}) só porque há muitos verdadeiros-negativos — `orientation` tem
> acurácia 98% mas **recall 0%** (não acertou nenhum). Por isso, para "como o modelo está **acertando**
> cada grupo", a métrica honesta é o **recall (taxa de acerto)** + o **suporte (n)**. A **acurácia global
> multiclasse** ({orc['acuracia_global']:.1%}) é a média real (micro: precisão = recall = acurácia).
> **Precisão `—`** = o modelo **nunca previu** essa classe ({undef_str}): precisão 0/0 é **indefinida**, não medida.

## Como o modelo está acertando (leitura das confusões)

- ✅ **`black_bars` é a única classe utilizável** — precisão **{g('black_bars','precision'):.0%}** / recall **{g('black_bars','recall'):.0%}** (n={int(g('black_bars','support'))}).
- ⚠️ **`empty_space` vaza para `black_bars`**: {cmcell('empty_space','black_bars')} de {int(g('empty_space','support'))} ({cmcell('empty_space','black_bars')/g('empty_space','support'):.0%}) viram `black_bars` — é mandado para `black_bars` **mais vezes** do que é acertado (recall só {g('empty_space','recall'):.0%}).
- ⚠️ **`overlay` empata acerto e erro**: vai para `disordered_layout` ({cmcell('overlay','disordered_layout')}) **tão frequentemente quanto acerta** ({cmcell('overlay','overlay')}) — ambos {cmcell('overlay','overlay')/g('overlay','support'):.0%} da linha. Na figura, a **diagonal tem borda vermelha** para distinguir o acerto.
- ⛔ **Classes raras (n<5) não são interpretáveis** ({rare_str}) — `orientation` nunca foi prevista (precisão indefinida); `distortion` é {int(g('distortion','tp'))}/{int(g('distortion','support'))} (parece 100% de precisão por acaso de n).
- **Maiores confusões (verdadeiro → previsto):**
{conf_lines}
## Matriz de confusão (linhas = verdadeiro · colunas = previsto)

| verdadeiro ↓ \\ previsto → | {' | '.join(PRETTY.get(c, c) for c in classes)} | **n** |
|---|{'---|' * len(classes)}---|
"""
    for i in range(len(classes)):
        cells = " | ".join(f"**{cm_orc[i,j]}**" if i == j else f"{cm_orc[i,j]}" for j in range(len(classes)))
        md += f"| **{names[i]}** | {cells} | {int(cm_orc[i].sum())} |\n"

    # apendice condicional
    if "condicional_ao_gate" in out["modos"]:
        cd = out["modos"]["condicional_ao_gate"]
        cdclasses = e2["condicional_ao_gate"]["fina"]["por_prototipo"]["classes"]
        m_cd = verify(e2["condicional_ao_gate"]["fina"]["por_prototipo"], np.array(cd["confusion_matrix"]), cdclasses)
        order_cd = list(np.argsort(-m_cd["support"]))
        md += f"""
## Apêndice — condicional ao gate E1 (produção, n={cd['n_erro']})

Só os erros que o Estágio 1 sinalizou ({cd['n_erro']}/{orc['n_erro']}). Acurácia global **{cd['acuracia_global']:.1%}**, F1-macro **{cd['f1_macro']:.2f}**.

| Grupo | n | acertos | **Precisão** | Recall | F1 | **Acurácia¹** |
|---|---|---|---|---|---|---|
{md_table(cdclasses, m_cd, order_cd)}
"""

    md += """
## Figuras geradas
- `confusion_matrix_stage2_fina.png` / `.pdf` (PT) · `_en` (EN) — matriz 6×6 (contagens + recall).
- `metricas_stage2_por_grupo.png` / `.pdf` (PT) · `stage2_per_group_metrics_en.*` (EN) — tabela por grupo.

*Métricas recalculadas da matriz crua e conferidas contra `evaluation_report.json` (assert).*
"""
    (REP / "STAGE2_MATRIZ_CONFUSAO.md").write_text(md)

    print(json.dumps({k: v for k, v in out.items() if k != "modos"}
                     | {"modos": {kk: {x: vv[x] for x in ("n_erro", "acuracia_global", "f1_macro")}
                                   for kk, vv in out["modos"].items()}}, indent=2, ensure_ascii=False))
    print(f"\n[ok] figuras + STAGE2_MATRIZ_CONFUSAO.md em {REP}")


if __name__ == "__main__":
    main()
