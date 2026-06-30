# Spec de coleta #2 — telas limpas **foldable near-square** (a única alavanca que move o domínio de produção)

> Escrita após o experimento #1 (jun/2026). **Tese, agora provada por dois experimentos:** o gargalo
> do detector é a **pobreza de telas limpas no domínio foldable near-square** — e é um problema de
> **conteúdo**, não de aspecto/resolução. Síntese barata não resolve; **só dado foldable real resolve.**
> Esta spec é executável: alvos quantificados, fontes ranqueadas, mudanças de código nomeadas,
> critérios de aceite medidos com o harness que já isola o domínio.

---

## 0. Por que #2 (e por que #1 não bastou) — a evidência

| Experimento | O que testou | Resultado no domínio **foldable** (v3-test, 41 clean near-square) |
|---|---|---|
| **Cross-eval** (modelo plus+L_reg4 → test do v3) | o modelo atual no domínio de produção | free-confound **0.731** (vs 0.721 baseline = +0.01, ruído); especificidade **0.512** (20 FP/41) |
| **#1 reflow near-square AR** (síntese de aspecto 0.96 a partir de conteúdo phone/desktop) | a alavanca **barata** | especificidade **0.512 → 0.512** (os MESMOS 20 FP); free-confound 0.731 → 0.713 |

**Conclusão dura:** reflowar conteúdo phone/desktop para AR 0.96 ensina *"aspecto 0.96 pode ser limpo"*
(ajuda clean diversas) mas **não cobre o conteúdo foldable** — as 41 telas foldable reais continuam com
os exatos 20 falso-positivos. O gap é **conteúdo de renderização foldable** (status bar, multi-pane,
densidade, fontes, app-em-postura), que **só telas foldable reais** trazem.

**Causa-raiz quantificada** (assimetria que cria o gap):

| | LIMPAS foldable (hoje) | ERROS (tickets) |
|---|---|---|
| imagens / grupos | 172 / **16 sessões** | 277 / 217 tickets |
| device / campanha | **1 device, 1 dia** | múltiplos |
| resoluções | **1** (2076×2152) | 108 |
| form factors | **só `unknown`** | unfold·fold·tent·laptop·desktop |
| orientações | **só `unknown`** | portrait·landscape |

O modelo constrói "limpo" a partir de **16 sessões quase idênticas**; toda tela foldable nova cai fora →
falso-positivo. Diversificar o **limpo foldable** é o que fecha isso.

---

## 1. Alvo de coleta (derivado da distribuição REAL dos erros)

Resoluções/AR dos erros (medido em `processed_v3_plus`, n=1577; mediana AR **0.96**, **41% near-square**):

| Bucket (AR = w/h) | Resoluções-chave dos erros | Prioridade da coleta |
|---|---|---|
| **near-square 0.85–1.18 (foldable desdobrado)** ⭐ | **2076×2152 (0.96)**, 2232×2484 (0.90), 2484×2232 (1.11), 2200×2480 (0.89) | **MÁXIMA** (o gargalo) |
| portrait < 0.85 (foldable dobrado / phone) | 1080×2520, 1170×2532, 1080×2400 | média (cobrir cover-display) |
| landscape > 1.18 (foldable tent/laptop, desktop) | 2360×1640 (1.44), 1920×1080, 2880×1800 | média |

> ⚠️ **Não** é "tablet near 2076×2152": a evidência (AR 0.96 + postures unfold/fold/tent/laptop) aponta
> **dispositivo dobrável** (Galaxy Z Fold / Pixel Fold) em várias posturas. Priorizar **foldable
> desdobrado** (near-square) e as outras posturas — **conteúdo de app real**, muitos apps distintos.

### 1.1 Metas quantificadas

| Eixo | Hoje | **Meta #2** |
|---|---|---|
| nº de limpas foldable | 172 | **+300 a +500 novas** |
| grupos distintos (device×app×tela) | 16 | **≥ 50** |
| devices / perfis foldable | 1 | **≥ 3** (ex.: Z Fold + Pixel Fold + 1 emulador) |
| postures (`form_factor`) | só `unknown` | **unfold · fold · tent · laptop** (≥ 4) |
| orientações | só `unknown` | **portrait E landscape** |
| apps/telas distintos | ~poucos | **≥ 40** (home, settings, listas, forms, mídia, chat, maps, mensagens…) |
| **clean foldable no HELD-OUT (val+test)** | 0 (test é só conteúdo de 1 dia) | **val ≥ 20, test ≥ 40** nas resoluções dos erros |

---

## 2. Fontes, ranqueadas (sem depender de hardware)

### A. Emulador foldable (Android Studio AVDs) — **PRIMÁRIA, escalável, grátis** ⭐
- AVDs nativos: **Pixel Fold**, **7.6" Fold-in** (Z-Fold-like, desdobrado near-square), **8" Fold-out**,
  **Resizable (Experimental)** (alterna phone↔unfolded↔tablet), **Pixel Tablet**.
- Conteúdo: apps do próprio OS (Settings, Files, Contacts, Calculator, Clock, Messages, Maps…) + APKs/Play.
- Captura **nativa**: `adb exec-out screencap -p > tela.png` (resolução real do AVD, sem moldura).
- Postures: estados fold/unfold via controles do AVD / `adb emu`; tent/laptop nos AVDs que suportam.
- **Scriptável** (adb + UIAutomator/`monkey`) → centenas de telas de apps reais em renderização foldable
  genuína. **É o melhor caminho prático**: conteúdo foldable real, controlável, reprodutível.

### B. Devices físicos foldable — **PADRÃO-OURO, oportunístico**
- Galaxy Z Fold 4/5/6, Pixel Fold / Pixel 9 Pro Fold, OnePlus Open, Honor Magic V…
- Menor domain-gap (mesmos devices que geram os tickets). **Mesmo 1 device** rende algumas centenas de
  screenshots de alto valor → priorizar para **val/test** (onde o domain-gap mais contamina a métrica).
- Popular `form_factor` (posture real) e `orientation` na captura.

### C. Público/scrape — **SUPLEMENTO, curar com rigor**
- Marketing **tablet/large-screen/foldable** da App Store / Play (muitos apps já publicam) — near-square,
  mas estilizado (molduras/overlays de texto) → **cortar para a tela, sem overlay** (senão colide com
  `overlay`/`black_bars`). Copyright → **só treino interno**.
- Press kits Samsung Newsroom / Google Pixel (alta-res foldable) — copyright, idem.

> **Lição do #1 aplicada às fontes:** maximizar **diversidade de CONTEÚDO** (apps/telas), não só de
> aspecto. 500 telas de 3 apps re-introduzem confound de conteúdo; 300 telas de 40 apps, não.

---

## 3. Protocolo de captura

Para cada **(device/AVD × app × tela)** — cada combinação é **1 grupo**:
1. Navegar até uma tela **limpa, completa, sem erro** (conteúdo carregado, sem loading/placeholder).
2. Capturar em **≥ 2 postures** quando aplicável (unfold + fold; +tent/laptop se houver) e **2 orientações**.
3. Salvar **PNG nativo** (sem redimensionar, sem moldura de device, sem barra do emulador).
4. Registrar metadados: `device`, `app`, `screen`, `form_factor` (posture), `orientation`.
5. **Variar apps e telas** — meta ≥ 40 apps/telas distintos; **teto de ~8 telas por app** (anti-confound).

**Checklist de qualidade (rejeitar se):** tela com loading/skeleton; moldura/sombra de device; overlay de
marketing; barra de status do emulador visível; near-duplicata (mesmo app+tela+posture já capturado);
conteúdo que **parece** erro (faixa preta legítima de cinema, splash) — ambíguo, descartar.

---

## 4. Ingestão no pipeline (+ mudanças de código necessárias)

A foldable entra pelo mesmo caminho do pool público — `data/clean_extra/` → `merge_clean_extra.py` →
**novo** `data/processed_v3_fold/` (não-destrutivo, como o `_plus`). **Três mudanças são obrigatórias**
(o caminho atual perde o posture, o que cega a métrica controlada):

### 4.1 Manifesto estendido (`labels_extra.csv` / novo `capture_foldable.py`)
Schema atual = `path, source, w, h, aspect, group, phash`. **Adicionar**: `form_factor` (unfold/fold/tent/
laptop), `orientation` (portrait/landscape), `device`. (O `group` deve ser `device:app:screen`, não 1-por-imagem.)

### 4.2 `scripts/merge_clean_extra.py` — usar o posture real (hoje hardcoda `"external"`)
- **L136**: `form_factor: "external"` → **ler do manifesto** (`r.get("form_factor") or "external"`); idem
  `orientation` (usar o do manifesto, não só o heurístico `h≥w`). **Sem isso, o subconjunto controlado
  (`unfold·portrait·screenshot`) continua degenerado** e a métrica honesta de acurácia real não nasce.
- **Split por GRUPO, não por imagem** (L78–81 hoje assume 1 img = 1 grupo): agrupar as linhas por `group`
  e atribuir **grupos inteiros** a train/val/test → postures/orientações da mesma tela não vazam entre
  splits. A asserção de vazamento (L165–170) já existe; manter.
- Garantir **clean foldable em val E test** (frações 0.15/0.25) **nas resoluções dos erros** → cria o
  held-out onde **limpo e erro compartilham resolução** (critério de aceite da auditoria, hoje impossível).

### 4.3 Resto do pipeline — **sem mudança** (a propriedade que torna #2 barato)
- `merge` **regenera todo o train/synthetic** das limpas de treino (orig + **foldable**) → os erros
  sintéticos **herdam as resoluções foldable** (anti-confound pelo lado do erro, de graça).
- Depois: `run_experiment.py --config configs/<fold>.yaml --processed data/processed_v3_fold`
  (criar config com `emb_dir`/`reports_dir` próprios; backbone L_reg4; **`reflow_match_error_ar` pode
  ficar OFF** — #2 ataca conteúdo, não aspecto).

### 4.4 Regras metodológicas (senão a coleta NÃO quebra o confound)
- **Manter resoluções nativas** foldable; **nunca** redimensionar tudo para 2076×2152.
- **Nunca** padding **preto** (colide com `black_bars`); o pré-processo `pad` usa cinza + máscara.
- Dedup **sha256 + p-hash** (o `fetch` já faz; manter no `capture_foldable`).
- Reencodar tudo em **PNG** mesma qualidade (senão formato vira atalho).
- Popular **`form_factor`/`orientation`** (o §4.2 acima) — destrava o subconjunto controlado.

---

## 5. Critérios de aceite (medidos isolando o domínio foldable)

Medir com o **harness de cross-eval** (mesmo do #1) + uma fatia **foldable-only** do held-out (filtra test
para clean foldable + erros). Isola o ganho no domínio de produção — não deixa clean diversas mascararem.

| # | Gate | Hoje (plus+L_reg4) | **Meta #2** |
|---|---|---:|---:|
| 1 ⭐ | **Especificidade no clean foldable** (held-out) | **0.512** | **> 0.65** estável (CI95 acima do alvo) |
| 2 ⭐ | **Free-confound AUROC no domínio foldable** | **0.731** | **> 0.78** |
| 3 | Subconjunto controlado `unfold·portrait·screenshot` | degenerado (clean ausente) | **não-degenerado** e modelo **supera** o baseline de confound |
| 4 | Gap treino→teste do gate AUROC | 0.18 | **< 0.15** |
| 5 | Held-out com limpo **e** erro na MESMA resolução near-square | inexistente | **existe** → 1ª acurácia real honesta |
| 6 | `disordered_layout` fora do zero **ou** fundido na grossa | classif. 0.00 (n=10) | recall/F1 > 0 **ou** decisão de fundir |
| 7 | Estabilidade multi-seed (`multiseed_stability.py`) | 0.86 ± 0.001 | mantém (sem regressão) |

> **Como provar que valeu (e não foi clean diversa de novo):** rodar a fatia foldable-only ANTES e DEPOIS;
> o gate #1 (especificidade foldable) **tem que sair de 0.512**. Foi exatamente onde o #1 falhou (0.512→0.512).

---

## 6. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Emulador ≠ device real (fontes/sub-pixel) → gap residual | usar **≥ 1 device físico** para **val/test**; emulador para volume de **train** |
| Marketing com molduras/overlay → ruído de rótulo / colisão de classe | cortar para a tela, **sem overlay**; curar; preferir A/B |
| Coletar muito de poucos apps → confound de conteúdo (lição do #1) | teto ~8 telas/app; **≥ 40 apps**; estratificar |
| Foldable só no train | forçar val ≥ 20 / test ≥ 40 **nas resoluções dos erros** (§4.2) |
| Vazamento de grupo (postures da mesma tela em splits diferentes) | **split por grupo** `device:app:screen` (§4.2) + asserção existente |

---

## 7. Ordem de execução

```
1. capture_foldable.py (emulador A + device B) → data/clean_extra_fold/ + labels_extra estendido  [o grosso]
2. patch merge_clean_extra.py (§4.2: form_factor real + split por grupo) → data/processed_v3_fold
3. config fold (L_reg4, emb/reports próprios) → run_experiment.py --processed data/processed_v3_fold
4. cross-eval harness + fatia foldable-only → checar gates §5 (especificidade foldable É o número)
5. se passar: atualizar ROADMAP/RELATORIO_FINAL com a 1ª acurácia real honesta; congelar config
```

**Resumo de uma linha:** colete **conteúdo foldable real, diverso em apps e postures, em val/test nas
resoluções dos erros** — é a única alavanca que tira a especificidade foldable de 0.512, provado por A/B.
