---
marp: true
theme: default
paginate: true
footer: "Detector de Erros de Layout em UI · Rede Siamesa sobre DINOv2"
math: katex
---

<!--
========================================================================
COMO USAR ESTE DECK
- VS Code: instale a extensão "Marp for VS Code" → abra este arquivo →
  ícone de preview. Exportar: paleta de comandos → "Marp: Export Slide Deck"
  (PDF / PPTX / HTML).
- CLI:   npx @marp-team/marp-cli@latest docs/slides_pipeline_executivo.md --pdf
         npx @marp-team/marp-cli@latest docs/slides_pipeline_executivo.md --pptx
- Os comentários "🎤 Fala:" em cada slide são NOTAS DO APRESENTADOR
  (aparecem no modo apresentador e na exportação para PPTX).
- reveal.js: troque o front-matter por uma config reveal e use "---" como
  separador de slides horizontais (o conteúdo Markdown é compatível).
========================================================================
-->

<style>
section { font-size: 25px; }
h1 { font-size: 1.5em; }
h2 { font-size: 1.15em; color: #37474F; }
table { font-size: 0.78em; }
code { font-size: 0.92em; }
section.lead { text-align: center; }
.chip { font-weight: 700; letter-spacing: .02em; }
.muted { color: #607D8B; }
.big { font-size: 1.25em; line-height: 1.4; }
</style>

<!-- _class: lead -->
<!-- _paginate: false -->

# Detector de Erros de Layout em UI

## Rede Siamesa sobre DINOv2 — pipeline executivo

<br>

**Decisão em dois estágios:** a tela **tem erro** de layout? → **de que tipo** é o erro?

<span class="muted">Apresentação do pipeline · 5 etapas até a decisão · números **jun/2026** (held-out honesto)</span>

<br>

<span class="muted" style="font-size:0.7em">⚠️ Versão jun/2026 — corrige números REVOGADOS (acc 0.85 / AUROC 0.90 / "test 54": inválidos por vazamento + seleção que via o teste, auditoria Fase 0). Números abaixo = held-out honesto, teste trancado, 1×. Fonte: <code>RELATORIO_APRESENTACAO.md</code>.</span>

<!--
🎤 Fala (abertura, 15s): "Esse pipeline recebe o print de uma tela e decide em DOIS
estágios: primeiro um gate — tem erro de layout ou não? — e, quando tem, de que tipo é o
erro. O desafio não foi a rede em si — foi garantir que ela aprendesse a reconhecer o ERRO,
e não características do APARELHO que vazam nos dados. Vou percorrer as etapas até a decisão.
AVISO: esta versão corrige números antigos REVOGADOS (acc 0.85 / AUROC 0.90 / test 54 imgs),
que vinham de split com vazamento. Os de hoje são o held-out honesto, teste trancado, 1×."
-->

---

## A mensagem central

<br>

<div class="big">

Todo o pipeline existe para responder uma coisa:

### 🎯 "Como impedir que o modelo **trapaceie**?"

</div>

<br>

- As telas **limpas** vieram **todas de um único aparelho** (resolução 2076×2152).
- As telas **com erro** são variadas (resoluções diversas, fotos, dobráveis).
- A regra trivial — *"resolução ≠ 2076×2152 ⇒ erro"* — dá **AUROC 0.99** **sem olhar o layout** → a métrica **global** é ~98% **trapaça**.

> Cada etapa adiante força o modelo a olhar o **conteúdo do erro**, não o **aparelho**.
> Honestidade: o confound foi **atenuado, não vencido** — o teto do gate é de **DADO**.

<!--
🎤 Fala: "Existe uma armadilha nos dados: como as telas limpas vieram todas do mesmo
device, dá pra 'acertar' 98% só olhando a resolução — detectando o aparelho, não o erro.
Guardem essa frase: tudo aqui serve pra impedir a trapaça. Se eu travar em alguma
pergunta, eu volto pra ela."
-->

---

<!-- _class: lead -->

# As 5 etapas

![bg right:55% fit](pipeline-executivo.png)

<div style="text-align:left">

1. 🗂️ **Fontes de Dados**
2. 🧪 **Geração de Sintéticos** *(anti-trapaça)*
3. 🔒 **Extração de Características**
4. 🎯 **Rede Siamesa** *(a parte que aprende)*
5. ⚖️ **Decisão** → ✅ / ❌

</div>

<span class="muted">Etapas 1–2 rodam **1× na preparação**; 3→5 rodam **a cada tela nova**.</span>

<!--
🎤 Fala: "Esse é o mapa. Sigam as setas grossas: é o caminho de uma tela até a decisão.
Importante separar duas fases — preparar o modelo (etapas 1 e 2, uma vez) e usar o modelo
(etapas 3 a 5, toda vez que chega uma tela). Vou caixa por caixa."
-->

---

## 🗂️ Etapa 1 — Fontes de Dados

<br>

| Fonte | Quantidade | Característica |
|---|---|---|
| Telas **LIMPAS** reais | 172 | **um único device** · todas 2076×2152 (o **confound**) |
| Telas **COM ERRO** reais | 369 | 6 categorias · resoluções diversas · fotos · fold/laptop/tent |

<br>

- **541 imagens reais únicas.** A assimetria é o problema central (tratado na etapa 2).
- Split **agrupado por ticket + sessão**, estratificado por categoria → **0 vazamento** (`tests/test_split_isolation.py`). *(train **330**: 105+225 · val **81**: 26+55 · test **130**: 41+89)*
- Augmentação de treino (não é dado real): **+420 sintéticos-erro + +420 limpas-reflow**.

<!--
🎤 Fala: "Duas fontes. Repare na assimetria: as limpas são todas do mesmo aparelho; as com
erro são de tudo quanto é tipo. É daí que vem a armadilha. O split é por ticket pra não
vazar imagens do mesmo bug entre treino e teste."
💡 Analogia: detector de fraude onde toda transação honesta veio do mesmo banco.
-->

---

## 🧪 Etapa 2 — Anti-confound dos DOIS lados

> **A etapa mais importante — o diferencial do trabalho.**

Atacamos o confound pelos **dois lados**:

- **Lado do ERRO:** injetamos erros **artificiais** nas próprias telas **limpas**, na **mesma
  resolução / aparelho** → o par difere **só pelo erro**.
- **Lado do LIMPO (REFLOW — novo):** variantes **limpas** de layout legítimo (scroll,
  dual-pane, outro aspect-ratio, espaçamento), algumas em **outras resoluções** → a classe
  limpa deixa de ser só 2076×2152.

| `black_region` · `empty_space` · `overlay` · `disorder` · `cropped` (erro) | `reflow` (limpo) |
|---|---|
| faixa preta · região apagada · sobreposição · desalinho · corte | mesmo conteúdo, layout diferente = **ainda limpo** |

**Resultado medido (honesto):** o reflow **reduz** o rastreamento do confound (na validação,
prever resolução **0.62** < prever erro **0.65**), mas no **held-out** o gap fica ~0 → confound
**atenuado, não vencido**.

<!--
🎤 Fala: "Para tirar a trapaça, ataco pelos DOIS lados. Lado do erro: pego a tela limpa e
estrago ela mesma, mantendo resolução e device idênticos — a única diferença é o erro. Lado
do limpo, a novidade desta rodada, o reflow: mostro variantes LIMPAS de layout legítimo,
algumas em outras resoluções, rotuladas como limpas — assim a classe limpa não é mais só um
device. Honestidade: isso ATENUA o confound, não vence. No held-out o gap volta a ~0; vencer
depende de DADO (telas limpas diversas), não de tuning."
💡 Analogia: estrago a MESMA maçã (lado erro) e mostro a maçã sã em vários ângulos (reflow).
-->

---

## 🔒 Etapa 3 — Extração de Características

**Imagem → vetor de números**, em 3 passos:

1. **Pré-processamento sem distorção** — padding **cinza neutro** → 518×518 (preserva a
   geometria do erro; faixa preta não é espremida).
2. **Backbone DINOv2 ViT-S/14 — CONGELADO** — 22M params, **0 treináveis**. Grade 37×37 patches.
3. **Vetor de 1152-d** = token global (CLS, 384) **+ média + desvio-padrão** dos patches.

> 💬 Congelado ⇒ o vetor de cada imagem é **fixo** ⇒ **cacheado** em disco ⇒ treino em **segundos**.

<!--
🎤 Fala: "A tela vira números. Primeiro um pré-processamento que NÃO distorce — padding cinza
em vez de esticar. Depois o DINOv2, um modelo de visão pré-treinado em 142 milhões de
imagens, que está CONGELADO: ele não aprende nada do nosso problema, é só um extrator. A
saída é um vetor de 1152 dimensões."
🛡️ Se perguntarem por que congelar: poucas centenas de imagens reais + 22M params = overfit
imediato; ele decoraria o confound. O desvio-padrão dos patches captura faixa preta / espaço
vazio (regiões uniformes).
💡 Analogia: o DINOv2 é um olho especialista já formado; não reeducamos o olho.
-->

---

## 🎯 Etapa 4 — Rede Siamesa (a parte que aprende)

A **única** parte treinável (~330k params). Reorganiza o vetor para **separar limpo de erro**.

- **Cabeça de projeção `g(·)`** — *a mesma função, mesmos pesos, para qualquer tela* →
  vetor **`z` de 128-d** na **hiperesfera**. *Comparar duas telas = comparar `z₁` e `z₂`.*
- **Siamesa "one-class":** não comparamos contra **uma** tela de referência (a classe limpa é
  diversa!) — comparamos contra **protótipos** do conjunto limpo (etapa 5).

`LayerNorm → Linear(1152→256) → GELU → Dropout → Linear(256→128) → L2-norm`

<!--
🎤 Fala: "Aqui está a única parte que de fato aprende — só uns 330 mil parâmetros. Ela
reescreve o vetor num espaço de 128 dimensões DESENHADO para que telas limpas formem um
aglomerado e os erros caiam fora. 'Siamesa' = a mesma função aplicada a qualquer tela, e a
comparação acontece nos vetores. Não comparamos contra UMA tela boa, porque telas limpas de
apps diferentes são legitimamente diferentes — comparamos contra protótipos do conjunto limpo."
-->

---

## 🎯 Etapa 4 — A cabeça de aprendizado e a perda

**Duas saídas sobre `z`:**

| Saída | O que é | Para quê |
|---|---|---|
| Cabeça de **projeção** | produz `z` (128-d) | o coração métrico (siamês) · **decisor canônico** |
| Cabeça **auxiliar** | `Linear(128→7)` softmax (clean + 6 cat.) | classificador direto; gate lê `P(erro)=1−P(clean)` · **diagnóstico** |

**Função de perda (treino):**

$$\mathcal{L} = \text{SupCon}(z) + 0.6 \cdot \text{CE}(\text{aux 7 classes})$$

- **SupCon** (Supervised Contrastive): aproxima a mesma classe, afasta classes diferentes
  (limpas reais **e** limpas-reflow no mesmo cluster).
- Generalização em lote da ideia âncora/positivo/negativo — **não** Triplet clássica.

<!--
🎤 Fala: "São duas saídas: a projeção, que dá o vetor z (o decisor canônico), e uma cabeça
auxiliar multi-classe — clean + 6 categorias — de onde o gate lê P(erro)=1−P(clean); ela é
diagnóstico, não o decisor. A perda combina a SupCon, que organiza o espaço, com um termo CE
da auxiliar (peso 0.6). As classes auxiliares também regularizam: gate binário-puro satura o
sintético e não transfere. SupCon é a versão em lote de âncora/positivo/negativo."
🛡️ Se perguntarem 'usaram Triplet?': não a clássica; SupCon, mais estável com pouco dado.
-->

---

## ⚖️ Etapa 5 — Decisão em DOIS estágios

Com o espaço organizado, a decisão é **geométrica** — a mesma ideia (distância a protótipos no espaço `z`) para os dois estágios:

**Estágio 1 — gate "tem erro?"** *(decisor canônico = protótipo)*
1. **Protótipo** do cluster limpo = o **centro do que é normal** (k-means).
2. **Distância cosseno:** `score = 1 − cos(z, protótipo)`. *(perto = saudável · longe = suspeita)*
3. Vira **probabilidade** `p(erro)` ∈ [0,1] — **calibrada na VALIDAÇÃO LIVRE DE CONFOUND**.
4. **Limiar** (specificity-first): `p(erro) > limiar` → ❌ erro ; senão → ✅ limpa.

**Estágio 2 — categoria** *(só se o gate = erro)*
- **Um** método: o **protótipo de categoria** mais próximo — **mesma matemática `1 − cos`**.
- Taxonomia primária = **3 super-classes**: região morta · deslocado · geometria.

> 💡 *"Mede a distância da tela ao centro do normal (gate); se suspeita, vê de que bairro-de-problema está mais perto (categoria)."*

<!--
🎤 Fala: "A decisão é geométrica e tem DOIS estágios, com a mesma ideia: distância a
protótipos. Estágio 1, o gate: resumimos as limpas num protótipo, medimos a distância cosseno,
viramos probabilidade — agora CALIBRADA na validação livre de confound, não em 26 limpas — e
cortamos num limiar specificity-first. Estágio 2, só se o gate diz erro: a mesma matemática,
mas vendo de qual protótipo de CATEGORIA a tela está mais perto. Consolidamos para UM método
(antes eram dois em paralelo) e taxonomia grossa de 3 super-classes."
-->

---

## 🔎 O corte / limiar — as 4 respostas diretas

<br>

| Pergunta | Resposta |
|---|---|
| **É por uma distância?** | **Sim** — distância cosseno ao protótipo do "normal" (decisor canônico). |
| **Qual o cálculo?** | `score = 1 − cos(z, protótipo)` → `p(erro) = sigmoid(...)` → corte em `p(erro)`. |
| **É algoritmo de ML?** | **Sim, três:** k-means (protótipo) + **fusão logística** (calibra) + escolha do limiar. |
| **Como escolhem o limiar?** | **Na validação LIVRE DE CONFOUND** (limpas + sintéticos + reflow), specificity-first — nunca no teste. |

<!--
🎤 Fala (a pergunta que a equipe fez): "É por distância? Sim, o núcleo é a distância cosseno
ao protótipo — o decisor canônico. Não corto a distância direto: ela vira uma probabilidade
calibrada. Tem ML: k-means pro protótipo, fusão logística pra calibrar, e o limiar é escolhido
na validação LIVRE DE CONFOUND (não em 26 limpas), nunca no teste. Não é número chutado."
-->

---

## 🔎 O corte / limiar — a cadeia completa

```text
            z (vetor 128-d da tela, na hiperesfera)
                         │
   ┌─────────────────────┴─────────────────────┐
   ▼                                            ▼
(A) DISTÂNCIA ao protótipo limpo          (B) CABEÇA AUXILIAR (7 classes)
    score = 1 − cos(z, protótipo)             P(erro) = 1 − P(clean)
    ← DECISOR CANÔNICO                        (diagnóstico)
   └─────────────────────┬─────────────────────┘
                         ▼
   (C) FUSÃO  calibrada na VALIDAÇÃO LIVRE DE CONFOUND
       (limpas + sintéticos-erro + reflow — NÃO em 26 limpas)
       p(erro) = sigmoid( c₀·score + c₁·P(erro)_aux + b₀ )   ∈ [0,1]
                         ▼
   (D) LIMIAR (specificity-first) → "ERRO" se p(erro) > limiar, senão "LIMPA"
```

**Por que 2 entradas?** A distância sozinha é detector de *novidade* (dispara em app novo); a
auxiliar segura isso. **Honestidade:** a fusão foi calibrada para **não** explorar a resolução,
então sua AUROC **global cai de propósito** (≈0.68) → lidere pelo **protótipo** (A).

<!--
🎤 Fala: "A distância (A, o decisor canônico) e a cabeça auxiliar multi-classe (B) entram numa
fusão logística (C), CALIBRADA na validação livre de confound, que devolve p(erro); em cima
vem o limiar (D). Honestidade: a fusão foi calibrada pra NÃO usar o atalho de resolução, então
a AUROC global da fusão cai de propósito — por isso lidero pelo protótipo, não pela fusão."
-->

---

## 🔎 Fidelidade ao diagrama (importante)

<br>

O **slide executivo simplifica** a etapa 5: mostra só o caminho do protótipo.

> A caixa **"Cálculo da Probabilidade de Erro"** = a **fusão (C)** — é ali que a distância
> vira `p(erro)`, via **regressão logística**, incorporando o sinal auxiliar.

<br>

- Se perguntarem *"como a distância vira probabilidade?"* → **é a regressão logística da fusão.**
- O diagrama **detalhado** (`pipeline.mmd`) mostra as duas entradas explicitamente.

<!--
🎤 Fala: "Um aviso honesto: esse slide executivo resume. A caixa 'Cálculo da Probabilidade de
Erro' é exatamente essa fusão por regressão logística. Se alguém da equipe pedir o detalhe, é
isso, e o diagrama detalhado mostra as duas entradas."
-->

---

## ✅ / ❌ Resultado — held-out honesto (130 imgs · teste trancado · 1×)

| Estágio 1 — gate "tem erro?" | Valor | Leitura |
|---|---|---|
| **Detecção LIVRE DE CONFOUND** (sintético) | **AUROC 0.72 · AP 0.89** | a prova honesta |
| **AUROC protótipo** (decisor canônico) | **0.73** | sinal mais limpo · controlado 0.71 |
| **Ponto de operação** (calib. livre de confound) | especificidade **0.27** · bAcc **0.57** · F1 **0.79** | o conserto mais forte |

| Estágio 2 — categoria (condicional ao gate) | F1-macro |
|---|---|
| **Grossa (3 super-classes)** ⭐ | **0.62** (IC95 0.38–0.76) |
| Fina (6 classes) — secundária | 0.36 |

> **Lidere pelo livre de confound, nunca pela global** (a regra de resolução sozinha dá AUROC 0.99).
> Confound **atenuado, não vencido** — o teto do gate é de **DADO** (telas limpas diversas).

<!--
🎤 Fala: "O que sai: estágio 1 dá veredito; estágio 2, a categoria. NÃO lidero pela global
(que é ~98% confound). A detecção livre de confound dá AUROC 0.72, AP 0.89, e o protótipo 0.73.
O conserto mais forte foi o ponto de operação: a especificidade saltou pra 0.27. No estágio 2,
taxonomia grossa F1-macro 0.62, sempre com o IC. E sou honesto: o confound foi atenuado, não
vencido — vencer depende de DADO, não de tuning."
-->

---

## ⭐ O conserto principal — calibração do ponto de operação

> A regra trivial **só de resolução** dá **AUROC 0.99** — **mas é trapaça** (detecta o aparelho).
> Nosso modelo **evita esse atalho de propósito**; lidere pelo livre de confound + ponto de operação.

O limiar antigo, fixado em **26 telas limpas**, **inundava de falso-alarme**. Calibrando na
**validação livre de confound**, a especificidade **dobra** sem perder F1 (held-out):

| Ponto de operação (TEST) | especificidade | FPR | bAcc | F1 |
|---|---|---|---|---|
| Antigo (calib. 26 limpas) | 0.12 | 0.88 | 0.54 | 0.82 |
| **Novo (calib. livre de confound)** | **0.27** | **0.73** | **0.57** | 0.79 |

> **Prova (mesmo modelo):** a calibração legada dá especificidade **0.00** (sinaliza TODA limpa
> como erro) vs **0.27** da nova. **Não é o modelo, é a calibração** — e nunca toca o teste.

<!--
🎤 Fala: "Aviso de comparação: se um concorrente mostrar 98%, cheque se não está só explorando
o confound de resolução — sozinho ele dá AUROC 0.99. O conserto mais forte que entregamos é a
calibração do ponto de operação: deixamos de calibrar em 26 limpas e passamos a calibrar na
validação LIVRE de confound. A especificidade dobrou, de 0.12 pra 0.27, sem perder F1. A prova
de que é a calibração e não o modelo: no MESMO modelo, a calibração legada dá especificidade
zero — sinaliza toda tela limpa como erro. E nada disso toca o teste."
-->

---

<!-- _class: lead -->

## Mensagem-chave

<div class="big">

O sistema vira a tela em números (DINOv2 **congelado**), uma **cabeça siamesa** organiza o
espaço (limpo + reflow juntos, erros fora), e a decisão em **dois estágios** é a **distância a
protótipos** — gate "tem erro?" → categoria — com o limiar **calibrado na validação livre de confound**.

</div>

<br>

**Todo o resto existe para que ele aprenda o _erro_, não o _aparelho_.**
O confound foi **atenuado, não vencido**: o ganho forte é o **ponto de operação** (especificidade 0.27) + a **clareza** (1 método por estágio), não o AUROC do gate.

<span class="muted">Maior alavanca de melhoria: **dados** — telas limpas de outros devices/resoluções/fotos (o teto do gate é de DADO).</span>

<!--
🎤 Fala (fechamento): "Resumindo numa frase: extrai com DINOv2 congelado, a cabeça siamesa
organiza o espaço, e a decisão é a distância ao centro do normal, calibrada e cortada por um
limiar da validação. Tudo serve pra aprender o erro e não o aparelho. E a maior alavanca
daqui pra frente não é arquitetura — é coletar telas limpas mais diversas."
-->

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Apêndice — Q&A

<span class="muted">Slides de backup para as perguntas mais prováveis</span>

<!-- 🎤 Use estes slides só se as perguntas surgirem. -->

---

## Q&A — Extração & Rede Siamesa

**Por que DINOv2 e não treinar uma CNN do zero?**
Auto-supervisionado em 142M de imagens; com poucas centenas de imagens reais nada treinado do zero competiria.

**Por que média + desvio dos patches, não só o CLS?**
Erros espaciais são anomalias de homogeneidade; o desvio cai em áreas uniformes — dá a pista
espacial que o CLS sozinho não carrega (detecção livre de confound AUROC 0.72 / AP 0.89).

**Por que não comparar contra uma tela boa de referência?**
A classe limpa é diversa (apps diferentes) → falso-positivo estrutural. Por isso **protótipos**.

**Por que padding cinza, não esticar/cortar?**
Esticar distorce o erro; cortar perde topo/laterais. Cinza some na normalização. `pad ≥ resize` (medido).

<!-- 🎤 Fala: respostas curtas; se quiserem número, cito o livre-de-confound 0.72/0.89 e o pad≥resize. -->

---

## Q&A — Decisão, limiar & dados

**O limiar é uma distância fixa?**
Não. O *sinal* é distância; o *corte* é sobre `p(erro)`, **calibrado na validação LIVRE DE CONFOUND** (specificity-first).

**Por que protótipos via k-means?**
Resumem o cluster limpo (limpas reais + reflow). Multimodal? basta aumentar o k na config.

**Por que cosseno e não euclidiana?**
Vetores normalizados (hiperesfera) → cosseno mede direção/conteúdo, casa com o treino contrastivo.

**Por que não liderar pela acurácia/AUROC global?**
A global é ~98% trapaça (regra de resolução → AUROC 0.99). Liderar pelo livre de confound (0.72/0.89) + ponto de operação (espec. 0.27).

**O confound foi resolvido?**
**Atenuado, não vencido:** no held-out, prever resolução (0.679) ≈ prever erro (0.681). Vencer depende de **DADO**.

**Estágio 2 melhorou?**
Ficou **claro** (1 método, condicional ao gate). Grossa F1-macro 0.62 (IC95 0.38–0.76); o salto fina→grossa é em grande parte agregação 6→3.

<!-- 🎤 Fala: aqui mora a pergunta do limiar; se aprofundarem, volto à cadeia A→D. Honestidade: confound atenuado, não vencido. -->

---

<!-- _class: lead -->
<!-- _paginate: false -->

## Obrigado

**Roteiro completo:** `docs/ROTEIRO_PIPELINE_EXECUTIVO.md`
**Relatório + números:** `docs/RELATORIO_APRESENTACAO.md`
**Decisões técnicas:** `docs/DESIGN.md`

<!-- 🎤 Fala: "Os detalhes e os números todos estão nesses três documentos. Perguntas?" -->
