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

**Classificação binária:** a tela **tem erro** de layout ou **não tem**?

<span class="muted">Apresentação do pipeline · 5 etapas até a decisão</span>

<!--
🎤 Fala (abertura, 15s): "Esse pipeline recebe o print de uma tela e responde uma
pergunta binária: tem erro de layout ou não? O desafio não foi a rede em si — foi
garantir que ela aprendesse a reconhecer o ERRO, e não características do APARELHO que
vazam nos dados. Vou percorrer as 5 etapas até a decisão final."
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
- As telas **com erro** são variadas (74 resoluções, fotos, dobráveis).
- Uma regra boba — *"resolução ≠ padrão ⇒ erro"* — já acerta **~98%** **sem olhar o layout**.

> Cada etapa adiante força o modelo a olhar o **conteúdo do erro**, não o **aparelho**.

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
| Telas **LIMPAS** reais | 172 | **um único device** · todas 2076×2152 |
| Telas **COM ERRO** reais | 188 | 74 resoluções · fotos · fold/laptop/tent |

<br>

- **A assimetria é o problema central** (tratado na etapa 2).
- Split **agrupado por ticket** (IKSWW): imagens do mesmo bug nunca cruzam treino/teste → **0 vazamento**. *(treino 252 · val 54 · test 54)*

<!--
🎤 Fala: "Duas fontes. Repare na assimetria: as limpas são todas do mesmo aparelho; as com
erro são de tudo quanto é tipo. É daí que vem a armadilha. O split é por ticket pra não
vazar imagens do mesmo bug entre treino e teste."
💡 Analogia: detector de fraude onde toda transação honesta veio do mesmo banco.
-->

---

## 🧪 Etapa 2 — Sintéticos (anti-confound)

> **A etapa mais importante — o diferencial do trabalho.**

Injetamos erros **artificiais** nas próprias telas **limpas**, na **mesma resolução / aparelho**.
→ o par (limpa, corrompida) difere **só pelo erro**; todo o resto fica constante.

| Tipo sintético | Simula |
|---|---|
| `black_region` · `empty_space` | faixa preta · região apagada |
| `overlay` · `disorder` · `cropped` | sobreposição · desalinhamento · corte |

**Resultado medido:** treinar só com erros reais faz o modelo prever a *resolução* tão bem
quanto o erro (trapaça). Sintéticos quebram isso.

<!--
🎤 Fala: "Para tirar a trapaça, pego a tela limpa e estrago ela mesma, mantendo resolução e
device idênticos. Agora a única diferença entre as duas é o erro — o modelo é obrigado a
aprender o erro. São 5 tipos espelhando as categorias reais. E isso foi medido: sem
sintético o modelo aprende a resolução; com sintético ele aprende conteúdo."
💡 Analogia: em vez de comparar maçã de uma fazenda com laranja de outra, estrago a MESMA maçã.
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
🛡️ Se perguntarem por que congelar: 360 imagens + 22M params = overfit imediato; ele decoraria
o confound. O desvio-padrão dos patches captura faixa preta / espaço vazio (regiões uniformes).
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
| Cabeça de **projeção** | produz `z` (128-d) | o coração métrico (siamês) |
| Cabeça **auxiliar** | `Linear(128→1)` | detector binário **direto**, sem depender de referências |

**Função de perda (treino):**

$$\mathcal{L} = \text{SupCon}(z) + 0.3 \cdot \text{BCE}(\text{auxiliar})$$

- **SupCon** (Supervised Contrastive): aproxima a mesma classe, afasta classes diferentes.
- Generalização em lote da ideia âncora/positivo/negativo — **não** Triplet clássica.

<!--
🎤 Fala: "Na verdade são duas saídas: a projeção, que dá o vetor z, e uma cabeça auxiliar,
um classificador binário direto. A perda combina a SupCon — que organiza o espaço — com um
termo da auxiliar. SupCon é a versão em lote da ideia de âncora/positivo/negativo; cada
amostra do batch é âncora, mesma classe é positivo, classe diferente é negativo."
🛡️ Se perguntarem 'usaram Triplet?': não a clássica; SupCon, mais estável com pouco dado.
-->

---

## ⚖️ Etapa 5 — Decisão One-Class

Com o espaço organizado, a decisão é **geométrica**:

1. **Protótipo** do cluster limpo = o **centro do que é normal**.
2. **Distância cosseno** da tela nova ao protótipo: `score = 1 − cos(z, protótipo)`.
   *(perto = saudável · longe = suspeita)*
3. A distância vira uma **probabilidade de erro** `p(erro)` ∈ [0, 1].
4. **Limiar:** `p(erro) > limiar` → ❌ erro ; senão → ✅ limpa.

> 💡 *"Definimos o centro do bairro das telas saudáveis e medimos a que distância a tela nova
> está dele. O limiar é o raio da cerca."*

<!--
🎤 Fala: "A decisão é geométrica. Resumimos as telas limpas num protótipo — o centro do
normal. Para uma tela nova, medimos a distância cosseno até esse centro. Quanto mais longe,
mais provável o erro. Essa distância vira uma probabilidade, e se passar de um limiar,
marcamos erro. O detalhe de como a distância vira probabilidade vem no próximo slide."
-->

---

## 🔎 O corte / limiar — as 4 respostas diretas

<br>

| Pergunta | Resposta |
|---|---|
| **É por uma distância?** | **Sim** — distância cosseno ao protótipo do "normal". |
| **Qual o cálculo?** | `score = 1 − cos(z, protótipo)` → `p(erro) = sigmoid(...)` → corte em `p(erro)`. |
| **É algoritmo de ML?** | **Sim, três:** k-means (protótipo) + **regressão logística** (calibra) + otimização do limiar. |
| **Como escolhem o limiar?** | **Na validação**, por otimização: F1 máximo (padrão) **ou** precisão-alvo (modo alta-precisão). |

<!--
🎤 Fala (a pergunta que a equipe fez): "É por distância? Sim, o núcleo é a distância cosseno
ao protótipo. Mas não corto a distância direto — ela vira uma probabilidade calibrada. Tem
ML: k-means pro protótipo, regressão logística pra calibrar, e o limiar é escolhido por
otimização na validação, nunca no teste. Não é um número chutado."
-->

---

## 🔎 O corte / limiar — a cadeia completa

```text
            z (vetor 128-d da tela, na hiperesfera)
                         │
   ┌─────────────────────┴─────────────────────┐
   ▼                                            ▼
(A) DISTÂNCIA ao protótipo limpo          (B) CABEÇA AUXILIAR
    score = 1 − cos(z, protótipo)             aux = w·z + b
   └─────────────────────┬─────────────────────┘
                         ▼
   (C) FUSÃO  (Regressão Logística, ajustada na VALIDAÇÃO)
       p(erro) = sigmoid( c₀·score + c₁·aux + b₀ )      ∈ [0,1]
                         ▼
   (D) LIMIAR  → "ERRO" se p(erro) > limiar, senão "LIMPA"
```

**Por que 2 entradas?** A distância sozinha é detector de *novidade* (dispara em app novo); a
auxiliar segura isso. **Fundir** pega "longe do normal" **e** "parece erro".

<!--
🎤 Fala: "Aqui está a cadeia inteira. A distância (A) e a cabeça auxiliar (B) entram numa
regressão logística (C) que devolve a probabilidade de erro. Em cima dela vem o limiar (D).
Uso duas entradas porque a distância pura dispararia em qualquer app novo — a auxiliar
corrige isso."
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

## ✅ / ❌ Resultado

<br>

| Métrica (test held-out, 54 imgs) | Valor |
|---|---|
| **Acurácia** | **0.85** (IC 95%: 0.75–0.94) |
| **Precisão / Recall / F1** | **0.86 / 0.86 / 0.86** |
| **AUROC / AP** (livres de limiar) | **0.90 / 0.92** |
| **Detecção sintética livre de confound** *(prova honesta)* | **AUROC 0.88 · AP 0.97** |
| precision@10 (topo do ranking) | **1.00** |

> Números **honestos** — sem usar o atalho da resolução.

<!--
🎤 Fala: "E é isso que sai: para cada tela, uma probabilidade e um veredito. No ponto padrão,
acurácia 0.85, precisão e recall 0.86. E a prova de que detecta erro de verdade, sem o
confound, é o teste sintético: AUROC 0.88."
-->

---

## ⚠️ Comparação justa entre modelos

<br>

- A regra trivial **só de resolução** dá **acurácia ~98%** (AUROC 0.982) — **mas é trapaça**
  (detecta o aparelho).
- Nosso modelo **evita esse atalho de propósito** → marca **0.85 honesto**.

**Dois pontos de operação (mesmo modelo):**

| Operação | Acurácia | Precisão | Recall | Quando usar |
|---|---|---|---|---|
| **Balanceado (padrão)** | **0.85** | 0.86 | 0.86 | comparação / uso geral |
| Alta precisão (opcional) | 0.74 | **1.00** | 0.50 | falso-alarme é caro |

<!--
🎤 Fala: "Aviso para a comparação: se algum modelo concorrente mostrar 98%, vale checar se
ele não está só explorando o confound de resolução — medimos que isso sozinho dá 98%. O
nosso 0.85 é o número honesto. E o '100% de precisão' não é o headline: é um MODO de operação,
pra quando falso-alarme é caro."
-->

---

<!-- _class: lead -->

## Mensagem-chave

<div class="big">

O sistema vira a tela em números (DINOv2 **congelado**), uma **cabeça siamesa** organiza o
espaço (limpo junto, erro fora), e a decisão é a **distância ao centro do normal** —
calibrada em probabilidade e cortada por um **limiar escolhido na validação**.

</div>

<br>

**Todo o resto existe para que ele aprenda o _erro_, não o _aparelho_.**

<span class="muted">Maior alavanca de melhoria: **dados** — telas limpas de outros devices/resoluções/fotos.</span>

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
Auto-supervisionado em 142M de imagens; com 360 imagens nada treinado do zero competiria.

**Por que média + desvio dos patches, não só o CLS?**
Erros espaciais são anomalias de homogeneidade; o desvio cai em áreas uniformes.
Ganho medido: detecção sintética **0.71 → 0.88**.

**Por que não comparar contra uma tela boa de referência?**
A classe limpa é diversa (apps diferentes) → falso-positivo estrutural. Por isso **protótipos**.

**Por que padding cinza, não esticar/cortar?**
Esticar distorce o erro; cortar perde topo/laterais. Cinza some na normalização. `pad ≥ resize` (medido).

<!-- 🎤 Fala: respostas curtas; se quiserem número, cito o 0.71→0.88 e o pad≥resize. -->

---

## Q&A — Decisão, limiar & dados

**O limiar é uma distância fixa?**
Não. O *sinal* é distância; o *corte* é sobre `p(erro)` e seu valor é **otimizado na validação**.

**Por que k=1 protótipo?**
O conjunto limpo é unimodal aqui. O código suporta **k-means com k>1** (multimodal) por config.

**Por que cosseno e não euclidiana?**
Vetores normalizados (hiperesfera) → cosseno mede direção/conteúdo, casa com o treino contrastivo.

**Por que a acurácia é "só" 0.85 se a regra boba dá 98%?**
98% é trapaça (detecta device). 0.85 é honesto; prova livre de confound: AUROC 0.88.

**Maior alavanca de melhoria?**
**Dados**, não arquitetura — telas limpas diversas (outros devices/resoluções/fotos).

<!-- 🎤 Fala: aqui mora a pergunta do limiar; se aprofundarem, volto à cadeia A→D. -->

---

<!-- _class: lead -->
<!-- _paginate: false -->

## Obrigado

**Roteiro completo:** `docs/ROTEIRO_PIPELINE_EXECUTIVO.md`
**Relatório + números:** `docs/RELATORIO_APRESENTACAO.md`
**Decisões técnicas:** `docs/DESIGN.md`

<!-- 🎤 Fala: "Os detalhes e os números todos estão nesses três documentos. Perguntas?" -->
