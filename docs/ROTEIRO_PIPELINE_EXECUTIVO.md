# Roteiro de Apresentação — Pipeline Executivo

### Detector de Erros de Layout em UI · Rede Siamesa sobre DINOv2

> ⚠️ **Esta versão (jun/2026) corrige números REVOGADOS.** Versões antigas deste roteiro
> traziam acc 0.85 / precisão-recall 0.86 / AUROC 0.90 / "test 54 imgs" / "treino 252 · val 54
> · test 54" — **inválidos**: vinham de *split com vazamento* + *seleção que enxergava o teste*
> (auditoria Fase 0, agora REVOGADOS). Os números abaixo são o **held-out honesto** (teste
> trancado, seleção/calibração só na validação, avaliado **1×**) e já refletem a consolidação
> de jun/2026 (reflow + calibração livre de confound + Estágio 2 consolidado). Fonte da verdade:
> [`RELATORIO_APRESENTACAO.md`](RELATORIO_APRESENTACAO.md).

> Este é o **roteiro para apresentar o diagrama** `docs/pipeline-executivo.png`
> (fonte: [`pipeline_executivo.mmd`](pipeline_executivo.mmd)). Ele te dá: a **mensagem
> central**, uma **metodologia** de como conduzir, a **fala etapa-a-etapa** (as 5 etapas
> até a decisão — agora em **dois estágios**: gate "tem erro?" → categoria do erro), o
> **aprofundamento do "corte/limiar"** (a pergunta técnica mais provável) e um **banco de
> perguntas e respostas**.
>
> Documentos irmãos (use como referência, não precisa abrir na reunião):
> - [`RELATORIO_APRESENTACAO.md`](RELATORIO_APRESENTACAO.md) — relatório completo + números + diagrama detalhado (**fonte da verdade**).
> - [`DESIGN.md`](DESIGN.md) — decisões técnicas e justificativas medidas.

---

## Índice

1. [A mensagem central (decore isto)](#0-a-mensagem-central-decore-isto)
2. [Metodologia: como conduzir a apresentação](#1-metodologia-como-conduzir-a-apresentação)
3. [Roteiro das 5 etapas (fala a fala)](#2-roteiro-das-5-etapas-fala-a-fala)
4. [O CORTE / LIMIAR em profundidade (a pergunta central)](#3-o-corte--limiar-em-profundidade-a-pergunta-central)
5. [Banco de perguntas e respostas (Q&A)](#4-banco-de-perguntas-e-respostas-qa)
6. [Colinha — números e termos na ponta da língua](#5-colinha--números-e-termos-na-ponta-da-língua)
7. [Versões curtas (5 min e 15 min)](#6-versões-curtas-5-min-e-15-min)

---

## 0. A mensagem central (decore isto)

Tudo no diagrama gira em torno de **uma única história**:

> **"Como impedir que o modelo trapaceie."**

O dataset tem uma armadilha: **toda tela limpa veio de um único aparelho** (resolução
2076×2152), enquanto as telas com erro são variadas. Então uma regra boba — *"a resolução é
diferente da padrão? então tem erro"* — já acerta **~98%** sem olhar o layout. Qualquer
modelo ingênuo aprenderia esse atalho e estaria **detectando o aparelho, não o erro**.

**O pipeline inteiro existe para forçar o modelo a olhar o conteúdo do erro, não o aparelho.**
Se você amarrar cada etapa a essa frase, a apresentação fica coesa e você responde quase
qualquer pergunta voltando a ela.

> 🎯 **Honestidade que precisa estar na sua boca:** o confound **não foi vencido — foi
> atenuado.** O ganho forte desta rodada **não é** o AUROC do gate (que tem teto de **dado**:
> a classe limpa vem de **um device** só), e sim **(a)** a calibração do ponto de operação
> (a especificidade dobrou) e **(b)** a clareza do Estágio 2 (um único método). Lidere por
> métricas **livres de confound**, nunca pela acurácia/AUROC global (que é ~98% trapaça).

A frase de 15 segundos para abrir:

> 🎤 *"Esse pipeline recebe o print de uma tela e decide em **dois estágios**: primeiro um
> **gate** — **tem erro de layout ou não?**; e, quando tem, **de que tipo** é o erro. O
> desafio não foi a rede em si — foi garantir que ela aprendesse a reconhecer o **erro**, e
> não características do **aparelho** que vazam nos dados. Vou percorrer as etapas mostrando
> como cada uma contribui pra isso, até a decisão final."*

---

## 1. Metodologia: como conduzir a apresentação

**Princípio condutor: uma frase por caixa, aprofunda só se perguntarem.** O diagrama é
executivo — ele resume. Sua função é narrar o fluxo, não despejar tudo. Guarde as fórmulas
para o Q&A (Parte 3 e 4), onde elas estão prontas.

**Ordem sugerida (siga a linha grossa do diagrama, de cima para baixo):**

| Momento | O que fazer | Tempo |
|---|---|---|
| 1. Abertura | Diga o objetivo + a "grande sacada" (o confound). Use a frase de 15s acima. | 1 min |
| 2. As 5 etapas | Percorra **Etapa 1 → 5** seguindo as setas grossas (o caminho de uma tela até a decisão). Para cada caixa: **o que faz** (1 frase) → **por que** → **analogia**. | 5–7 min |
| 3. A decisão | Feche nos dois estágios (gate ✅/❌ → categoria) com o **número honesto** e o **ponto de operação calibrado** (especificidade 0.27, bAcc 0.57). | 1 min |
| 4. Q&A | Deixe o aprofundamento técnico (limiar, fórmulas) para aqui. Você tem tudo na Parte 3 e 4. | livre |

**Três técnicas que ajudam:**

- **Aponte fisicamente para a caixa** enquanto fala dela. O diagrama já tem cores: azul =
  dados, laranja = sintéticos (anti-trapaça), verde-água = extração, verde = rede siamesa
  (a parte que **aprende**), roxo = decisão.
- **Separe "treino" de "uso real".** Etapas 1 e 2 acontecem **uma vez**, na preparação.
  Etapas 3→5 são o que roda **toda vez que chega uma tela nova**. Diga isso explicitamente,
  evita confusão.
- **Quando travar numa pergunta difícil, volte à mensagem central:** *"isso conecta com a
  ideia de não deixar o modelo trapacear pelo aparelho…"* — e siga para a resposta.

**O que NÃO fazer:** **não lidere por acurácia/AUROC global** (neste dataset isso é ~98%
confound — a regra trivial de resolução sozinha dá AUROC 0.99). Não prometa "alta acurácia"
como headline; o que defendemos é a **detecção livre de confound** (sintético 0.72, AP 0.89)
e o **ponto de operação calibrado** (especificidade 0.27). Não diga que o confound "foi
vencido" (foi **atenuado**). Não diga que o backbone "foi treinado por nós" (ele é
**congelado**). Não diga que comparamos a tela contra "uma tela boa de referência"
(comparamos contra **protótipos** — ver Etapa 5).

---

## 2. Roteiro das 5 etapas (fala a fala)

> Cada etapa abaixo tem: 🎤 **fala sugerida** (o que dizer apontando a caixa), 💡 **analogia**
> (para os menos técnicos) e 🛡️ **se perguntarem** (defesa rápida). O número da etapa bate
> com o número no diagrama.

### Etapa 1 — Fontes de Dados (caixa azul)

🎤 *"Temos duas fontes. De um lado, **telas limpas reais** — mas atenção: todas vêm de **um
único dispositivo**, mesma resolução. Do outro, **telas com erro reais**, essas sim de
vários aparelhos, fotos, dobráveis. Essa assimetria é o problema central que vamos tratar
na etapa 2."*

💡 *Analogia:* "É como treinar um detector de fraude onde, por acaso, todas as transações
honestas vieram do mesmo banco. O modelo pode aprender a reconhecer **o banco**, não a
fraude."

🛡️ *Se perguntarem "quantas imagens?"*: **541 imagens reais únicas** (172 limpas + 369 com
erro, em 6 categorias), divididas em treino/validação/teste **agrupadas por ticket + sessão**
e **estratificadas por categoria** — imagens do mesmo bug/sessão nunca aparecem em dois
conjuntos (**0 vazamento**, verificado em `tests/test_split_isolation.py`). O split é **train
330** (105 limpas + 225 erros) · **val 81** (26+55) · **test 130** (41+89). No treino ainda
somamos augmentação que **não conta como dado real**: +420 sintéticos-erro + +420
limpas-reflow (anti-confound dos dois lados).

---

### Etapa 2 — Geração de Sintéticos (caixa laranja · *anti-confound*)

> **Esta é a etapa que mais importa.** É o diferencial do trabalho. Gaste mais tempo aqui.

🎤 *"Para impedir a trapaça, atacamos o confound pelos **dois lados**. Pelo lado do **erro**:
pegamos as **telas limpas** e **injetamos erros artificiais nelas mesmas**, mantendo
**exatamente a mesma resolução e aparelho** — o par (limpa, corrompida) difere **só pelo
erro**. E pelo lado do **limpo** (o **reflow** — a novidade desta rodada): geramos variantes
**limpas** de layout legítimo (scroll, dual-pane, outro aspect-ratio, espaçamento), algumas
em **outras resoluções**, todas rotuladas como **limpas**. Assim a classe limpa deixa de ser
exclusivamente 2076×2152, e o atalho da resolução quebra dos dois lados."*

💡 *Analogia:* "Em vez de comparar maçãs de uma fazenda com laranjas de outra, eu pego a
mesma maçã e estrago um pedaço (a única diferença é o **estrago**). E mostro a mesma maçã sã
em vários ângulos e tamanhos, dizendo 'isso ainda é uma maçã boa' — pra ele não confundir
'tamanho/ângulo diferente' com 'estragada'."

São **5 tipos** de erro sintético, espelhando as categorias reais:

| Tipo | O que simula |
|---|---|
| `black_region` | faixa preta nas laterais/topo (dobrável não expandido) |
| `empty_space` | região grande apagada com a cor de fundo |
| `overlay` | um pedaço da tela colado sobre outra região (sobreposição) |
| `disorder` | blocos deslocados/desalinhados (layout quebrado) |
| `cropped` | conteúdo cortado deixando faixa vazia |

E o **reflow** (lado limpo, NOVO): scroll · dual-pane · outro aspect-ratio · espaçamento —
variantes **limpas** que ensinam "mesmo conteúdo, layout diferente = ainda limpo".

🛡️ *Se perguntarem "e isso funciona mesmo?"*: foi **medido — e a resposta é honesta**. O
reflow **reduz** o quanto o modelo rastreia o confound (sonda de falseabilidade: nível
absoluto de "prever resolução" cai de 0.72 → **0.68** no held-out; e na **validação** o reflow
abre o gap a favor da detecção: prever resolução **0.62** < prever erro **0.65**). **Mas no
held-out o gap fica ~0** (prever resolução 0.679 ≈ prever erro 0.681): o confound **não foi
vencido, foi atenuado** — vencê-lo depende de **dado** (telas limpas diversas), não de tuning.
Ablação completa no `RELATORIO_APRESENTACAO.md`, §6.6.

---

### Etapa 3 — Extração de Características (caixa verde-água)

> Aqui é onde a equipe vai perguntar "**como vocês extraem as características?**". Resposta em
> três partes: pré-processamento → backbone congelado → o vetor de saída.

🎤 *"A imagem passa por um **pré-processamento sem distorção** e entra num **backbone
DINOv2** — um modelo de visão pré-treinado em 142 milhões de imagens. Ele está **congelado**:
não treina, não aprende nada do nosso problema. Ele é só um **extrator de características**
muito bom, que transforma a imagem num **vetor numérico** que resume o que há nela."*

💡 *Analogia:* "O DINOv2 é um **olho especialista** já formado. Não reeducamos o olho — seria
caro e, com poucas centenas de imagens reais, ele decoraria os defeitos do nosso dataset
(inclusive o confound). Usamos o olho como está e ensinamos só uma **régua leve** em cima do
que ele enxerga (isso é a etapa 4)."

**Os três pontos técnicos (tenha na ponta da língua):**

1. **Pré-processamento — "Pad/Resize sem distorção".** A imagem é colocada num quadrado com
   **padding cinza neutro** (a média do ImageNet, que "some" após a normalização) e só então
   redimensionada para 518×518. Isso **preserva a geometria do erro** — uma faixa preta não
   é espremida. (Cinza, nunca preto: preto imitaria o erro "black region".)
2. **Backbone DINOv2 ViT-S/14 — congelado.** Divide a imagem numa grade de **37×37 patches**
   e produz vetores. 22 milhões de parâmetros, **zero treináveis**.
3. **O vetor de saída (1152 dimensões).** Concatenamos: o **token CLS** (resumo global,
   384-d) + a **média** e o **desvio-padrão** dos patches de conteúdo (384 + 384). O
   desvio-padrão é o que captura erros espaciais: uma região grande e uniforme (faixa preta,
   espaço vazio) **derruba o desvio** dos patches — pista direta de anomalia.

> 💬 **Por que congelado + cache:** como o DINOv2 não muda, o vetor de cada imagem é **fixo**.
> Calculamos uma vez e guardamos em disco (cache `.npz`). Resultado: treinar a parte que
> aprende leva **segundos**, não horas.

🛡️ *Se perguntarem "por que não treinar o DINOv2 (fine-tuning)?"*: com poucas centenas de
imagens reais, ajustar 22 milhões de parâmetros causaria **overfitting** imediato — ele
decoraria justamente os confounds (resolução, device). Congelar é a decisão correta para
pouco dado.

---

### Etapa 4 — Rede Siamesa / Aprendizado Métrico (caixa verde)

> Aqui vem "**como funciona a rede siamesa?**" e "**o que é a cabeça de aprendizado?**".

🎤 *"O vetor do DINOv2 entra na **cabeça de projeção** — e essa é a **única parte que
realmente aprende** no sistema (cerca de 330 mil parâmetros). Ela reescreve o vetor num
espaço novo, de 128 dimensões, **desenhado para separar limpo de errado**: telas limpas
viram um aglomerado compacto, telas com erro caem para fora. O resultado é um ponto `z` na
superfície de uma esfera — a 'hiperesfera' do diagrama."*

💡 *Analogia:* "O DINOv2 te dá as coordenadas brutas de cada tela. A cabeça é um **tradutor**
que reorganiza o mapa para que todas as telas saudáveis fiquem **no mesmo bairro**, e as com
problema, longe dele."

**O que faz isso ser "siamesa"** (explicação curta e correta):

> Uma rede siamesa é a **mesma função, com os mesmos pesos, aplicada a qualquer entrada** —
> de modo que comparar duas telas vira **comparar os dois vetores `z`**. É exatamente o que
> temos: a mesma cabeça `g(·)` projeta toda imagem; a comparação acontece no espaço `z`.

**Ponto importante — por que NÃO comparamos contra "uma tela boa de referência":** a classe
"limpa" é **diversa** (apps diferentes, cores e idiomas diferentes). Duas telas limpas de
apps distintos são legitimamente diferentes — comparar contra **uma** referência marcaria
"tela diferente" como "tela errada" (falso-positivo). Por isso comparamos contra
**protótipos do conjunto limpo** (etapa 5). Tecnicamente: é uma **siamesa one-class / de
aprendizado métrico**.

**A cabeça de aprendizado, em detalhe** (são na verdade **duas** saídas):

| Saída | O que é | Para que serve |
|---|---|---|
| **Cabeça de projeção `g(·)`** | `LayerNorm → Linear(1152→256) → GELU → Dropout → Linear(256→128) → normaliza` | produz o `z` métrico; é o coração siamês |
| **Cabeça auxiliar** | `Linear(128→7)` (softmax: clean + 6 categorias) | classificador multi-classe **direto** sobre `z`; o gate lê `P(erro) = 1 − P(clean)`. Fica como **diagnóstico**, não como decisor canônico |

> O treino usa a **Supervised Contrastive Loss** (aproxima telas da mesma classe, afasta as
> de classes diferentes — limpas reais **e** limpas-reflow caem no mesmo cluster) somada a um
> termo da cabeça auxiliar:
> **`Perda = SupCon(z) + 0.6 × CE(auxiliar de 7 classes)`**.

🛡️ *Se perguntarem "usaram Triplet Loss / âncora-positivo-negativo?"*: Não a Triplet clássica.
Usamos **SupCon**, que é a generalização em lote: dentro do batch, **cada amostra é âncora**,
as da mesma classe são positivos e as de outra classe são negativos. Mais estável com pouco
dado. (O conceito de âncora reaparece na inferência: a **tela-alvo é a âncora** comparada aos
protótipos.)

---

### Etapa 5 — Decisão em DOIS estágios (caixa roxa)

> Aqui vem a pergunta mais técnica: **"como, em cima da projeção, é feito o corte? É
> distância? Qual o cálculo? É algum algoritmo de ML?"**. Esta seção dá a resposta no nível
> executivo; a Parte 3 dá o aprofundamento completo com fórmulas.

🎤 *"Com o espaço já organizado, a decisão é geométrica e tem **dois estágios**. **Estágio 1
(o gate, 'tem erro?')**: resumimos o aglomerado das telas limpas num **protótipo** — o
'centro do que é normal' — e medimos a **distância** entre o `z` da tela nova e esse
protótipo. Quanto mais longe do normal, mais provável o erro; viramos isso numa
**probabilidade** e cortamos num **limiar**. **Estágio 2 (a categoria)**, que só roda quando
o gate diz 'erro': a mesma matemática, mas agora vendo de qual **protótipo de categoria** a
tela está mais perto."*

💡 *Analogia:* "Definimos o 'centro do bairro das telas saudáveis'. Quando chega uma tela
nova, medimos **a que distância ela está desse centro** (estágio 1). Se for suspeita,
perguntamos **de que bairro-de-problema ela está mais perto** (estágio 2). O limiar é o
**raio da cerca** que separa saudável de suspeita."

**As peças do Estágio 1 — gate (na ordem do diagrama):**

1. **Protótipo do cluster limpo.** O centro das telas limpas no espaço `z`. (Cálculo:
   k-means sobre os `z` limpos de treino, recolocados na esfera. Limpas reais **e**
   limpas-reflow caem no mesmo cluster.)
2. **Distância cosseno.** `distância = 1 − cosseno(z_da_tela, protótipo)`. Zero = idêntico ao
   normal; quanto maior, mais anômalo. **É literalmente uma distância** — responde "sim" à
   pergunta da equipe. **Este é o decisor canônico do gate.**
3. **Cálculo da probabilidade de erro.** A distância é convertida numa probabilidade
   calibrada `p(erro)` ∈ [0, 1]. *(Dentro dessa caixa há uma fusão que também incorpora a
   cabeça auxiliar — ver Parte 3.)* **A calibração agora é feita na VALIDAÇÃO LIVRE DE
   CONFOUND** (limpas + sintéticos-erro + reflow), não em 26 telas limpas — esse é o conserto
   mais forte da rodada (ver Parte 3.5).
4. **Limiar.** Decisão: **`p(erro) > limiar`** → erro; senão → limpa. O limiar **não é
   chutado**: é escolhido na **validação livre de confound** (specificity-first / alta
   precisão) — ver Parte 3.

**O Estágio 2 — categoria (só roda se o gate = ERRO):**

- **UM método canônico** (decisão consolidada nesta rodada): a categoria é a do **protótipo
  de categoria mais próximo** em `z` — **a mesma matemática `1 − cos` do Estágio 1**. A cabeça
  auxiliar fica como diagnóstico, **não** como segundo decisor. *(O desenho antigo reportava
  dois métodos em paralelo — protótipo e softmax — e era a fonte da confusão.)*
- **Taxonomia primária = 3 super-classes** (agrupadas por não-colisão): **região morta**
  (black bars + empty space) · **deslocado** (overlay + disordered) · **geometria**
  (distortion + orientation). A taxonomia fina de 6 classes é **secundária/exploratória**.

### O desfecho — gate ✅/❌ → categoria

🎤 *"E é isso que sai do pipeline: para cada tela, **estágio 1** dá uma probabilidade de erro
e um veredito; quando há erro, o **estágio 2** dá a categoria. Lidero pelo número **honesto**,
não pelo global (que é ~98% confound): no gate, a detecção **livre de confound** dá **AUROC
0.72 (AP 0.89)**, e o protótipo no held-out dá **AUROC 0.73**. O conserto mais forte foi a
**calibração do ponto de operação**: a especificidade saltou para **0.27** (bAcc 0.57, F1
0.79) — o limiar antigo, fixado em 26 telas limpas, inundava de falso-alarme. No estágio 2, a
taxonomia grossa (3 super-classes) dá **F1-macro 0.62 (IC95 0.38–0.76)**."*

---

## 3. O CORTE / LIMIAR em profundidade (a pergunta central)

> A equipe perguntou, literalmente: *"como é definido em cima da projeção qual o corte é
> feito para definir esse limiar? É por uma distância? Qual o cálculo? É algum algoritmo de
> ML?"*. Aqui está tudo, em camadas — do resumo de 30 segundos às fórmulas.

### 3.1 As respostas diretas (decore estas 4 frases)

| Pergunta da equipe | Resposta direta |
|---|---|
| **É por uma distância?** | **Sim, o núcleo (decisor canônico) é uma distância** — a distância cosseno entre o vetor da tela e o protótipo do "normal". A fusão calibrada combina essa distância com um segundo sinal (a cabeça auxiliar), mas **o protótipo é quem lidera**. |
| **Qual o cálculo?** | Três passos: (1) `score = 1 − cos(z, protótipo)`; (2) `p(erro) = sigmoid` de uma combinação de `score` + sinal da cabeça auxiliar (`P(erro)=1−P(clean)`), **calibrada na validação livre de confound**; (3) decide erro se `p(erro) > limiar`. |
| **É algum algoritmo de ML?** | **Sim, três peças clássicas empilhadas:** k-means (acha o protótipo), uma **fusão logística** (funde os sinais e calibra em probabilidade) e uma **escolha do limiar** na validação. E quem aprende o espaço onde a distância faz sentido é a rede siamesa. |
| **Como o limiar é escolhido?** | Na **validação LIVRE DE CONFOUND** (limpas + sintéticos-erro + reflow), nunca no teste — perfil **specificity-first / alta precisão**. O conserto-chave da rodada foi justamente **deixar de calibrar em 26 telas limpas** (instável → falso-alarme). |

### 3.2 A cadeia completa: do `z` à decisão (4 camadas)

```
            z (vetor 128-d da tela, na hiperesfera)
                         │
   ┌─────────────────────┴─────────────────────┐
   ▼                                            ▼
(A) DISTÂNCIA ao protótipo limpo          (B) CABEÇA AUXILIAR (7 classes)
    score = 1 − cos(z, protótipo)             P(erro) = 1 − P(clean)
   "longe do normal?"  ← DECISOR CANÔNICO     "parece erro?"  (diagnóstico)
   └─────────────────────┬─────────────────────┘
                         ▼
   (C) FUSÃO CALIBRADA na VALIDAÇÃO LIVRE DE CONFOUND
       (limpas + sintéticos-erro + reflow — NÃO em 26 limpas)
       p(erro) = sigmoid( c₀·score + c₁·P(erro)_aux + b₀ )   ∈ [0, 1]
                         ▼
   (D) LIMIAR  (escolhido na validação livre de confound · specificity-first)
       decisão = "ERRO" se p(erro) > limiar, senão "LIMPA"
```

**Por que duas entradas (A e B) e não só a distância?** A distância ao protótipo sozinha é um
**detector de novidade** — ele dispararia em qualquer app visualmente novo, mesmo limpo. A
cabeça auxiliar (B), multi-classe (`P(erro)=1−P(clean)`), é um detector direto que não depende
do banco de protótipos. **Fundir as duas** equilibra. **Atenção (honestidade):** a fusão foi
calibrada para **não** explorar o atalho de resolução, então a AUROC *global* da fusão **cai
de propósito** (≈0.68); por isso o **decisor canônico é o protótipo** (A), que **melhora** nas
métricas livres de confound. Lidere pelo protótipo, não pela fusão global.

> ⚠️ **Fidelidade ao diagrama executivo:** o slide mostra só o caminho do protótipo
> (A → "Cálculo da Probabilidade de Erro" → limiar) para não poluir. A caixa **"Cálculo da
> Probabilidade de Erro" é exatamente a fusão (C)** — é onde a distância vira `p(erro)`. Se
> alguém perguntar "como a distância vira probabilidade?", a resposta é: *"uma regressão
> logística calibra a distância (junto com o sinal auxiliar) numa probabilidade entre 0 e 1"*.
> O diagrama **detalhado** (`pipeline.mmd`, §3-bis do relatório) mostra as duas entradas.

### 3.3 As fórmulas (para o quadro branco, se pedirem)

**(A) Protótipo e distância** — `decision.py`:
```
protótipo  p_j = k-means( z das telas limpas de treino )   # recolocados na esfera
score(z)   = 1 − máx_j cos(z, p_j)          # distância cosseno ao protótipo mais próximo
```

**(C) Fusão (logística de 2 variáveis)** — calibrada na **validação LIVRE DE CONFOUND**
(limpas + sintéticos-erro + reflow):
```
p(erro) = sigmoid( c₀ · score + c₁ · P(erro)_aux + b₀ )   # P(erro)_aux = 1 − P(clean)
```

**(D) Escolha do limiar** — na **validação livre de confound** (specificity-first):
```
limiar escolhido para PRIORIZAR ESPECIFICIDADE / alta precisão,
mantendo recall e F1 razoáveis (o gate de produção não pode inundar de falso-alarme)
```
Decisão: **`p(erro) > limiar`**.[^eps]

[^eps]: Detalhe de implementação: o código compara estritamente `>` (com um epsilon), enquanto
o diagrama escreve `≥` por simplicidade. Sem efeito prático.

### 3.4 "É distância? É ML? Qual o cálculo?" — resposta de 30 segundos

> 🎤 *"O coração é **sim** uma distância: medimos o quão longe o vetor da tela está do
> protótipo das telas limpas, com **distância cosseno** — esse é o decisor canônico. Essa
> distância entra numa **fusão logística** que a calibra numa **probabilidade de erro**. E o
> **corte** não é chutado: escolhemos, **na validação livre de confound** (limpas +
> sintéticos + reflow, nunca no teste), o valor que **prioriza especificidade** sem inundar de
> falso-alarme. Então tem distância, tem ML (k-means pro protótipo, fusão logística pra
> calibrar) e o limiar é fruto de calibração, não de palpite."*

### 3.5 ⭐ O conserto principal — calibrar o ponto de operação na validação LIVRE DE CONFOUND

Esta é a **maior alavanca de operação** entregue na rodada. O limiar/fusão antigo era fixado
em **26 telas limpas** de validação — instável, e **inundava o teste de falso-alarme**.
Calibrando na **validação livre de confound** (limpas + sintéticos-erro + reflow), a
especificidade **dobra** sem perder F1 (held-out, **mesmo modelo**, só muda o conjunto de
calibração):

| Ponto de operação (TEST) | especificidade | FPR | bAcc | F1 |
|---|---|---|---|---|
| Antigo (calib. 26 limpas) | 0.12 | 0.88 | 0.54 | 0.82 |
| **Novo (calib. livre de confound)** | **0.27** | **0.73** | **0.57** | 0.79 |

> **A prova de que a causa é a calibração** (não o modelo): no **mesmo modelo novo**, a
> calibração legada (26 limpas) dá especificidade **0.00** — sinaliza **TODA** tela limpa como
> erro — contra **0.27** da nova. Na **validação** o efeito é ainda maior (especificidade
> 0.00 → 0.77 sem-reflow / 0.46 com-reflow). **Não há vazamento:** a calibração é sempre na
> validação, o teste é medido **1×**. Apresente a **especificidade 0.27 / bAcc 0.57** como o
> ganho de operação — **não** uma "acurácia headline" (a global é ~98% confound).

---

## 4. Banco de perguntas e respostas (Q&A)

> Organizado pelos quatro temas que a equipe sinalizou + armadilhas de dados.

### Extração de características

**P: Como vocês extraem as características da imagem?**
R: Pré-processamento sem distorção (padding cinza → 518×518) → **DINOv2 ViT-S/14 congelado**
→ vetor de **1152 dimensões** = token global (CLS) + média + desvio-padrão dos patches.

**P: Por que DINOv2 e não uma CNN treinada do zero / ResNet?**
R: DINOv2 é auto-supervisionado em 142M de imagens e dá features visuais riquíssimas **sem
rótulos**. Treinar do zero com poucas centenas de imagens reais não competiria. Usamos como
extrator fixo.

**P: Por que congelar o backbone?**
R: 22M de parâmetros + poucas centenas de imagens reais = overfitting garantido; ele decoraria
os confounds. Congelado, ele é estável e os vetores podem ser **cacheados** (treino em segundos).

**P: Por que incluir média e desvio dos patches, e não só o CLS?**
R: Erros espaciais (faixa preta, espaço vazio) são anomalias de **homogeneidade** — o
desvio-padrão dos patches cai em áreas grandes uniformes. As estatísticas de patch (média +
desvio dos tokens de conteúdo) dão à cabeça a pista espacial que o CLS sozinho não carrega; a
detecção livre de confound (sintético) fica em **AUROC 0.72 / AP 0.89** no held-out.

**P: Por que padding cinza e não esticar a imagem (resize) ou cortar (crop)?**
R: Esticar distorce a geometria do erro (faixa preta vira fininha); cortar descartaria topo/
laterais (onde moram vários erros). Padding **cinza** preserva o aspecto e some na
normalização. Medido: `pad` ≥ `resize` em todas as métricas honestas.

### Rede siamesa

**P: O que exatamente é "siamês" aqui?**
R: A **mesma função com os mesmos pesos** (a cabeça `g`) é aplicada a qualquer tela, e a
comparação acontece no espaço dos vetores `z`. Comparar duas telas = comparar `z₁` e `z₂`.

**P: Vocês comparam a tela contra uma tela de referência?**
R: **Não.** A classe limpa é diversa (apps diferentes). Comparamos contra **protótipos** do
conjunto limpo — por isso é uma siamesa **one-class**. Comparar contra uma única referência
geraria falso-positivo em qualquer app novo.

**P: Qual a função de perda? Usaram Triplet?**
R: **Supervised Contrastive (SupCon)** + um termo CE da cabeça auxiliar **multi-classe**
(`Perda = SupCon(z) + 0.6·CE(aux de 7 classes: clean + 6 categorias)`). SupCon é a
generalização em lote da ideia âncora/positivo/negativo, mais estável com pouco dado. Triplet
clássica não.

**P: O que é a "hiperesfera"?**
R: Os vetores `z` são **normalizados** (comprimento 1), então vivem na superfície de uma
esfera em 128 dimensões. Nesse espaço, "distância" vira **ângulo/cosseno** — natural para
comparar direção (conteúdo) ignorando magnitude.

### Cabeça de aprendizado

**P: O que é a "cabeça de projeção" e quantos parâmetros tem?**
R: Uma MLP pequena: `LayerNorm → Linear(1152→256) → GELU → Dropout → Linear(256→128) →
L2-norm`. ~330 mil parâmetros — a **única** parte treinável do sistema.

**P: Por que existe uma cabeça auxiliar além da projeção?**
R: A projeção + protótipo é um detector de **novidade** (dispara em app novo). A auxiliar
(`Linear 128→7`, multi-classe: clean + 6 categorias; o gate lê `P(erro)=1−P(clean)`) é um
classificador **direto** que não depende de referências e serve de **diagnóstico**; o decisor
canônico do gate é o **protótipo**. As classes auxiliares também regularizam (o gate
binário-puro satura o sintético e não transfere). Fundir os dois sinais dá robustez (Parte 3.2).

**P: Quanto tempo leva o treino?**
R: Segundos. O gargalo (DINOv2) roda 1× por imagem e fica em cache; a cabeça é minúscula.
Treino curto com early-stopping pela métrica de validação.

### Decisão / limiar

**P: Como é o corte? (resposta curta)** → ver **Parte 3.4** (resposta de 30s).

**P: O limiar é uma distância fixa?**
R: O **sinal** é uma distância (cosseno ao protótipo), mas o **corte** é feito sobre a
**probabilidade calibrada** `p(erro)`, e seu valor é **escolhido na validação LIVRE DE
CONFOUND** (limpas + sintéticos-erro + reflow), com perfil **specificity-first**. Não é um
número fixo arbitrário — e calibrar aqui (em vez de em 26 limpas) foi o conserto que dobrou a
especificidade.

**P: Por que protótipos via k-means? E se as telas limpas forem variadas?**
R: Resumimos o cluster limpo com **k-means** sobre os `z` das limpas de treino (limpas reais
e limpas-reflow no mesmo cluster). Se houver múltiplos estilos de tela limpa, basta aumentar
o k na config.

**P: Por que a distância cosseno e não euclidiana?**
R: Como os vetores estão normalizados (hiperesfera), cosseno mede **direção/conteúdo**
ignorando escala — é o casamento natural com o treino contrastivo (que também usa cosseno).

**P: A fusão logística não é "outro modelo"? Não é overkill?**
R: É um modelo linear de **2 variáveis** (`score` do protótipo e `P(erro)` da auxiliar),
calibrado na validação livre de confound só para **fundir** os dois sinais numa probabilidade.
É leve; o **decisor canônico continua sendo o protótipo** (a fusão global até cai de propósito
para não explorar o confound).

### Dados, avaliação e armadilhas

**P: Por que vocês não lideram por acurácia/AUROC global? Uma regra boba não dá 98%?**
R: Dá — e por isso **a global é ~98% trapaça**: a regra "resolução ≠ 2076×2152 ⇒ erro" detecta
o **aparelho**, não o erro (AUROC 0.99). Nosso modelo **evita** esse atalho de propósito (a
AUROC global da fusão até **cai**). Liderar pela métrica **livre de confound**: gate AUROC
0.72 / AP 0.89 (sintético), protótipo 0.73, subconjunto controlado 0.71 — e pelo **ponto de
operação calibrado** (especificidade 0.27, bAcc 0.57, F1 0.79).

**P: Então o confound foi resolvido?**
R: **Não — foi atenuado, não vencido.** No held-out, prever resolução (0.679) ≈ prever erro
(0.681): o gap é ~0. O reflow **reduziu** o rastreamento absoluto e na **validação** abriu o
gap a favor da detecção, mas vencer o confound depende de **dado** (telas limpas diversas),
não de tuning. É a honestidade central da apresentação.

**P: E o Estágio 2 (categoria) melhorou?**
R: Ficou **claro**, não necessariamente "melhor de qualidade". Consolidamos para **um único
método** (protótipo de categoria), condicional ao gate. Taxonomia grossa (3 super-classes)
dá F1-macro **0.62 (IC95 0.38–0.76)**; fina (6 classes) **0.36**. Honestidade: o salto fina→
grossa é em grande parte **agregação 6→3** (tarefa mais fácil), e o IC95 tem limite inferior
**perto do acaso** (0.33) — sempre reportar com o IC. O valor é o **desenho claro** (1 método,
vs os 2 paralelos de antes), não o número.

**P: Como sabemos que não há vazamento entre treino e teste?**
R: O split é **agrupado por ticket + sessão** e estratificado por categoria: todas as imagens
de um mesmo bug/sessão ficam no mesmo conjunto. Verificado: **0 grupos cruzando splits**
(`tests/test_split_isolation.py`). O teste é **trancado** programaticamente e medido **1×**.

**P: Qual a maior alavanca para melhorar?**
R: **Dados, não arquitetura.** Coletar telas **limpas** de outros aparelhos/resoluções/fotos.
Enquanto a classe limpa vier de um device só, a métrica global continua dominada por confound
e o AUROC do gate tem teto de **dado**.

---

## 5. Colinha — números e termos na ponta da língua

**Números — held-out honesto (130 imgs · teste trancado · 1×). Decore os 6 primeiros e SEMPRE
lidere pelos livres de confound:**

| | Valor |
|---|---|
| **Gate — detecção livre de confound (prova honesta)** | **AUROC 0.72 · AP 0.89** (sintético) |
| **Gate — AUROC protótipo** (decisor canônico, sinal mais limpo) | **0.73** · subconjunto controlado **0.71** |
| **Ponto de operação (calib. livre de confound)** | especificidade **0.27** (FPR 0.73) · bAcc **0.57** · F1 **0.79** |
| **Prova da calibração** (mesmo modelo) | legada (26 limpas) → especificidade **0.00**; nova → **0.27** |
| **Falseab. (honesto)** | prever resolução **0.679** ≈ prever erro **0.681** (gap ~0 → confound **atenuado**, não vencido) |
| **Estágio 2 — categoria** | grossa (3 super-classes) F1-macro **0.62 (IC95 0.38–0.76)** · fina (6) **0.36** |
| Confound (a trapaça, NÃO usar como nossa métrica) | regra de resolução sozinha → AUROC **0.99** (~98% trapaça) |
| Dataset / split | **541 imgs reais** (172 limpas + 369 erro · 6 cat.) · train **330** (105+225) / val **81** (26+55) / test **130** (41+89) · +420 sint.-erro +420 limpas-reflow no treino |
| Backbone | DINOv2 ViT-S/14 · 22M params · **congelado** · grade 37×37 |
| Embedding / cabeça / `z` | 1152-d → cabeça ~330k params → `z` 128-d · aux `Linear 128→7` |

**Termos (tradução rápida para leigos):**

- **Confound** → "atalho enganoso nos dados" (aqui: a resolução; toda limpa é de 1 device).
- **Atenuado, não vencido** → "reduzimos o atalho, mas ele ainda existe — vencê-lo é questão de DADO".
- **Reflow** → "variantes LIMPAS de layout legítimo (scroll, dual-pane, outro aspecto) que quebram o confound pelo lado limpo".
- **Calibração livre de confound** → "escolher o limiar numa validação que NÃO tem o atalho (em vez de em 26 limpas)".
- **Especificidade** → "das telas limpas, quantas o modelo deixou passar como limpas".
- **Embedding / vetor de características** → "a tela virada em números".
- **Backbone congelado** → "olho pré-treinado que não reeducamos".
- **Hiperesfera** → "a superfície onde os vetores vivem; comparar = medir ângulo".
- **Protótipo** → "o centro do que é normal" (gate) ou "o centro de cada tipo de erro" (categoria).
- **Distância cosseno** → "o quão longe/diferente do normal".
- **Limiar** → "a cerca que separa erro de não-erro".
- **Super-classe (taxonomia grossa)** → "agrupar as 6 categorias em 3: região morta · deslocado · geometria".
- **AUROC** → "nota de 0 a 1 da capacidade de separar as classes, sem depender do corte".
- **Recall** → "dos erros que existem, quantos achamos"; **Precisão** → "do que marcamos como
  erro, quanto era erro de verdade".

---

## 6. Versões curtas (5 min e 15 min)

**Versão 5 minutos (só o caminho principal):**
1. Objetivo (dois estágios: gate → categoria) + a trapaça do confound (frase de abertura). *(45s)*
2. Etapa 2 — sintéticos **+ reflow**: anti-trapaça pelos dois lados. *(1 min)*
3. Etapa 3 — DINOv2 congelado vira a tela em números. *(45s)*
4. Etapa 4 — a cabeça siamesa organiza o espaço (limpo+reflow juntos, erros fora). *(1 min)*
5. Etapa 5 — distância ao protótipo → probabilidade → limiar (gate) → categoria. *(1 min)*
6. Números honestos: detecção livre de confound **AUROC 0.72 (AP 0.89)**; o conserto forte é o
   **ponto de operação** (especificidade 0.27, bAcc 0.57); confound **atenuado, não vencido**. *(30s)*

**Versão 15 minutos:** a de 5 min + abrir a Parte 3 (cadeia A→D do limiar com o quadro
branco) + 2–3 perguntas do Q&A que você achar mais prováveis na sua equipe.

**Se só puder dizer uma frase:**
> *"O sistema transforma a tela num vetor com um modelo de visão congelado, uma cabeça
> siamesa reorganiza esse vetor para que telas limpas formem um aglomerado, e a decisão é a
> distância até o centro desse aglomerado — calibrada em probabilidade e cortada por um
> limiar escolhido na validação. Todo o resto do pipeline existe para que ele aprenda o
> **erro**, e não o **aparelho**."*
