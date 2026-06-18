# Roteiro de Apresentação — Pipeline Executivo

### Detector de Erros de Layout em UI · Rede Siamesa sobre DINOv2

> Este é o **roteiro para apresentar o diagrama** `docs/pipeline-executivo.png`
> (fonte: [`pipeline_executivo.mmd`](pipeline_executivo.mmd)). Ele te dá: a **mensagem
> central**, uma **metodologia** de como conduzir, a **fala etapa-a-etapa** (as 5 etapas
> até a decisão), o **aprofundamento do "corte/limiar"** (a pergunta técnica mais provável)
> e um **banco de perguntas e respostas**.
>
> Documentos irmãos (use como referência, não precisa abrir na reunião):
> - [`RELATORIO_APRESENTACAO.md`](RELATORIO_APRESENTACAO.md) — relatório completo + números + diagrama detalhado.
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

A frase de 15 segundos para abrir:

> 🎤 *"Esse pipeline recebe o print de uma tela e responde uma pergunta binária: **tem erro
> de layout ou não?**. O desafio não foi a rede em si — foi garantir que ela aprendesse a
> reconhecer o **erro**, e não características do **aparelho** que vazam nos dados. Vou
> percorrer as 5 etapas mostrando como cada uma contribui pra isso, até a decisão final."*

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
| 3. A decisão | Feche nas duas telas (✅/❌) com o **número honesto** (acurácia 0.85). | 1 min |
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

**O que NÃO fazer:** não prometa "95% de acurácia" como número único (o número honesto é
**0.85**; o 95% existe só num modo de operação específico — ver Parte 3.5 e 4). Não diga que
o backbone "foi treinado por nós" (ele é **congelado**). Não diga que comparamos a tela
contra "uma tela boa de referência" (comparamos contra **protótipos** — ver Etapa 5).

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

🛡️ *Se perguntarem "quantas imagens?"*: 360 no total (172 limpas + 188 com erro), divididas
em treino/validação/teste **agrupadas por ticket** — imagens do mesmo bug nunca aparecem em
dois conjuntos (zero vazamento).

---

### Etapa 2 — Geração de Sintéticos (caixa laranja · *anti-confound*)

> **Esta é a etapa que mais importa.** É o diferencial do trabalho. Gaste mais tempo aqui.

🎤 *"Para impedir a trapaça, pegamos as **telas limpas** e **injetamos erros artificiais
nelas mesmas**, mantendo **exatamente a mesma resolução e aparelho**. Assim criamos pares
em que a única coisa que muda é o **erro** — todo o resto (resolução, aspecto, device) fica
constante. O modelo é obrigado a aprender o erro, porque é a única pista que sobrou."*

💡 *Analogia:* "Em vez de comparar maçãs de uma fazenda com laranjas de outra, eu pego a
mesma maçã e estrago um pedaço. Agora a diferença entre as duas fotos **só pode ser o
estrago**."

São **5 tipos** de erro sintético, espelhando as categorias reais:

| Tipo | O que simula |
|---|---|
| `black_region` | faixa preta nas laterais/topo (dobrável não expandido) |
| `empty_space` | região grande apagada com a cor de fundo |
| `overlay` | um pedaço da tela colado sobre outra região (sobreposição) |
| `disorder` | blocos deslocados/desalinhados (layout quebrado) |
| `cropped` | conteúdo cortado deixando faixa vazia |

🛡️ *Se perguntarem "e isso funciona mesmo?"*: Sim, e foi **medido**. Treinar só com erros
reais faz o modelo prever a **resolução** tão bem quanto o erro (trapaça, AUROC 0.92 em
prever resolução). Treinar com sintéticos derruba isso e dá a melhor detecção de **conteúdo**
(ablação completa no `RELATORIO_APRESENTACAO.md`, §6.6).

---

### Etapa 3 — Extração de Características (caixa verde-água)

> Aqui é onde a equipe vai perguntar "**como vocês extraem as características?**". Resposta em
> três partes: pré-processamento → backbone congelado → o vetor de saída.

🎤 *"A imagem passa por um **pré-processamento sem distorção** e entra num **backbone
DINOv2** — um modelo de visão pré-treinado em 142 milhões de imagens. Ele está **congelado**:
não treina, não aprende nada do nosso problema. Ele é só um **extrator de características**
muito bom, que transforma a imagem num **vetor numérico** que resume o que há nela."*

💡 *Analogia:* "O DINOv2 é um **olho especialista** já formado. Não reeducamos o olho — seria
caro e, com só 360 imagens, ele decoraria os defeitos do nosso dataset. Usamos o olho como
está e ensinamos só uma **régua leve** em cima do que ele enxerga (isso é a etapa 4)."

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

🛡️ *Se perguntarem "por que não treinar o DINOv2 (fine-tuning)?"*: com ~360 imagens, ajustar
22 milhões de parâmetros causaria **overfitting** imediato — ele decoraria justamente os
confounds (resolução, device). Congelar é a decisão correta para pouco dado.

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
| **Cabeça auxiliar** | `Linear(128→1)` | um detector binário **direto** sobre `z`, que não depende dos protótipos |

> O treino usa a **Supervised Contrastive Loss** (aproxima telas da mesma classe, afasta as
> de classes diferentes) somada a um termo da cabeça auxiliar:
> **`Perda = SupCon(z) + 0.3 × BCE(auxiliar)`**.

🛡️ *Se perguntarem "usaram Triplet Loss / âncora-positivo-negativo?"*: Não a Triplet clássica.
Usamos **SupCon**, que é a generalização em lote: dentro do batch, **cada amostra é âncora**,
as da mesma classe são positivos e as de outra classe são negativos. Mais estável com pouco
dado. (O conceito de âncora reaparece na inferência: a **tela-alvo é a âncora** comparada aos
protótipos.)

---

### Etapa 5 — Decisão One-Class (caixa roxa)

> Aqui vem a pergunta mais técnica: **"como, em cima da projeção, é feito o corte? É
> distância? Qual o cálculo? É algum algoritmo de ML?"**. Esta seção dá a resposta no nível
> executivo; a Parte 3 dá o aprofundamento completo com fórmulas.

🎤 *"Com o espaço já organizado, a decisão é geométrica. Primeiro, resumimos o aglomerado das
telas limpas num **protótipo** — o 'centro do que é normal'. Para uma tela nova, medimos a
**distância** entre o `z` dela e esse protótipo. Quanto mais longe do normal, mais provável
o erro. Essa distância é convertida numa **probabilidade de erro**, e se ela passar de um
**limiar**, marcamos como erro."*

💡 *Analogia:* "Definimos o 'centro do bairro das telas saudáveis'. Quando chega uma tela
nova, medimos **a que distância ela está desse centro**. Perto = saudável; longe = suspeita.
O limiar é o **raio da cerca** que separa os dois."

**As quatro peças da etapa 5 (na ordem do diagrama):**

1. **Protótipo do cluster limpo.** O centro das telas limpas no espaço `z`. (Cálculo: a
   média dos `z` limpos, recolocada na esfera. Se houvesse vários estilos de tela limpa,
   usaríamos k-means com k>1; aqui k=1 basta.)
2. **Distância cosseno.** `distância = 1 − cosseno(z_da_tela, protótipo)`. Zero = idêntico ao
   normal; quanto maior, mais anômalo. **É literalmente uma distância** — responde "sim" à
   pergunta da equipe.
3. **Cálculo da probabilidade de erro.** A distância é convertida numa probabilidade
   calibrada `p(erro)` ∈ [0, 1]. *(Dentro dessa caixa há uma regressão logística que também
   incorpora a cabeça auxiliar — ver Parte 3, é onde a equipe vai querer detalhe.)*
4. **Limiar.** Decisão: **`p(erro) > limiar`** → erro; senão → limpa. O limiar **não é
   chutado**: é escolhido na **validação** para otimizar uma métrica (ver Parte 3).

### O desfecho — ❌ Tela com erro / ✅ Tela limpa

🎤 *"E é isso que sai do pipeline: para cada tela, uma probabilidade de erro e um veredito.
No ponto de operação padrão, o modelo entrega **acurácia 0.85, precisão e recall 0.86**, de
forma **honesta** — sem usar o atalho da resolução. A prova de que ele detecta erro de
verdade está no teste com erros sintéticos, livre do confound: **AUROC 0.88**."*

---

## 3. O CORTE / LIMIAR em profundidade (a pergunta central)

> A equipe perguntou, literalmente: *"como é definido em cima da projeção qual o corte é
> feito para definir esse limiar? É por uma distância? Qual o cálculo? É algum algoritmo de
> ML?"*. Aqui está tudo, em camadas — do resumo de 30 segundos às fórmulas.

### 3.1 As respostas diretas (decore estas 4 frases)

| Pergunta da equipe | Resposta direta |
|---|---|
| **É por uma distância?** | **Sim, o núcleo é uma distância** — a distância cosseno entre o vetor da tela e o protótipo do "normal". Mas a decisão final combina essa distância com um segundo sinal. |
| **Qual o cálculo?** | Três passos: (1) `score = 1 − cos(z, protótipo)`; (2) `p(erro) = sigmoid` de uma combinação linear de `score` + cabeça auxiliar; (3) decide erro se `p(erro) > limiar`. |
| **É algum algoritmo de ML?** | **Sim, três peças clássicas empilhadas:** k-means (acha o protótipo), **regressão logística** (funde os sinais e calibra em probabilidade) e uma **otimização do limiar** na validação. E quem aprende o espaço onde a distância faz sentido é a rede siamesa. |
| **Como o limiar é escolhido?** | Por **otimização numa métrica de negócio, na validação** (nunca no teste): no padrão, o limiar que **maximiza o F1**; no modo alta-precisão, o menor limiar cuja **precisão ≥ alvo** (ex.: 0.95). |

### 3.2 A cadeia completa: do `z` à decisão (4 camadas)

```
            z (vetor 128-d da tela, na hiperesfera)
                         │
   ┌─────────────────────┴─────────────────────┐
   ▼                                            ▼
(A) DISTÂNCIA ao protótipo limpo          (B) CABEÇA AUXILIAR
    score = 1 − cos(z, protótipo)             aux = w·z + b   (logit direto)
   "longe do normal?"                         "parece erro?"
   └─────────────────────┬─────────────────────┘
                         ▼
        (C) FUSÃO CALIBRADA  (Regressão Logística, ajustada na validação)
            logit = c₀·score + c₁·aux + b₀
            p(erro) = sigmoid(logit)   ∈ [0, 1]
                         ▼
        (D) LIMIAR  (escolhido na validação)
            decisão = "ERRO" se p(erro) > limiar, senão "LIMPA"
```

**Por que duas entradas (A e B) e não só a distância?** A distância ao protótipo sozinha é um
**detector de novidade** — ele dispararia em qualquer app visualmente novo, mesmo limpo. A
cabeça auxiliar (B) é um detector binário direto que não depende do banco de protótipos.
**Fundir as duas** equilibra: pega o que é "longe do normal" **e** o que "parece erro".

> ⚠️ **Fidelidade ao diagrama executivo:** o slide mostra só o caminho do protótipo
> (A → "Cálculo da Probabilidade de Erro" → limiar) para não poluir. A caixa **"Cálculo da
> Probabilidade de Erro" é exatamente a fusão (C)** — é onde a distância vira `p(erro)`. Se
> alguém perguntar "como a distância vira probabilidade?", a resposta é: *"uma regressão
> logística calibra a distância (junto com o sinal auxiliar) numa probabilidade entre 0 e 1"*.
> O diagrama **detalhado** (`pipeline.mmd`, §3-bis do relatório) mostra as duas entradas.

### 3.3 As fórmulas (para o quadro branco, se pedirem)

**(A) Protótipo e distância** — `decision.py`:
```
protótipo  p = normaliza( média dos z das telas limpas de treino )   # k=1
score(z)   = 1 − máx_j cos(z, p_j)          # distância cosseno ao protótipo mais próximo
```

**(C) Fusão (regressão logística de 2 variáveis)** — ajustada na validação:
```
p(erro) = sigmoid( c₀ · score + c₁ · aux_logit + b₀ )
```

**(D) Escolha do limiar** — na validação, dois modos:
```
• padrão (objective = f1):       limiar = argmax_t  F1(t)          # ponto balanceado
• alta precisão (= precision):   limiar = menor t com Precisão(t) ≥ alvo   (ex.: 0.95)
```
Decisão: **`p(erro) > limiar`**.[^eps]

[^eps]: Detalhe de implementação: o código compara estritamente `>` (com um epsilon), enquanto
o diagrama escreve `≥` por simplicidade. Sem efeito prático.

### 3.4 "É distância? É ML? Qual o cálculo?" — resposta de 30 segundos

> 🎤 *"O coração é **sim** uma distância: medimos o quão longe o vetor da tela está do
> protótipo das telas limpas, com **distância cosseno**. Mas não paramos aí — essa distância
> entra numa **regressão logística** que a calibra numa **probabilidade de erro** de 0 a 100%.
> E o **corte** nessa probabilidade não é chutado: a gente escolhe, **na validação**, o valor
> que dá o melhor equilíbrio entre acertar erros e não dar falso-alarme. Então tem distância,
> tem ML (k-means pro protótipo, regressão logística pra calibrar) e o limiar é fruto de uma
> otimização, não de um palpite."*

### 3.5 Os dois "pontos de operação" (evita a confusão do "95%")

O **mesmo modelo** pode operar em dois cortes, conforme o custo do erro:

| Ponto de operação | Acurácia | Precisão | Recall | Quando usar |
|---|---|---|---|---|
| **Balanceado (padrão)** | **0.85** | **0.86** | **0.86** | comparação justa / uso geral |
| Alta precisão (opcional) | 0.74 | **1.00** | 0.50 | quando falso-alarme é caro (fila de triagem) |

> Mover o limiar é como **ajustar a sensibilidade de um filtro de spam**: mais sensível pega
> mais erros mas gera mais falso-alarme. **Não há roubo nem vazamento** — o limiar é sempre
> fixado na **validação** e medido no **teste**. Apresente o **0.85 balanceado** como número
> principal; o "1.00 de precisão" é um **modo**, não o headline.

---

## 4. Banco de perguntas e respostas (Q&A)

> Organizado pelos quatro temas que a equipe sinalizou + armadilhas de dados.

### Extração de características

**P: Como vocês extraem as características da imagem?**
R: Pré-processamento sem distorção (padding cinza → 518×518) → **DINOv2 ViT-S/14 congelado**
→ vetor de **1152 dimensões** = token global (CLS) + média + desvio-padrão dos patches.

**P: Por que DINOv2 e não uma CNN treinada do zero / ResNet?**
R: DINOv2 é auto-supervisionado em 142M de imagens e dá features visuais riquíssimas **sem
rótulos**. Treinar do zero com 360 imagens não competiria. Usamos como extrator fixo.

**P: Por que congelar o backbone?**
R: 22M de parâmetros + 360 imagens = overfitting garantido; ele decoraria os confounds.
Congelado, ele é estável e os vetores podem ser **cacheados** (treino em segundos).

**P: Por que incluir média e desvio dos patches, e não só o CLS?**
R: Erros espaciais (faixa preta, espaço vazio) são anomalias de **homogeneidade** — o
desvio-padrão dos patches cai em áreas grandes uniformes. Medimos o ganho: detecção
sintética subiu de **AUROC 0.71 → 0.88** ao adicionar as estatísticas de patch.

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
R: **Supervised Contrastive (SupCon)** + um termo BCE da cabeça auxiliar
(`Perda = SupCon + 0.3·BCE`). SupCon é a generalização em lote da ideia âncora/positivo/
negativo, mais estável com pouco dado. Triplet clássica não.

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
(`Linear 128→1`) é um detector binário **direto** que não depende de referências. Fundir as
duas dá robustez (ver Parte 3.2).

**P: Quanto tempo leva o treino?**
R: Segundos. O gargalo (DINOv2) roda 1× por imagem e fica em cache; a cabeça é minúscula.
300 épocas com early-stopping pela métrica de validação.

### Decisão / limiar

**P: Como é o corte? (resposta curta)** → ver **Parte 3.4** (resposta de 30s).

**P: O limiar é uma distância fixa?**
R: O **sinal** é uma distância (cosseno ao protótipo), mas o **corte** é feito sobre a
**probabilidade calibrada** `p(erro)`, e seu valor é **escolhido na validação** por
otimização (F1 máximo no padrão; precisão-alvo no modo alta-precisão). Não é um número fixo
arbitrário.

**P: Por que k=1 protótipo? E se as telas limpas forem variadas?**
R: Aqui o conjunto limpo é unimodal, então 1 centro basta. O código suporta **k-means com
k>1** para o caso de múltiplos estilos de tela limpa — é só mudar a config.

**P: Por que a distância cosseno e não euclidiana?**
R: Como os vetores estão normalizados (hiperesfera), cosseno mede **direção/conteúdo**
ignorando escala — é o casamento natural com o treino contrastivo (que também usa cosseno).

**P: A regressão logística não é "outro modelo"? Não é overkill?**
R: É um modelo linear de **2 variáveis** (`score` e `aux`), ajustado na validação só para
**calibrar** os dois sinais numa probabilidade e pesá-los corretamente. É leve e evita
fundir por "OU" (que dispararia recall e derrubaria precisão).

### Dados, avaliação e armadilhas

**P: Por que a acurácia é "só" 0.85 se uma regra boba dá 98%?**
R: Porque os 98% são **trapaça** — a regra "resolução ≠ padrão ⇒ erro" detecta o **aparelho**,
não o erro. Nosso modelo evita esse atalho de propósito; o 0.85 é **honesto**. A prova:
no teste sintético **livre de confound**, AUROC 0.88.

**P: Como sabemos que não há vazamento entre treino e teste?**
R: O split é **agrupado por ticket** (IKSWW): todas as imagens de um mesmo bug ficam no mesmo
conjunto. Verificado: 0 grupos cruzando splits.

**P: Qual a maior alavanca para melhorar?**
R: **Dados, não arquitetura.** Coletar telas **limpas** de outros aparelhos/resoluções/fotos.
Enquanto a classe limpa vier de um device só, a métrica global continua dominada por confound.

---

## 5. Colinha — números e termos na ponta da língua

**Números (decore os 6 primeiros):**

| | Valor |
|---|---|
| Acurácia (ponto balanceado, padrão) | **0.85** |
| Precisão / Recall / F1 | **0.86 / 0.86 / 0.86** |
| AUROC / AP (livres de limiar) | **0.90 / 0.92** |
| **Detecção sintética livre de confound (prova honesta)** | **AUROC 0.88 / AP 0.97** |
| Modo alta precisão | precisão **1.00**, recall **0.50** |
| Regra trivial de resolução (a trapaça) | acurácia **~98%** (AUROC 0.982) |
| Dataset / split | 360 imgs (172 limpas + 188 erro) · treino 252 / val 54 / test 54 |
| Backbone | DINOv2 ViT-S/14 · 22M params · **congelado** · grade 37×37 |
| Embedding / cabeça / `z` | 1152-d → cabeça ~330k params → `z` 128-d |

**Termos (tradução rápida para leigos):**

- **Confound** → "atalho enganoso nos dados" (aqui: a resolução).
- **Embedding / vetor de características** → "a tela virada em números".
- **Backbone congelado** → "olho pré-treinado que não reeducamos".
- **Hiperesfera** → "a superfície onde os vetores vivem; comparar = medir ângulo".
- **Protótipo** → "o centro do que é normal".
- **Distância cosseno** → "o quão longe/diferente do normal".
- **Limiar** → "a cerca que separa erro de não-erro".
- **AUROC** → "nota de 0 a 1 da capacidade de separar as classes, sem depender do corte".
- **Recall** → "dos erros que existem, quantos achamos"; **Precisão** → "do que marcamos como
  erro, quanto era erro de verdade".

---

## 6. Versões curtas (5 min e 15 min)

**Versão 5 minutos (só o caminho principal):**
1. Objetivo + a trapaça do confound (frase de abertura). *(45s)*
2. Etapa 2 — sintéticos: a solução anti-trapaça. *(1 min)*
3. Etapa 3 — DINOv2 congelado vira a tela em números. *(45s)*
4. Etapa 4 — a cabeça siamesa organiza o espaço (limpo junto, erro fora). *(1 min)*
5. Etapa 5 — distância ao protótipo → probabilidade → limiar → ✅/❌. *(1 min)*
6. Número honesto: 0.85 / AUROC 0.88 livre de confound. *(30s)*

**Versão 15 minutos:** a de 5 min + abrir a Parte 3 (cadeia A→D do limiar com o quadro
branco) + 2–3 perguntas do Q&A que você achar mais prováveis na sua equipe.

**Se só puder dizer uma frase:**
> *"O sistema transforma a tela num vetor com um modelo de visão congelado, uma cabeça
> siamesa reorganiza esse vetor para que telas limpas formem um aglomerado, e a decisão é a
> distância até o centro desse aglomerado — calibrada em probabilidade e cortada por um
> limiar escolhido na validação. Todo o resto do pipeline existe para que ele aprenda o
> **erro**, e não o **aparelho**."*
