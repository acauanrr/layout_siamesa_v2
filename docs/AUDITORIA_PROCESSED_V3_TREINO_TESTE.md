# Auditoria do treino/teste em `processed_v3`

Data da auditoria: 2026-06-26  
Dataset auditado: `data/processed_v3`  
Artefatos principais: `artifacts/reports/processed_v3/`

## 1. Resumo executivo

O comportamento observado em `clusters_treino.html` nao e, sozinho, evidencia de erro de
treinamento. A separacao visual quase perfeita no treino e esperada porque o proprio objetivo
de treino usa SupCon multi-classe + cross-entropy para aproximar exemplos da mesma categoria e
afastar categorias diferentes no espaco `z`. Alem disso, a visualizacao de treino e
ressubstituicao: ela mostra pontos que participaram diretamente do ajuste da cabeca siamesa.

Mesmo assim, a preocupacao do seu chefe e tecnicamente valida. O teste held-out mostra que a
separacao aprendida nao generaliza bem para tickets/dispositivos novos:

| Regime | Acuracia | F1-macro / F1 | AUROC | Observacao |
|---|---:|---:|---:|---|
| Gate treino, in-sample | 0.875 | F1 0.888 | 0.991 | Nao e metrica de generalizacao |
| Gate teste, held-out | 0.583 | F1 0.634 | 0.596 | Generalizacao fraca/moderada |
| 5 classes treino, in-sample | 0.875 | F1-macro 0.880 | 0.989 | Categoria vista no treino |
| 5 classes teste, held-out | 0.380 | F1-macro 0.291 | 0.604 | Categorias confundidas no teste |

Conclusao: nao encontrei evidencia de vazamento classico no dataset atual, mas ha forte
evidencia de avaliacao otimista no treino e de limite estrutural dos dados. O principal problema
nao parece ser "o codigo treinou errado"; parece ser uma combinacao de confound de resolucao,
baixo suporte por classe, dominio sintetico diferente do real e apresentacao visual que facilita
interpretacao excessivamente otimista.

## 2. Evidencias dos artefatos

### 2.1 `clusters_treino.html` / `clusters_treino.png`

A visualizacao de treino mostra grupos muito separados, mas isso deve ser lido como "o modelo
conseguiu ajustar o conjunto visto", nao como "o modelo generaliza". A matriz de treino tambem
nao e literalmente perfeita:

- `clean`: 104/105 corretos.
- `black_bars`: 52/52 corretos.
- `disordered_layout`: 26/26 corretos.
- `empty_space`: 36/36 corretos.
- `overlay`: apenas 21/54 corretos como `overlay`; 33/54 foram para `clean`.

Ou seja, a perfeicao visual esta concentrada em algumas categorias. `overlay` ja apresenta falha
relevante mesmo no treino quando o sistema fim-a-fim aplica o gate antes do prototipo de categoria.

### 2.2 `clusters_teste.html` / `clusters_teste.png`

No teste, os pontos aparecem misturados nas regioes do treino. Isso bate com a matriz 5 classes:

- `disordered_layout`: 0/10 corretamente classificados.
- `overlay`: 3/21 corretos fim-a-fim; 11/21 viram `clean`.
- `empty_space`: 3/14 corretos.
- `black_bars`: melhor classe, 11/22 corretos.

O teste e a melhor evidencia de que o espaco `z` aprendido nao transfere a fronteira de categoria
de forma robusta para exemplos novos.

Um detalhe importante: o treino do modelo usou reais + sinteticos + reflow-clean, mas
`clusters_treino.html` mostra apenas as 273 imagens reais de treino. Portanto, ele nao mostra todo
o conjunto efetivo de otimizacao; mostra um recorte real, in-sample, ja moldado pela cabeca treinada.

### 2.3 Gate erro/sem-erro

No teste, a matriz binaria e:

- TP = 39
- TN = 24
- FP = 17
- FN = 28

Isso produz precisao 0.696, recall 0.582, especificidade 0.585 e AUROC 0.596. O resultado e
operacionalmente util como sinal fraco/moderado, mas nao sustenta a narrativa de um modelo
altamente separador.

## 3. Auditoria de dados e vazamento

Executei uma auditoria fresca com:

```bash
.venv/bin/python scripts/audit_dataset.py \
  --root data/processed_v3 \
  --labels data/processed_v3/labels.csv \
  --near-dist 4 \
  --json /tmp/processed_v3_audit_fresh.json
```

Resultado principal:

| Checagem | Resultado |
|---|---:|
| Imagens totais | 868 |
| Corrompidas | 0 |
| Duplicatas exatas | 0 |
| Duplicatas exatas cruzando split | 0 |
| Vazamento por grupo real | 0 |
| Parent sintetico em val/test | 0 |
| Reais sem rotulo | 0 |
| Near-duplicates cruzando split | 4 pares |

Os 4 near-duplicates cruzando split devem ser revisados manualmente. Eles incluem um caso
importante: uma tela limpa de treino (`Screenshot_20260614_231859.png`) e suas variantes
sinteticas `overlay` ficam perceptualmente proximas de um `overlay` real do teste
(`IKSWW-189858_unfold_portrait_screenshot1.png`). Isso nao configura vazamento exato nem de
grupo, mas pode inflar ou confundir leituras pontuais. Recomendo tratar como achado P1.

Tambem confirmei diretamente em `labels.csv`:

- 868 linhas.
- 0 grupos cruzando `train`/`val`/`test`.
- Distribuicao esperada por split e categoria.

Observacao: `artifacts/audit/processed_v3.json` esta desatualizado em relacao ao dataset atual e
lista vazamentos antigos. Nao deve ser usado como evidencia do estado atual sem regeneracao.

## 4. Confound estrutural

O confound de resolucao continua sendo a principal explicacao alternativa.

Contagem por resolucao canonica `(2076, 2152)`:

| Classe | Resolucao canonica / total |
|---|---:|
| `clean` | 172/172 |
| `black_bars` | 104/191 |
| `disordered_layout` | 92/134 |
| `empty_space` | 114/173 |
| `overlay` | 113/198 |

No teste real, todas as 41 imagens `clean` estao em 2076x2152 e 0/67 erros reais estao nessa
resolucao. Por isso:

- baseline trivial de resolucao: AUROC 1.000;
- baseline de fracao de padding cinza: AUROC 1.000;
- o score do modelo prediz "erro" e "resolucao nao canonica" com o mesmo AUROC global: 0.596.

Isso nao prova que o modelo so usa resolucao, mas prova que o teste real atual nao consegue
separar completamente "deteccao de erro" de "deteccao de dominio/dispositivo". A melhor evidencia
anti-confound ainda e a sonda sintetica livre de confound, com AUROC 0.721, mas ela mede erros
sinteticos, nao erros reais independentes.

Outro baseline relevante: LogReg em DINOv2 bruto chega a AUROC 0.716 no teste global, acima do
modelo final por fusao/prototipo global. Como o global e confundido, isso nao significa que o
baseline seja "melhor" em conteudo de erro; significa que ele deve virar piso obrigatorio em
comparacoes controladas.

## 5. Pipeline auditado

O pipeline atual e:

1. `data/processed_v3/labels.csv` define split, classe, grupo e metadados.
2. `scripts/extract_features.py` extrai embeddings DINOv2 congelados para `artifacts/embeddings/*.npz`.
3. `scripts/make_synthetic.py` gera sondas `val_synth`, `test_synth` e variantes `*_reflow`.
4. `src/siamese/train.py` treina uma cabeca `ProjectionHead` sobre embeddings DINOv2:
   - `LayerNorm -> Linear -> GELU -> Dropout -> Linear -> L2 normalize`;
   - perda SupCon multi-classe;
   - cross-entropy auxiliar;
   - batches balanceados por classe;
   - treino com erros reais, erros sinteticos e reflow-clean.
5. Early stop usa `val_synth_gate`, uma sonda livre de confound na validacao.
6. `src/siamese/evaluate.py` ajusta prototipos limpos, fusao logistica `[score_proto, aux_err]` e
   limiar usando validacao.
7. Decisao final:
   - Estagio 1: gate erro/sem-erro.
   - Estagio 2: se gate=erro, categoria por prototipo de erro mais proximo.

O desenho e coerente com a proposta. O problema e que a evidencia visual de treino e muito mais
forte que a evidencia held-out, e o dataset real continua sem contraponto suficiente para provar
robustez fora do confound.

## 6. Tratamentos de erro e blindagens existentes

Pontos positivos implementados:

- `src/siamese/protocol.py` bloqueia leitura de artefatos `test*` sem `--final-test`.
- `features.load_embeddings()` passa pelo guard de teste.
- `train.py` salva proveniencia no checkpoint, incluindo config e hashes dos embeddings.
- `rebuild_processed_v3.py` faz staging e auto-verificacao antes de substituir o dataset.
- `evaluate.py` reporta baselines de confound, IC95 bootstrap por grupo, ponto de operacao e
  comparacao de calibracao.
- `train.py` avisa quando `train_reflow.npz` esta ausente e evita usar sintetico quando
  `synthetic.enabled=false`.

Riscos/fragilidades:

- `scripts/report_processed_v3.py` destrava o teste por design para analise post-hoc. Isso e aceitavel
  para relatorio final, mas deve ser tratado como artefato de apresentacao, nao como selecao.
- O UMAP do relatorio visual e ajustado em `train + test + prototipos`. Isso nao contamina o treino,
  mas e uma visualizacao transdutiva. Para convencer terceiros, e melhor ajustar o redutor no treino
  e apenas projetar o teste.
- `scripts/audit_dataset.py` tem default de labels em `data/splits/all.csv`; para `processed_v3`,
  o correto e `data/processed_v3/labels.csv`. Esse default pode gerar auditorias inconsistentes.
- Os testes automatizados de split focam `data/processed`, nao `data/processed_v3`.
- `run_experiment.py` pode pular embeddings existentes sem validar se `labels.csv` mudou. O checkpoint
  guarda hashes, mas o pipeline deveria invalidar artefatos automaticamente.
- O teste `auroc_label_shuffle_no_estrato` no relatorio final saiu 0.718, embora o comentario espere
  algo perto de 0.5. Isso sugere que a falseabilidade por shuffle esta degenerada pelo proprio
  confound e nao deve ser usada como prova forte sem revisao.

## 7. Existe erro no treinamento?

Minha avaliacao: nao encontrei um erro unico e evidente do tipo "vazamento de teste no treino",
"split quebrado" ou "rotulo errado em massa". O treino atual faz o que foi programado para fazer:
separar categorias no espaco aprendido a partir de exemplos vistos e de sinteticos.

Mas ha tres problemas reais:

1. A visualizacao de treino esta sendo interpretada como evidencia de generalizacao, e ela nao e.
2. O dataset real permite um atalho perfeito de resolucao, entao qualquer narrativa precisa liderar
   por metricas livres de confound e baselines.
3. O desempenho held-out por categoria e fraco para `disordered_layout`, `empty_space` e `overlay`;
   a taxonomia fina ainda deve ser apresentada como exploratoria.

Portanto, eu nao defenderia "o treinamento esta perfeito". Eu defenderia:

> "O protocolo atual reduziu vazamento e snooping, mas o dataset ainda nao permite provar robustez
> em dados reais. O treino separa bem o conjunto visto; o teste mostra generalizacao limitada. O
> proximo passo e endurecer a auditoria, melhorar a diversidade de dados e trocar a narrativa de
> 'cluster perfeito' por 'sinal moderado com limites claros'."

## 8. Plano de execucao

### Fase 0 - Arrumar a defesa experimental

Prazo: 1 a 2 dias.

- Regenerar e versionar `artifacts/audit/processed_v3.json` a partir de `data/processed_v3/labels.csv`.
- Corrigir `scripts/audit_dataset.py` para usar `root/labels.csv` automaticamente quando existir.
- Adicionar testes automatizados para `processed_v3`:
  - 0 grupos cruzando split;
  - 0 duplicatas exatas cruzando split;
  - 0 parent sintetico em val/test;
  - contagem por categoria igual ao `DATASET_CARD.md`.
- Criar uma visualizacao nova:
  - UMAP/PCA ajustado apenas no treino;
  - teste projetado depois;
  - pontos coloridos por `correto/errado`, nao apenas por classe real;
  - hover com score, predicao, rotulo e grupo.
- Incluir no relatorio um aviso explicito: `clusters_treino` e in-sample.

### Fase 1 - Fechar perguntas do chefe com experimentos falsificaveis

Prazo: 2 a 4 dias.

- Rodar ablações controladas:
  - sem erros reais, so sinteticos;
  - sem sinteticos;
  - sem reflow;
  - somente prototipo;
  - somente aux head;
  - DINOv2 cru + LogReg;
  - one-class kNN em DINOv2.
- Reportar sempre:
  - teste held-out;
  - sonda sintetica livre de confound;
  - subconjunto controlado;
  - baseline resolucao/padding;
  - IC95 por grupo;
  - gap treino-teste.
- Repetir com 5 a 10 seeds e grouped CV. O objetivo e provar estabilidade, nao apenas um split.

### Fase 2 - Melhorar dados

Prazo: depende da coleta; e a fase de maior impacto.

- Coletar telas `clean` em resolucoes/form-factors equivalentes aos erros:
  - fold/unfold;
  - portrait/landscape;
  - laptop/tent/desktop;
  - fotos e screenshots;
  - apps/telas variadas.
- Coletar erros reais em 2076x2152 para quebrar o confound pelo lado positivo.
- Definir meta minima por estrato, por exemplo 30-50 limpas por combinacao critica.
- Separar um novo teste held-out temporal, coletado depois do congelamento da proxima configuracao.

### Fase 3 - Melhorar modelo e decisao

Prazo: 1 a 2 semanas apos Fase 1.

- Trocar selecao de limiar por objetivo operacional claro:
  - alta especificidade se falso alarme e caro;
  - alta sensibilidade se perder bug e caro;
  - reportar curva completa, nao um unico ponto.
- Testar prototipos OOF por grupo para reduzir otimismo dos prototipos de treino.
- Testar hard-negative mining: limpas reais/reflow que ficam perto dos prototipos de erro.
- Considerar perda metric-learning com margem por categoria ou ArcFace/CosFace leve.
- Reavaliar `overlay` e `disordered_layout` como multi-label ou taxonomia grossa. Hoje a taxonomia
  fina tem suporte pequeno e classes visualmente sobrepostas.
- Usar o baseline LogReg DINOv2 cru como piso obrigatorio: o modelo novo deve supera-lo no regime
  controlado, nao so em metricas globais confundidas.

### Fase 4 - Criterios de aceite

O projeto deveria ser considerado convincente apenas quando:

- O AUROC livre de confound ficar acima de 0.75 de forma estavel em multi-seed/grouped CV.
- O modelo superar DINOv2 cru + LogReg e one-class kNN no subconjunto controlado.
- O gap treino-teste de AUROC cair para uma faixa aceitavel, por exemplo menor que 0.15.
- A especificidade em telas limpas held-out tiver limite inferior de IC95 acima da meta operacional.
- `disordered_layout` sair de recall/F1 zero ou for removido da promessa de classificacao fina.
- O novo teste held-out tiver limpas e erros compartilhando resolucao/form-factor suficientes para
  impedir a regra trivial de resolucao.

## 9. Recomendacao para apresentacao

Evite apresentar `clusters_treino.html` como prova principal. Use-o apenas para explicar o que o
modelo aprendeu no conjunto visto. A narrativa mais defensavel e:

1. "O treino separa porque o objetivo de treino força esse espaco."
2. "O teste real mostra generalizacao limitada."
3. "Nao ha vazamento classico encontrado na auditoria atual."
4. "O dataset tem confound estrutural de resolucao; por isso lideramos por metricas livres de
   confound e por baselines."
5. "O plano de melhoria prioriza dados e validacao, antes de prometer ganho de arquitetura."
