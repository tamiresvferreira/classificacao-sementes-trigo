# 🌱 Classificação de Sementes de Trigo

Modelo de machine learning que identifica a variedade de um grão de trigo (**Kama**, **Rosa** ou **Canadian**) a partir de sete medidas físicas, com análise completa em CRISP-DM e uma **demo interativa em Streamlit**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tamiresvferreira/classificacao-sementes-trigo/blob/main/notebooks/classificacao_sementes.ipynb)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

![Demo do classificador em Streamlit](assets/demo_app.png)

## 🎯 O problema e para quem é

Em cooperativas e unidades de beneficiamento, separar variedades de grãos ainda depende muito de inspeção visual. A pergunta do projeto: **dá para identificar a variedade só com medidas físicas simples do grão?** Se sim, a classificação pode ser automatizada com um sistema de medição e um modelo leve, sem depender de um especialista olhando grão por grão.

## ✨ Destaques

- **Validação sem vazamento de dados:** padronização e modelo ficam no mesmo `Pipeline`, então o scaler é reajustado dentro de cada fold da validação cruzada.
- **Avaliação robusta para dataset pequeno:** validação cruzada estratificada **repetida** (5 folds × 10 repetições, 50 scores por modelo) em vez de um único split.
- **Análise de erros:** identifica os grãos que confundem os modelos e mostra onde eles ficam na projeção LDA.
- **Do notebook à demo:** o modelo recomendado é salvo com `joblib` e servido em um app Streamlit, em que dá para digitar medidas ou sortear grãos reais.

## 📊 Dados

[Seeds Dataset](https://archive.ics.uci.edu/dataset/236/seeds) do UCI Machine Learning Repository (licença CC BY 4.0): 210 grãos, 70 de cada variedade, sem valores ausentes. Uma cópia está em `data/seeds_dataset.txt`.

| Feature | Descrição |
|---|---|
| `area` | Área do grão |
| `perimetro` | Perímetro |
| `compacidade` | 4π · área / perímetro² |
| `comprimento_nucleo` | Comprimento do núcleo |
| `largura_nucleo` | Largura do núcleo |
| `coef_assimetria` | Coeficiente de assimetria |
| `comprimento_sulco` | Comprimento do sulco do núcleo |

## 🔄 Metodologia (CRISP-DM)

1. **Compreensão dos dados:** estatísticas descritivas, histogramas e boxplots por variedade, correlação, dispersão e projeções PCA/LDA.
2. **Preparação:** split 70/30 estratificado (49 + 21 grãos por classe) e `StandardScaler` dentro do `Pipeline`.
3. **Modelagem:** KNN, SVM, Random Forest e Regressão Logística, quatro abordagens diferentes de classificação.
4. **Otimização:** `GridSearchCV` com 5 folds e F1-Score macro.
5. **Avaliação:** matrizes de confusão, validação cruzada repetida, análise de erros e importância das features.
6. **Implantação:** modelo final retreinado com os 210 grãos e servido no app Streamlit.

As variedades formam três grupos bem definidos. Kama fica no meio, entre os grãos grandes de Rosa e os pequenos de Canadian:

![Projeção PCA e LDA](assets/separabilidade_pca_lda.png)

## 📈 Resultados

| Modelo | F1 CV (50 folds) | Desvio | Acurácia no teste |
|---|---|---|---|
| **SVM (kernel linear, C=100)** | **0,964** | **0,029** | 88,9% |
| Regressão Logística | 0,953 | 0,039 | 88,9% |
| KNN (Manhattan, k=3) | 0,942 | 0,039 | 88,9% |
| Random Forest | 0,906 | 0,057 | 88,9% |

![Validação cruzada repetida](assets/validacao_cruzada.png)

**O que os números mostram:**

- **Os quatro modelos otimizados acertaram os mesmos 56 de 63 grãos no teste.** Quando abordagens tão diferentes empatam, o limite está nos dados, não no algoritmo.
- **A validação cruzada repetida desempata:** o SVM tem a maior média e o menor desvio. O Random Forest é o mais instável, com folds de 76% a 100%.
- **4 grãos foram errados pelos quatro modelos**, sempre com a mesma previsão. Pelas medidas, eles parecem de outra variedade. Resolver isso exigiria novas features (cor, textura, imagem), não outro modelo.

![Análise de erros na projeção LDA](assets/analise_erros_lda.png)

- **Área, perímetro e comprimento do sulco** somam cerca de 65% da importância no Random Forest. Compacidade e assimetria contribuem pouco.
- **O Grid Search não melhorou todos os modelos:** o Random Forest caiu de 92,1% para 88,9% no teste. Com 63 grãos no teste, cada erro vale ~1,6 ponto percentual, então a média dos 50 folds é a estimativa mais confiável.

**Recomendação:** SVM com kernel linear, pela maior média e menor variação na validação cruzada, pela robustez a outliers e por ser leve em produção.

## 🖥️ Demo em Streamlit

O app recebe as sete medidas na escala original e mostra:

- a variedade prevista e a confiança;
- a probabilidade de cada variedade;
- onde o grão fica em relação ao dataset, com eixos escolhidos por você;
- um botão para sortear grãos reais de cada variedade e conferir se o modelo acerta.

## 📁 Estrutura de pastas

- **`app.py`**: app Streamlit da demo.
- **`notebooks/classificacao_sementes.ipynb`**: análise completa, do entendimento dos dados ao modelo final.
- **`data/seeds_dataset.txt`**: dataset original do UCI.
- **`models/modelo_sementes.joblib`**: pipeline final (scaler + SVM) usado pelo app.
- **`assets/`**: figuras geradas pelo notebook e captura da demo.
- **`requirements.txt`**: dependências.
- **`LICENSE`**: licença MIT.

## 🔧 Como executar

```bash
git clone https://github.com/tamiresvferreira/classificacao-sementes-trigo.git
cd classificacao-sementes-trigo
pip install -r requirements.txt
```

Rodar a demo:

```bash
streamlit run app.py
```

Rodar a análise:

```bash
jupyter notebook notebooks/classificacao_sementes.ipynb
```

Ou abra no Colab pelo badge no topo. O notebook lê os dados direto do GitHub quando o arquivo não está disponível localmente.

## 🗃 Histórico de lançamentos

- **1.0.0 - 13/11/2026**
  - Repositório próprio (antes o notebook ficava no repo do perfil).
  - Padronização dentro do `Pipeline` para eliminar vazamento na validação cruzada.
  - Validação cruzada estratificada repetida, projeções PCA/LDA e análise de erros.
  - Modelo final salvo com `joblib` e demo em Streamlit.
  - Notebook salvo com todas as saídas e dataset incluído no repositório.
- **0.1.0**
  - Entrega original: análise CRISP-DM com quatro modelos e Grid Search.

## 🤝 Contribuições

Sugestões e melhorias são bem-vindas:

1. Abra uma [issue](https://github.com/tamiresvferreira/classificacao-sementes-trigo/issues) descrevendo a ideia ou o problema.
2. Para mudanças no código, faça um fork, crie uma branch (`git checkout -b minha-melhoria`) e abra um pull request explicando o que mudou e por quê.

## 📋 Licença

Código sob a [licença MIT](LICENSE). O Seeds Dataset é do [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/236/seeds), sob licença CC BY 4.0 (Charytanowicz et al., 2010).

---

Feito por [Tamires Ferreira](https://github.com/tamiresvferreira) · estudante de Inteligência Artificial na FIAP · [LinkedIn](https://br.linkedin.com/in/tamiresvferreiras)
