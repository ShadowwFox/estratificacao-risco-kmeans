# Estratificação de Risco e Fenotipagem Exploratória (K-Means)

Repositório oficial contendo o pipeline de dados desenvolvido para o Trabalho de Conclusão de Curso do **MBA em Inteligência Artificial e Big Data** pelo ICMC - Universidade de São Paulo (USP).

**Autor:** Lucas Gabriel Silva de Santana

## 🧠 Sobre o Projeto
Este pipeline realiza o processamento de ponta a ponta de dados psicométricos de regulação emocional (DERS, EQR, LESS-II) e sofrimento psicológico (DASS-21). O objetivo principal é identificar perfis (fenótipos) de desregulação emocional utilizando aprendizado não supervisionado.

### Principais Etapas do Pipeline:
1. **Pré-processamento:** Higienização, tratamento de atrito amostral e recodificação de itens reversos.
2. **Escalonamento:** Transformação em *z-scores* das 22 dimensões clínicas.
3. **Clusterização:** Algoritmo *K-Means* (k-means++).
4. **Auditoria e Estabilidade:** 
   - Reamostragem via *Bootstrap* (100 iterações).
   - Análise probabilística via *Gaussian Mixture Models* (GMM).
   - Imputação múltipla via *MICE* (análise de sensibilidade).
5. **Testes de Hipótese:** Testes não paramétricos (Mann-Whitney e Kruskal-Wallis) com correção de Holm/Bonferroni.
6. **Explicabilidade (XAI):** Treinamento de um *Random Forest* auxiliado por *SHAP values* para interpretar os pesos das features na definição do cluster de Alto Risco.

## ⚙️ Como executar

### Pré-requisitos
Instale as dependências executando:
```bash
pip install -r requirements.txt
