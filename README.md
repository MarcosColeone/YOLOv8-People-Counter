# 🚶 Contador Inteligente de Pessoas com YOLOv8 (People Counter)

Projeto de Visão Computacional desenvolvido em Python para contagem e monitoramento de fluxo de pessoas em tempo real, utilizando o modelo YOLOv8 da Ultralytics e o rastreador ByteTrack.

## ✨ Funcionalidades Principais

* **Detecção e Rastreamento em Tempo Real:** Utiliza **YOLOv8n** (modelo nano) para detecção rápida e eficiente de pessoas (classe `0`).
* **Rastreamento Robusto:** Emprega o **ByteTrack** para atribuir IDs únicos e estáveis a cada pessoa.
* **Contagem de Lotação:** Calcula a **lotação atual** no frame e o **total de pessoas vistas** (acúmulo de IDs únicos).
* **Área de Interesse (ROI):** Usa uma **máscara binária** para delimitar a área de contagem, ignorando regiões irrelevantes do vídeo.
* **Exportação de Dados:** Gera um arquivo **Excel (.xlsx)** com logs de tempo, lotação e total acumulado, fundamental para relatórios e análises de BI (Business Intelligence).

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos

Certifique-se de ter o **Python 3.11** instalado.

### 2. Configuração do Ambiente

Crie e ative um ambiente virtual para isolar as dependências do projeto:

```bash
# 1. Cria o ambiente virtual
python -m venv .venv

# 2. Ativa o ambiente virtual (Windows)
.venv\Scripts\activate
# OU (Linux/macOS)
source .venv/bin/activate