💻 PeopleFlow v1.0

PeopleFlow é um sistema de Visão Computacional em Python para análise de fluxo e monitoramento de lotação em ambientes internos.
Utiliza YOLOv8 + ByteTrack para detecção e rastreamento, com registro de dados em SQLite e exportação automática para Excel.

✨ Funcionalidades Principais

Detecção e Rastreamento com YOLOv8n + ByteTrack

Contagem de Pessoas em Tempo Real

ROI (Zona de Interesse) definida via máscara (Assets/mask-1.png)

Persistência de Dados em Data/flow_log.db

Exportação para Excel (flow_log.xlsx)


⚙️ Como Executar
1. Criar Ambiente Virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

2. Instalar Dependências
pip install -r requirements.txt

3. Rodar a Aplicação
python app.py


Para usar webcam, altere USE_WEBCAM = True no app.py.