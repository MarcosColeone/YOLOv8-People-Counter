from ultralytics import YOLO
import cv2
import numpy as np
import time
import pandas as pd
import os
import sys

# ----------------------------------------------------------------------
# 1. CLASSE PRINCIPAL DO CONTADOR DE PESSOAS 
# ----------------------------------------------------------------------

class PeopleCounter:
    """
    Contador de Pessoas modularizado usando YOLOv8 e ByteTrack.
    Suporta vídeo ou webcam e registra dados para exportação em Excel.
    """
    def __init__(self, source_path, mask_path, model_path="yolov8n.pt", 
                 conf_threshold=0.5, resolution=(1280, 720), log_interval_seconds=5):
        # Configurações
        self.SOURCE_PATH = source_path
        self.MASK_PATH = mask_path
        self.MODEL_PATH = model_path
        self.CONF_THRESHOLD = conf_threshold
        self.FRAME_WIDTH, self.FRAME_HEIGHT = resolution
        
        # Log comercial por intervalo de tempo
        self.LOG_INTERVAL_SECONDS = log_interval_seconds 
        self.LAST_LOG_TIME = time.time()
        
        # Variáveis de Estado
        self.unique_ids = set() 
        self.active_ids = set() 
        self.data_log = []
        self.start_time = time.time()
        self.frame_count = 0
        self.model = None
        self.mask = None
        self.mask_resized = None

    def load_resources(self):
        """Carrega o modelo YOLOv8 e a máscara de área de interesse."""
        print(f"[INFO] Carregando modelo: {self.MODEL_PATH}...")
        self.model = YOLO(self.MODEL_PATH)
        
        print("[INFO] Preparando máscara...")
        self.mask = cv2.imread(self.MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise Exception(f"❌ Erro: Máscara não encontrada no caminho: {self.MASK_PATH}")

        # Prepara a máscara para a resolução de processamento
        self.mask_resized = cv2.resize(self.mask, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        _, self.mask_resized = cv2.threshold(self.mask_resized, 127, 255, cv2.THRESH_BINARY)
        
    def process_frame(self, frame):
        """
        Aplica máscara, realiza detecção/rastreamento e desenha resultados, 
        incluindo o ID do rastreamento e a porcentagem de confiança.
        """
        
        # 1. Redimensionar e Aplicar Máscara
        frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask_resized)
        
        # 2. Detecção + Rastreamento
        results = self.model.track(
            masked_frame,
            persist=True,
            conf=self.CONF_THRESHOLD,
            classes=[0],  # 0 = pessoa
            tracker="bytetrack.yaml",
            verbose=False
        )

        self.active_ids.clear()
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy() # Extrai a confiança!

            # 3. Iterar sobre os resultados
            for box, track_id, conf in zip(boxes, track_ids, confidences): 
                x1, y1, x2, y2 = box
                self.active_ids.add(track_id)
                self.unique_ids.add(track_id) 
                
                # Formata a confiança
                confidence_percent = f"{int(conf * 100)}%" 
                label = f"ID {track_id} ({confidence_percent})" # Texto final com ID e Confiança

                # Desenhar caixas e IDs/Confiança
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 4. Exibir Métricas no Frame
        lotacao_atual = len(self.active_ids)
        total_pessoas = len(self.unique_ids)

        cv2.putText(frame, f"Lotacao atual: {lotacao_atual}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Total de pessoas: {total_pessoas}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

        # 5. Desenhar máscara semi-transparente
        overlay = frame.copy()
        overlay[self.mask_resized == 0] = (0, 0, 0)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # 6. Log de Dados (Amostragem por tempo)
        self._log_data_by_time(lotacao_atual, total_pessoas)
        self.frame_count += 1

        return frame

    def _log_data_by_time(self, lotacao_atual, total_pessoas):
        """Registra as métricas por intervalo de TEMPO (comercial)."""
        current_time = time.time()
        
        if (current_time - self.LAST_LOG_TIME) >= self.LOG_INTERVAL_SECONDS:
            elapsed_time = current_time - self.start_time
            self.data_log.append({
                "Tempo (s)": round(elapsed_time, 2),
                "Lotacao Atual": lotacao_atual,
                "Total Pessoas Vistas (Acumulado)": total_pessoas
            })
            self.LAST_LOG_TIME = current_time

    def export_data(self, excel_path):
        """Salva o log de dados em um arquivo Excel."""
        if self.data_log:
            df = pd.DataFrame(self.data_log)
            df.to_excel(excel_path, index=False)
            print(f"\n[INFO] 📊 Dados de contagem salvos em: {excel_path}")

# ----------------------------------------------------------------------
# 2. EXECUÇÃO PRINCIPAL (Suporte a Vídeo/Webcam)
# ----------------------------------------------------------------------

def run_local(counter: PeopleCounter, is_webcam: bool):
    """Loop principal para exibição em tempo real e salvamento de dados."""
    
    counter.load_resources()
    
    # Se for Webcam, usa o ID 0. Se for Vídeo, usa o caminho do arquivo.
    source = 0 if is_webcam else counter.SOURCE_PATH
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise Exception(f"❌ Erro ao abrir a fonte. Webcam pode estar em uso ou vídeo não encontrado.")
        
    print(f"[INFO] Processando {'Webcam' if is_webcam else 'Vídeo'} (Pressione 'q' para fechar a janela)...")
    
    # Otimização para modo webcam
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, counter.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, counter.FRAME_HEIGHT)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Se for vídeo, sai. Se for webcam, continua tentando ler
                if not is_webcam:
                    break
                continue
                
            processed_frame = counter.process_frame(frame)
            cv2.imshow("People Counter - YOLOv8", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass # Captura Ctrl+C

    # Finalização e Exportação
    cap.release()
    cv2.destroyAllWindows()
    
    print("[INFO] ✅ Processamento finalizado.")
    
    excel_filename = f"registro_contagem_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    excel_path_full = os.path.join("Data", excel_filename)
    
    # Garante que a pasta Data exista antes de salvar
    if not os.path.exists("Data"):
        os.makedirs("Data")

    counter.export_data(excel_path_full)


if __name__ == '__main__':
    # ===============================================
    # === 🛑 CONFIGURAÇÕES DO PROJETO 🛑 ===
    # ===============================================
    
    # 1. ESCOLHA A FONTE DE VÍDEO:
    # MUDAR APENAS ESTA LINHA: True para Webcam, False para Vídeo
    USE_WEBCAM = True  # <--- Alterar aqui para True ou False 
    
    # 2. DEFINIÇÕES DE CAMINHO/MODELO:
    VIDEO_FILE = "Videos/people.mp4"       # Caminho do seu arquivo de vídeo
    MASK_FILE = "Assets/mask-1.png"        # Caminho da sua máscara
    MODEL_TO_USE = "yolov8n.pt"            # Use 'yolov8l.pt' para maior precisão, ou 'yolov8n.pt' para maior velocidade
    
    # 3. CONTROLE COMERCIAL DO LOG:
    # Registra uma entrada no Excel a cada X segundos.
    LOG_INTERVAL_SECONDS = 5 
    
    # 4. THRESHOLD DE CONFIANÇA:
    # Apenas detecções com confiança acima deste valor serão rastreadas.
    CONF_THRESHOLD = 0.5 

    # === INICIALIZAÇÃO ===
    
    source_to_use = "Webcam (ID 0)" if USE_WEBCAM else VIDEO_FILE
    
    try:
        contador = PeopleCounter(
            source_path=source_to_use, 
            mask_path=MASK_FILE, 
            model_path=MODEL_TO_USE, 
            conf_threshold=CONF_THRESHOLD,
            resolution=(1280, 720),
            log_interval_seconds=LOG_INTERVAL_SECONDS
        )
        
        run_local(contador, is_webcam=USE_WEBCAM)
        
    except Exception as e:
        print(f"\nOcorreu um erro fatal: {e}")