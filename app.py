from ultralytics import YOLO
import cv2
import numpy as np
import time
import pandas as pd
import math # Para arredondar confiança, se necessário (não usado na v8n, mas útil)
import os

# ----------------------------------------------------------------------
# 1. CLASSE PRINCIPAL DO CONTADOR DE PESSOAS (Mantida para modularidade)
# ----------------------------------------------------------------------

class PeopleCounter:
    """
    Contador de Pessoas modularizado usando YOLOv8 e ByteTrack.
    Adaptado para rodar em IDEs locais (VS Code) com exibição em tempo real.
    """
    def __init__(self, video_path, mask_path, output_path, model_path="yolov8n.pt", conf_threshold=0.5, resolution=(1280, 720)):
        # Configurações
        self.VIDEO_PATH = video_path
        self.MASK_PATH = mask_path
        self.OUTPUT_PATH = output_path # Será usado para salvar o Excel
        self.MODEL_PATH = model_path
        self.CONF_THRESHOLD = conf_threshold
        self.FRAME_WIDTH, self.FRAME_HEIGHT = resolution
        self.LOG_INTERVAL_FRAMES = 30 
        
        # Variáveis de Estado
        self.unique_ids = set() 
        self.active_ids = set() 
        self.data_log = []
        self.start_time = time.time()
        self.frame_count = 0
        self.model = None
        self.mask = None
        self.mask_resized = None # Máscara pré-redimensionada

    def load_resources(self):
        """Carrega o modelo e a máscara."""
        
        # Carregar modelo YOLOv8
        print(f"[INFO] Carregando modelo: {self.MODEL_PATH}...")
        self.model = YOLO(self.MODEL_PATH)
        
        # Preparar Máscara
        print("[INFO] Preparando máscara...")
        self.mask = cv2.imread(self.MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise Exception(f"❌ Erro: Máscara não encontrada no caminho: {self.MASK_PATH}")

        # Redimensiona a máscara para a resolução de processamento final
        self.mask_resized = cv2.resize(self.mask, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        # Normaliza (Garante que preto é 0 e branco é 255)
        _, self.mask_resized = cv2.threshold(self.mask_resized, 127, 255, cv2.THRESH_BINARY)
        
    def process_frame(self, frame):
        """Aplica máscara, realiza detecção/rastreamento e desenha resultados."""
        
        # 1. Redimensionar Frame
        frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        
        # 2. Aplicar Máscara ao Frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask_resized)
        
        # 3. Detecção + Rastreamento
        results = self.model.track(
            masked_frame,
            persist=True,
            conf=self.CONF_THRESHOLD,
            classes=[0],  # 0 = pessoa
            tracker="bytetrack.yaml",
            verbose=False
        )

        # 4. Atualizar Contadores
        self.active_ids.clear()
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                self.active_ids.add(track_id)
                self.unique_ids.add(track_id) 

                # Desenhar caixas e IDs
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5. Exibir Métricas
        lotacao_atual = len(self.active_ids)
        total_pessoas = len(self.unique_ids)

        cv2.putText(frame, f"Lotacao atual: {lotacao_atual}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Total de pessoas: {total_pessoas}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

        # 6. Desenhar máscara semi-transparente fora da área de detecção
        overlay = frame.copy()
        overlay[self.mask_resized == 0] = (0, 0, 0)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # 7. Log de Dados
        self._log_data(lotacao_atual, total_pessoas)

        return frame

    def _log_data(self, lotacao_atual, total_pessoas):
        """Registra as métricas de contagem em um intervalo definido."""
        if self.frame_count % self.LOG_INTERVAL_FRAMES == 0:
            current_time = time.time() - self.start_time
            self.data_log.append({
                "Tempo (s)": round(current_time, 2),
                "Lotacao Atual": lotacao_atual,
                "Total Pessoas Vistas (Acumulado)": total_pessoas
            })
        self.frame_count += 1

    def export_data(self, excel_path):
        """Salva o log de dados em um arquivo Excel."""
        if self.data_log:
            df = pd.DataFrame(self.data_log)
            df.to_excel(excel_path, index=False)
            print(f"\n[INFO] 📊 Dados de contagem salvos em: {excel_path}")

# ----------------------------------------------------------------------
# 2. EXECUÇÃO PRINCIPAL (Para IDE Local)
# ----------------------------------------------------------------------

def run_local(counter: PeopleCounter):
    """Loop principal para exibição em tempo real e salvamento de vídeo."""
    
    counter.load_resources()
    
    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture(counter.VIDEO_PATH)
    if not cap.isOpened():
        raise Exception("❌ Erro ao abrir o vídeo. Verifique o caminho e permissões.")
        
    print("[INFO] Processando vídeo (Pressione 'q' para fechar a janela)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = counter.process_frame(frame)
        
        # Exibir o frame no OpenCV
        cv2.imshow("People Counter - YOLOv8", processed_frame)
        
        # Sair do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finalização
    cap.release()
    cv2.destroyAllWindows()
    
    print("[INFO] ✅ Processamento finalizado.")
    
    # Exportação de Dados
    excel_filename = f"registro_contagem_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    excel_path_full = os.path.join("Data", excel_filename)
    
    # Verifica se a pasta 'Data' existe antes de tentar salvar
    if not os.path.exists("Data"):
        os.makedirs("Data") # Cria a pasta se ela não existir

    counter.export_data(excel_path_full)


if __name__ == '__main__':
    # === CONFIGURAÇÕES DO PROJETO ===
    
    # Use caminhos relativos ao seu projeto no VS Code!
    # Crie as pastas 'Videos', 'Assets' etc.
    VIDEO_FILE = "Videos/people.mp4" 
    MASK_FILE = "Assets/mask-1.png"
    # O output_path agora é apenas um placeholder, o nome do arquivo Excel será gerado com timestamp
    OUTPUT_PLACEHOLDER = "Data/output_data.xlsx" 

    # Crie as pastas necessárias se elas não existirem!
    
    try:
        contador = PeopleCounter(
            video_path=VIDEO_FILE, 
            mask_path=MASK_FILE, 
            output_path=OUTPUT_PLACEHOLDER,
            model_path="yolov8n.pt", 
            conf_threshold=0.5,
            resolution=(1280, 720) # Mantenha a mesma resolução para consistência visual
        )
        
        run_local(contador)
        
    except Exception as e:
        print(f"\nOcorreu um erro fatal: {e}")