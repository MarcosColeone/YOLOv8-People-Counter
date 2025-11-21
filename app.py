import os
import time
import cv2
import numpy as np
import pandas as pd
import sqlite3 
from ultralytics import YOLO
from datetime import datetime
import sys # Necessário para o PyInstaller (sys._MEIPASS)

# ----------------------------------------------------------------------
# 3. UTILITY PARA PYINSTALLER (CORREÇÃO DE CAMINHOS)
# ----------------------------------------------------------------------

def resource_path(relative_path):
    """
    Obtém o caminho absoluto para recursos, funcionando para desenvolvimento e PyInstaller.
    Quando empacotado, usa o caminho temporário (_MEIPASS) onde os arquivos são extraídos.
    """
    try:
        # PyInstaller cria um caminho temporário e armazena o caminho em _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Caso não esteja rodando como executável (ambiente de desenvolvimento/Python)
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# ----------------------------------------------------------------------
# 1. CLASSE PRINCIPAL DO CONTADOR DE PESSOAS
# ----------------------------------------------------------------------

class PeopleCounter:
    def __init__(self, source_path, mask_path, model_path="yolov8n.pt", 
                 conf_threshold=0.5, resolution=(1280, 720), log_interval_seconds=5):
        self.SOURCE_PATH = source_path
        self.MASK_PATH = mask_path
        self.MODEL_PATH = model_path
        self.CONF_THRESHOLD = conf_threshold
        self.FRAME_WIDTH, self.FRAME_HEIGHT = resolution
        
        self.LOG_INTERVAL_SECONDS = log_interval_seconds 
        self.LAST_LOG_TIME = time.time()
        
        self.unique_ids = set() 
        self.active_ids = set() 
        self.data_log = [] 
        # Caminhos fixos (sem carimbo de data/hora)
        # NOTA: O DB e XLSX devem ser salvos no diretório de execução, 
        # então não usamos resource_path aqui.
        self.DB_PATH = os.path.join("Data", "flow_log.db") 
        self.XLSX_PATH = os.path.join("Data", "flow_log.xlsx") 
        self.start_time = time.time()
        self.frame_count = 0
        self.model = None
        self.mask = None
        self.mask_resized = None

    def load_resources(self):
        print(f"[INFO] Carregando modelo: {self.MODEL_PATH}...")
        try:
            self.model = YOLO(self.MODEL_PATH)
        except Exception as e:
            raise Exception(f"❌ Erro ao carregar o modelo YOLOv8: {e}")

        print("[INFO] Preparando máscara...")
        try:
            # self.MASK_PATH já é o caminho absoluto correto, vindo de resource_path
            self.mask = cv2.imread(self.MASK_PATH, cv2.IMREAD_GRAYSCALE)
            if self.mask is None:
                raise FileNotFoundError
        except FileNotFoundError:
            raise Exception(f"❌ Erro: Máscara não encontrada ou arquivo inválido no caminho: {self.MASK_PATH}")

        self.mask_resized = cv2.resize(self.mask, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        _, self.mask_resized = cv2.threshold(self.mask_resized, 127, 255, cv2.THRESH_BINARY)
        
    def process_frame(self, frame):
        
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
            confidences = results[0].boxes.conf.cpu().numpy() 

            # 3. Iterar sobre os resultados e desenhar
            for box, track_id, conf in zip(boxes, track_ids, confidences): 
                x1, y1, x2, y2 = box
                self.active_ids.add(track_id)
                self.unique_ids.add(track_id) 
                
                confidence_percent = f"{int(conf * 100)}%" 
                label = f"ID {track_id} ({confidence_percent})" 

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
        """Registra as métricas por intervalo de TEMPO e aciona o salvamento no DB."""
        current_time = time.time()
        
        if (current_time - self.LAST_LOG_TIME) >= self.LOG_INTERVAL_SECONDS:
            
            # Captura a data/hora LOCAL exata da máquina para o registro.
            timestamp_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            self.data_log.append({
                "timestamp": timestamp_dt,
                "Lotacao_Atual": lotacao_atual,
                "Total_Pessoas_Vistas": total_pessoas
            })
            self.LAST_LOG_TIME = current_time
            
            # Salva no DB se o buffer atingir 5 registros
            if len(self.data_log) >= 5:
                self.export_to_sqlite()


    def export_to_sqlite(self):
        """Salva os dados acumulados no buffer para o Banco de Dados SQLite e exporta para Excel."""
        if not self.data_log:
            return

        os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)
        
        df_to_save = pd.DataFrame(self.data_log)
        self.data_log = []
        
        try:
            conn = sqlite3.connect(self.DB_PATH)
            
            # 1. SALVA NO SQLITE (APPEND)
            df_to_save.to_sql('contagem_log', conn, if_exists='append', index=False)
            print(f"\n[INFO] 💾 {len(df_to_save)} registros salvos em {self.DB_PATH}")

            # 2. LÊ TODO O BANCO DE DADOS PARA GARANTIR O EXCEL COMPLETO E ORGANIZADO
            full_df = pd.read_sql_query("SELECT * FROM contagem_log", conn)
            
            conn.close()
            
            # 3. CORREÇÃO DE LEITURA (Necessário para ordenação): 
            # Força o Pandas a ler o timestamp usando o formato exato que foi salvo.
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
            
            # 4. CORREÇÃO DE ESCRITA PARA EXCEL: 
            # Formata o objeto datetime de volta para uma string ANTES de exportar.
            # Isso impede que o Excel converta a data para um número de série.
            full_df['timestamp'] = full_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 5. FORMATAÇÃO E ORGANIZAÇÃO PARA EXCEL
            full_df = full_df.rename(columns={
                'timestamp': 'Carimbo de Data/Hora (Amostra)',
                'Lotacao_Atual': 'Lotação Atual (Pessoas Ativas)',
                'Total_Pessoas_Vistas': 'Total de Pessoas Vistas (Acumulado)'
            })
            
            full_df = full_df.sort_values(by='Carimbo de Data/Hora (Amostra)')
            
            # 6. EXPORTA PARA EXCEL (SOBRESCREVE o arquivo para ter a versão mais recente)
            full_df.to_excel(self.XLSX_PATH, index=False, sheet_name='Log_Contagem')
            print(f"[INFO] 📝 Arquivo Excel atualizado: {self.XLSX_PATH}")

        except Exception as e:
            self.data_log.extend(df_to_save.to_dict('records'))
            print(f"\n[ERRO] Falha ao salvar no SQLite ou exportar para Excel. Os dados serão mantidos no buffer. Erro: {e}")


# ----------------------------------------------------------------------
# 2. EXECUÇÃO PRINCIPAL (Suporte a Vídeo/Webcam)
# ----------------------------------------------------------------------

def run_local(counter: PeopleCounter, is_webcam: bool):
    """Loop principal para exibição em tempo real e salvamento de dados."""
    
    try:
        counter.load_resources()
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {e}")
        return 

    # Se for Webcam, usa o ID 0. Se for Vídeo, usa o caminho do arquivo (self.SOURCE_PATH, já ajustado).
    source = 0 if is_webcam else counter.SOURCE_PATH
    
    try:
        cap = cv2.VideoCapture(source)
    except Exception as e:
        raise Exception(f"❌ Erro ao inicializar VideoCapture: {e}")
    
    if not cap.isOpened():
        raise Exception(f"❌ Erro ao abrir a fonte. Webcam pode estar em uso ou vídeo '{source}' não encontrado.")
        
    print(f"[INFO] Processando {'Webcam' if is_webcam else 'Vídeo'} (Pressione 'q' para fechar a janela)...")
    
    # Otimização para modo webcam
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, counter.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, counter.FRAME_HEIGHT)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Se for vídeo, sai do loop. Se for webcam, continua tentando ler.
                if not is_webcam:
                    break
                time.sleep(0.1)
                continue
                
            processed_frame = counter.process_frame(frame)
            cv2.imshow("People Counter - YOLOv8", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupção manual (Ctrl+C).")
        
    finally:
        # Finalização e Exportação FINAL
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n[INFO] ✅ Processamento finalizado.")
        
        # Salva quaisquer dados restantes no buffer ANTES de fechar
        print("[INFO] Salvando registros finais no Banco de Dados e exportando para Excel...")
        counter.export_to_sqlite() 
        print(f"[INFO] O log completo está em: {counter.DB_PATH}")
        print(f"[INFO] O relatório de fluxo está em: {counter.XLSX_PATH}")


if __name__ == '__main__':
    # ===============================================
    # === 🛑 CONFIGURAÇÕES DO PROJETO 🛑 ===
    # ===============================================
    
    # MUDAR APENAS ESTA LINHA: True para Webcam, False para Vídeo
    USE_WEBCAM = False 
    
    # DEFINIÇÕES DE CAMINHO/MODELO (AGORA USANDO a função resource_path para incluir os assets no EXE):
    VIDEO_FILE = resource_path(os.path.join("Videos", "people.mp4")) 
    MASK_FILE = resource_path(os.path.join("Assets", "mask-1.png"))  
    MODEL_TO_USE = resource_path("yolov8n.pt") 
    
    # Registra uma entrada no DB a cada X segundos.
    LOG_INTERVAL_SECONDS = 5 
    
    # Apenas detecções com confiança acima deste valor serão rastreadas.
    CONF_THRESHOLD = 0.5 

    # === INICIALIZAÇÃO ===
    
    source_to_use = "Webcam (ID 0)" if USE_WEBCAM else VIDEO_FILE
    
    try:
        print(f"==================================================")
        print(f"SOURCE: {'Webcam' if USE_WEBCAM else VIDEO_FILE}")
        print(f"INTERVALO DE LOG: {LOG_INTERVAL_SECONDS} segundos")
        
        # Adiciona um aviso sobre a data/hora para o usuário
        current_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[AVISO IMPORTANTE] O log usará a data/hora do seu sistema: {current_dt}")
        print(f"Se esta data estiver errada (ex: no futuro), corrija o relógio do seu sistema.")
        
        print(f"==================================================")
        
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
        print(f"\nOcorreu um erro fatal no programa principal: {e}")
        cv2.destroyAllWindows()