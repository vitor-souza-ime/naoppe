import qi
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Configuração da sessão com o NAO
NAO_IP = "172.15.1.29"   # altere para o IP do seu NAO
NAO_PORT = 9559

session = qi.Session()
try:
    session.connect(f"tcp://{NAO_IP}:{NAO_PORT}")
    print("Conectado ao NAO!")
except RuntimeError:
    print("Erro: não foi possível conectar ao NAO.")
    exit(1)

# Serviços do NAO
video_service = session.service("ALVideoDevice")
tts_service = session.service("ALTextToSpeech")

# Inscrição na câmera do NAO (640x480, RGB)
subscriber_id = video_service.subscribeCamera(
    "camera_ppe", 0, 3, 11, 30
)

# Carregar modelo YOLO treinado
model_path = "/home/ime/Documentos/VA/Exemplos/ROS2ex/PPE/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

print("Modelo YOLO carregado com sucesso!")
print("Pressione 'q' para sair.")

# Variável para controlar o tempo
last_detection_time = 0
detection_interval = 2  # segundos

# Variáveis para armazenar último resultado
last_result_img = None
last_detections = []


# Obter serviço de movimento
motion_service = session.service("ALMotion")

# Deixar o corpo todo mole
motion_service.setStiffnesses("Body", 0.0)

while True:
    # Capturar imagem da câmera do NAO
    nao_image = video_service.getImageRemote(subscriber_id)
    
    if nao_image is None:
        continue
    
    # Extrair informações da imagem
    width, height = nao_image[0], nao_image[1]
    array = nao_image[6]
    
    try:
        frame = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))
    except:
        continue
    
    # Converter BGR para RGB para exibição correta
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Verificar se deve fazer nova detecção (a cada 5 segundos)
    current_time = time.time()
    if current_time - last_detection_time >= detection_interval:
        print(f"\n[{time.strftime('%H:%M:%S')}] Executando detecção de EPIs...")
        
        t1=time.time()
        # Fazer inferência com YOLO
        results = model.predict(source=frame_rgb, conf=0.3, verbose=False)
        tf=time.time()-t1

        # Processar resultados
        last_detections = []
        if len(results) > 0 and results[0].boxes is not None:
            for box, conf, cls in zip(results[0].boxes.xyxy, 
                                     results[0].boxes.conf, 
                                     results[0].boxes.cls):
                class_name = model.names[int(cls)]
                confidence = float(conf)
                last_detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': box.cpu().numpy()
                })
                print(f"  - {class_name}: {confidence:.2%}")
            
            print(f"  - Tempo: {tf:.2f} s")

            # Obter imagem com bounding boxes
            last_result_img = results[0].plot()  # BGR
            
            # Anunciar EPIs detectados
            if last_detections:
                epis_detectados = ", ".join([d['class'] for d in last_detections])
                #tts_service.say(f"EPIs detectados: {epis_detectados}")
            else:
                #tts_service.say("Nenhum EPI detectado")
                print("  Nenhum EPI detectado")
        else:
            print("  Nenhum EPI detectado")
            last_result_img = frame_rgb.copy()
            tts_service.say("Nenhum EPI detectado")
        
        last_detection_time = current_time
    
    # Exibir resultado (usa o último resultado disponível)
    if last_result_img is not None:
        display_frame = last_result_img.copy()
        
        # Adicionar informações na tela
        time_since_last = int(current_time - last_detection_time)
        next_detection = detection_interval - time_since_last
        
        cv2.putText(display_frame, 
                   f"Next detection in: {next_detection}s", 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(display_frame, 
                   f"PPE detected: {len(last_detections)}", 
                   (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("PPE Detection - NAO", display_frame)
    else:
        # Se ainda não há resultado, mostrar frame atual
        cv2.imshow("PPE Detection - NAO", frame_rgb)
    
    # Verificar se usuário quer sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalizar
print("\nFinalizando...")
video_service.unsubscribe(subscriber_id)
cv2.destroyAllWindows()
print("Finalizado!")
