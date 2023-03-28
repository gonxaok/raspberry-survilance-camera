import cv2
import os
import telebot
from time import sleep


TOKEN = 'TOKEN'
CHAT_ID = 'CHAT_ID'

bot = telebot.TeleBot(TOKEN)


img_folder = './images/'


known_faces = []
for file in os.listdir(img_folder):
    img_path = os.path.join(img_folder, file)
    known_faces.append(cv2.imread(img_path))


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al abrir la cámara.")
        break
    
   
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
 
    for (x, y, w, h) in faces:
      
        face = frame[y:y+h, x:x+w]
    
        face_found = False
        for known_face in known_faces:
          
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray_known_face = cv2.cvtColor(known_face, cv2.COLOR_BGR2GRAY)
          
            result = cv2.matchTemplate(gray_face, gray_known_face, cv2.TM_CCOEFF_NORMED)
            if result > 0.6:
                face_found = True
                break
        
        
        if not face_found:
            cv2.imwrite('unknown_face.jpg', face)
            with open('unknown_face.jpg', 'rb') as photo:
                bot.send_photo(chat_id=CHAT_ID, photo=photo)
                print("Enviando imagen...")
            os.remove('unknown_face.jpg')
    
   
    sleep(1)

cap.release()
cv2.destroyAllWindows()
