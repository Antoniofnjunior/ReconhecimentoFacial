import cv2
Algoritimo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') #carreguei o algoritimo treinado

img = cv2.imread('./Fotos/foto2.png.')#carreguei a iamgem

#trasnformando a imagem em cinza para reconhecer padroes

cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = Algoritimo.detectMultiScale(cinza)

print(faces)

for(x, y, l, a) in faces:#identificando os pontos
    cv2.rectangle(img, (x, y), (x + l, y + a), (0, 0, 255), 2)

cv2.imshow("Faces", img)
cv2.waitKey()
