import cv2
import dlib

# Carregar arquivo de shape predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Carregar imagem
image = cv2.imread('dark_paper.jpg')

# Detectar pontos de referência faciais na imagem
landmarks = predictor(image, dlib.rectangle(0, 0, image.shape[1], image.shape[0]))

# Desenhar pontos de referência na imagem
for point in landmarks.parts():
    cv2.circle(image, (point.x, point.y), 2, (0, 255, 0), -1)

# Exibir a imagem com os pontos de referência
cv2.imshow('Facial Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()