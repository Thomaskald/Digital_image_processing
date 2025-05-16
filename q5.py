from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')

# Φόρτωση εικόνας
image = 'parking-lot.jpg'
frame = cv2.imread(image)

# Ανίχνευση αντικειμένων
results = model(frame)

# Μεταβλητή για μέτρηση των "car"
car_count = 0

# Ανάλυση αποτελεσμάτων
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label == 'car':
            car_count += 1
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Σχεδίαση bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Εμφάνιση αποτελέσματος
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 5))
plt.imshow(frame_rgb)
plt.title(f'Ανίχνευση {car_count} αυτοκινήτων')
plt.axis('off')
plt.show()
