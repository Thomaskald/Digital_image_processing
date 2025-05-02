from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Φόρτωση προεκπαιδευμένου μοντέλου YOLOv8 (nano)
model = YOLO('yolov8n.pt')

# Φόρτωση εικόνας
image = '/home/thomas/Digital_Image_Processing/DIP-project-1/DIP-project-1/images-project-1/parking-lot.jpg'
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

            # Προσθήκη αριθμού αυτοκινήτου
            cv2.putText(frame, f'car #{car_count}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Εμφάνιση αποτελέσματος
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(frame_rgb)
plt.title(f'Ανίχνευση {car_count} αυτοκινήτων με YOLOv8')
plt.axis('off')
plt.show()
