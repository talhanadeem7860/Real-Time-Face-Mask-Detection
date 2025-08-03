import tensorflow as tf
import numpy as np
import cv2

# --- Load the Models ---
print("[INFO] Loading face detector model...")
face_net = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

print("[INFO] Loading face mask classifier model...")
mask_net = tf.keras.models.load_model("mask_classifier.model")

# --- Initialize Video Stream ---
print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(0)

# --- Define Classes ---
# Match the order from the training generator
CLASSES = ["WithMask", "WithoutMask"] # Adjust if you trained with 3 classes

# --- Process Video Stream ---
while True:
    # Grab the frame from the threaded video stream and resize it
    ret, frame = vs.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than 0.5
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI, convert it from BGR to RGB channel ordering,
            # resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                # Pass the face through the model to determine if the face has a mask or not
                preds = mask_net.predict(face)
                
            
                # Assuming 2 classes from the generator for simplicity here:
                (mask, withoutMask) = preds[0]
                
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # Include the probability in the label
                label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

                # Display the label and bounding box rectangle on the output frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
vs.release()