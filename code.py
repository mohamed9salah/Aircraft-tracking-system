# python real_time_yolo.py
import cv2
import numpy as np
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO
from gpiozero import Servo

SERVO_PIN_X = 14
SERVO_PIN_Y = 15

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN_X, GPIO.OUT)
GPIO.setup(SERVO_PIN_Y, GPIO.OUT)

servoX = GPIO.PWM(SERVO_PIN_X, 50)
servoY = GPIO.PWM(SERVO_PIN_Y, 50)

# pwm degerleri 2 ile 12 arasi olur
# servoX.start(7)
# servoY.start(4)

pwm_X = 7
pwm_Y = 4

# Load Yolo
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Configure Camera
width = 720
height = 720
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (width, height)}))
picam2.start()

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    frame = picam2.capture_array()
    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    
    
#     def set_servo_angle(angle):
#         duty = 2 + (angle / 18)
#         GPIO.output(servo_pin, True)
#         pwm.ChangeDutyCycle(duty)


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                farkX = center_x - (width / 2)  
                farkY = (height / 2) - center_y

                if farkX > 0 and pwm_X< 9 and pwm_X>3:
                    pwm_X -= 0.1
                    servoX.start(pwm_X)
                    print("Cisim sola gidiyor,pwm_X = ",pwm_X)

                
                elif farkX < 0 and pwm_X< 9 and pwm_X>3:
                    pwm_X += 0.1
                    servoX.start(pwm_X)
                    print("Cisim saga gidiyor,pxm_X =",pwm_X)
                    
                # Aşşağıdaysa : -
                # Yukarıdaysa: +
                if farkY > 0 and pwm_Y< 8 and pwm_Y>3:
                    pwm_Y -= 0.1
                    servoY.start(pwm_Y)
                    print("Cisim kalkiyor,pwm_Y = ",pwm_Y)

                    
                elif farkY < 0 and pwm_Y< 8 and pwm_Y>3:
                     pwm_Y +=0.1
                     servoY.start(pwm_Y)
                     print("Cisim iniyor,pwm_Y",pwm_Y)
                
                servoX.stop
                servoY.stop
                print("farkX:", farkX,", farkY:", farkY)
                

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]  
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            #orta nokta hesaplama
            a = x + (w/2)
            b = y + (h/2)
            cv2.circle(frame, ((int(a)), (int(b))), 1, (0, 0, 255), 6)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
    


    #Camera Middle Point
    camera_midpoint = (int(width / 2), int(height / 2))
    cv2.circle(frame, camera_midpoint, 1, (0, 255, 0), 6)
    
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
picam2.close()