# Aircraft-tracking-system
Developed aircraft tracking system
Utilized Python programming language and OpenCV computer vision library for implementing the tracking algorithms.Integrated Raspberry Pi for overall system control and management. Connected 
and synchronized the detection laser and camera with servo motors for precise tracking.Designed a 3D-printed mechanism and a tower to hold the camera apparatus, simulating an airport environment.
# Raspberry pi 4 model B
The Raspberry Pi 4 Model B provides a high-performance platform at a fraction of the cost of traditional computing hardware, making it accessible for budget-conscious projects. the Raspberry Pi 4 Model B delivers robust processing power, capable of handling data-intensive tasks such as real-time aircraft tracking and data processing. With the ability to use microSD cards and USB storage devices, the Raspberry Pi 4 Model B offers flexible storage solutions to accommodate large datasets and logs generated by the aircraft tracking system. The Raspberry Pi 4 Model B features multiple USB ports (including USB 3.0), GPIO pins, and other interfaces, providing extensive options for connecting sensors, external devices, and peripherals required for the tracking system. The Raspberry Pi 4 Model B is designed to be energy-efficient, making it suitable for continuous operation without significant power overhead, which is crucial for an always-on tracking system. according to its datasheet, required voltage is 5 v and power with 15 watt.
![WhatsApp Image 2024-06-12 at 00 32 21_26849212](https://github.com/mohamed9salah/Aircraft-tracking-system/assets/138705468/3db4eebd-d9e8-40b9-b83e-fcec4484a3f5)

# SolidWorks Design
In our project, we utilized SolidWorks to design a tower that holds camera apertures and the airport building. This design phase was crucial in ensuring that the physical components of our tracking system were optimally placed and secured for effective operation.
- Tower Design: The tower was meticulously designed to support multiple camera camera apparatus, ensuring a wide range of visibility for aircraft tracking. The structural integrity of the tower was a primary focus to withstand various environmental conditions and provide stable camera support.
 ![WhatsApp Image 2024-06-12 at 00 38 35_f7ff1cb2](https://github.com/mohamed9salah/Aircraft-tracking-system/assets/138705468/857e3573-b6aa-4284-901c-5ae2c3cc7726)
- Airport Building: The airport building was designed to integrate seamlessly with the tower, providing a centralized location for system operations and monitoring. This building houses the necessary hardware, including servers, monitors, and communication equipment, to facilitate efficient tracking and data processing.
![WhatsApp Image 2024-06-12 at 00 39 43_93d6c5f0](https://github.com/mohamed9salah/Aircraft-tracking-system/assets/138705468/10bd7a46-571b-40db-ab9e-5205ffc3c652)
- By creating a detailed simulation environment in SolidWorks, we were able to visualize and test the placement of all components within the system. This simulation helped in identifying potential design flaws, optimizing camera angles, and ensuring that all equipment was correctly positioned for maximum efficiency.
  ![WhatsApp Image 2024-06-12 at 00 41 14_f519ebe4](https://github.com/mohamed9salah/Aircraft-tracking-system/assets/138705468/5b642ad3-508b-4cd4-b363-5854d0a681de)

# Servo motors with camera apparatus and Raspberry pi  
Our aircraft tracking system employs two servo motors, each with 2-axis motion range, to control the camera apparatus. These servo motors are crucial for achieving precise and responsive camera positioning, enabling effective tracking of aircraft movements. 
- Precise Control: The servo motors provide precise control over the camera angles, allowing for accurate and smooth tracking of aircraft. This precision is essential for maintaining a clear and consistent view of the aircraft at all times.
- Wide Range of Motion: With 2-axis motion range, the servo motors offer extensive coverage, ensuring that the cameras can be adjusted to capture a wide field of view. This capability enhances the system's ability to track aircraft across various trajectories.
- Integration with Camera Apparatus: The servo motors are directly connected to the camera apparatus, allowing for seamless integration and synchronized movement. This setup ensures that the cameras move swiftly and accurately in response to the tracking system's commands.
![WhatsApp Image 2024-06-12 at 00 55 16_a81bdc11](https://github.com/mohamed9salah/Aircraft-tracking-system/assets/138705468/ea491f3b-2a41-4453-b2ef-16f204e09929)

- Hardware connection of servo motors with Raspberry pi
![image](https://github.com/mohamed9salah/Aircraft-tracking-system/assets/138705468/939f45ea-d878-4514-8325-43c54cba2af9)


# CODES TO CONTROL THE CAMERA ON RASPBERRY PI
```
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
                if farkY > 0 and pwm_Y< 8 and pwm_Y>4:
                    pwm_Y -= 0.1
                    servoY.start(pwm_Y)
                    print("Cisim kalkiyor,pwm_Y = ",pwm_Y)

                    
                elif farkY < 0 and pwm_Y< 8 and pwm_Y>4:
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

```

# Airplane detection 
![WhatsApp Image 2024-06-12 at 00 18 45_7918e933](https://github.com/mohamed9salah/Aircraft-tracking-system/assets/138705468/254153cc-7ef9-43e8-ad6f-c6a5ba0080fd)![WhatsApp Image 2024-06-12 at 00 18 47_d5ee83da](https://github.com/mohamed9salah/Aircraft-tracking-system/assets/138705468/714e870e-2306-4809-84f1-1b39eefdd49a)




















