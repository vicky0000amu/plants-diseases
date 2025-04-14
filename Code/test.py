import threading
import queue
import multiprocessing as mp
import sys
global stop_threads
import tensorflow as tf
import pandas as pd
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
model_final = load_model("model_alexnet_best2.h5")
import predict1
import attributes

stop_threads = False

def get_connection_method():
    print("\nSelect drone connection method:")
    print("1. Connect via WiFi (DJI Tello, etc.)")
    print("2. Connect via RTSP stream")
    print("3. Connect via USB/Other")
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        # WiFi connection for common drones
        drone_type = input("Enter drone type (e.g., 'tello', 'mavic', etc.): ").lower()
        if drone_type == "tello":
            return "udp://@0.0.0.0:11111"  # Default Tello stream
        else:
            ip = input("Enter drone IP address (e.g., 192.168.10.1): ")
            return f"udp://@{ip}:11111"
    elif choice == "2":
        # RTSP stream connection
        url = input("Enter RTSP URL (e.g., rtsp://192.168.1.1/live): ")
        return url
    elif choice == "3":
        # USB or other direct connection
        cam_index = input("Enter camera index (usually 0 for default): ")
        return int(cam_index)
    else:
        print("Invalid choice, using default Tello connection")
        return "udp://@0.0.0.0:11111"

def connect_drone_camera(q, threadName, connection_param):
    try:
        if isinstance(connection_param, str):
            # Handle network connection (WiFi/RTSP)
            cap = cv2.VideoCapture(connection_param)
            # Reduce buffer size to minimize latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # Handle USB/direct connection
            cap = cv2.VideoCapture(connection_param)
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # For WiFi drones, sometimes need to send keepalive
        if isinstance(connection_param, str) and "udp://" in connection_param:
            import socket
            # Setup keepalive for Tello-like drones
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            tello_address = ('192.168.10.1', 8889)  # Common Tello address
            sock.sendto(b'command', tello_address)
            sock.sendto(b'streamon', tello_address)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from drone, retrying...")
                time.sleep(0.1)
                continue
                
            q.put(frame)
            
            if stop_threads:
                break
                
    except Exception as e:
        print(f"Drone connection error: {e}")
    finally:
        cap.release()
        if 'sock' in locals():
            sock.close()

if input('Start drone connection? (y/n): ').lower() == 'y':
    import cv2
    import time
    import numpy as np
    
    # Get connection method from user
    connection_param = get_connection_method()
    
    q = queue.LifoQueue()
    
    # Start drone camera thread
    t = threading.Thread(target=connect_drone_camera, 
                        name='drone_camera_thread', 
                        args=(q, 'drone_camera_thread', connection_param),
                        daemon=True)
    t.start()
    
    label = []
    percents = []
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            try:
                frame = q.get(timeout=2)
                with q.mutex:
                    q.queue.clear()
                
                # Resize frame for processing
                frame = cv2.resize(frame, (480, 320))
                frame_count += 1
                
                # Calculate and display FPS
                if frame_count % 10 == 0:
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display original feed
                cv2.imshow('Drone Camera Feed', frame)
                
                # Process frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('GRAY', gray)
                
                blurred = cv2.GaussianBlur(frame, (45, 45), 10)
                cv2.imshow('GaussianBlur', blurred)
                
                # Make predictions
                labels = predict1.predict1(model_final, frame)
                label.append(labels)
                
                percent = attributes.attributes(frame)
                percents.append(percent)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('Exiting...')
                    break
                    
            except queue.Empty:
                print("No frames received from drone, check connection")
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_threads = True
        t.join(timeout=1)
        cv2.destroyAllWindows()
        
        # Save collected data
        if label and percents:
            data = pd.DataFrame({'label': label, 'percent': percents})
            data.to_csv(r'datasheet_very_2nd_useful.csv', index=False, header=True)
            print("Data saved successfully")
        sys.exit(0)