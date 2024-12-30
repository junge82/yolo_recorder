##command


#ffmpeg -re -i "13.mp4" -c:v copy -c:a aac -ar 44100 -ac 1 -f flv rtmp://127.0.0.1/live/stream
#/usr/local/nginx/conf/nginx.conf
#$ sudo /usr/local/nginx/sbin/nginx -s stop
#$ sudo /usr/local/nginx/sbin/nginx

import cv2
import numpy as np
import torch
import platform
from torch.amp import autocast
import subprocess as sp 
import glob
import os
import argparse  
import random  
import string  
from datetime import datetime  
  
def generate_random_string(length=10):  
    letters = string.ascii_letters + string.digits  
    result_str = ''.join(random.choice(letters) for i in range(length))  
    return result_str  
  
def get_current_timestamp():  
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')  
  
random_string = generate_random_string()  
timestamp = get_current_timestamp()  
  
result = f"{random_string}_{timestamp}"  

img_width = 1280
img_height = 1024
ffmpeg_cmd = "/bin/ffmpeg"
ffplay_cmd = "/bin/ffplay"  # May use full path like: 'c:\\FFmpeg\\bin\\ffplay.exe'
output_file = f'/mnt/SSD/yolo/{result}.mp4'

# Create 10 synthetic JPEG images for testing (image0001.jpg, image0002.jpg, ..., image0010.jpg).
#sp.run([ffmpeg_cmd, '-y', '-f', 'lavfi', '-i', f'testsrc=size={img_width}x{img_height}:rate=1:duration=10', 'image%04d.jpg'])

fps = 25
rtsp_server = 'rtsp://localhost:31415/live.stream'

# You will need to start the server up first, before the sending client (when using TCP). See: https://trac.ffmpeg.org/wiki/StreamingGuide#Pointtopointstreaming
#ffplay_process = sp.Popen([ffplay_cmd, '-rtsp_flags', 'listen', rtsp_server])  # Use FFplay sub-process for receiving the RTSP video.


def buildFFmpegCommand(file, file_name):
    output_str = "imgs/{}_%06d.png".format(file_name)
    final_user_input = {"input_file": file.file_name,
                        "output_file": output_str,
                        "encoding_speed": 'fast'
                        }

    commands_list = [
        ffmpeg_cmd,
        "-i",
        final_user_input["input_file"],
        "-preset",
        final_user_input["encoding_speed"],
        final_user_input["output_file"]
        ]

    return commands_list





        # Wait until a key is pressed and close the window  
        #cv2.waitKey(0)  
        #cv2.destroyAllWindows()

def inference(process):
    try:
        index = 0 # capture device index
        cap = cv2.VideoCapture(index)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Load a YOLOv5 model (change 'yolov5s' to 'yolov5m', 'yolov5l', or 'yolov5x' to use other variants)  
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print(f"Error during capturing")
                break   
        
            print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))
            with torch.amp.autocast(device_type="cuda"):  
                results = model(frame)        
                # Results
                results.print()

                detections = results.pandas().xyxy[0]  # or results.xyxy[0] depending on your version  
    
                for _, row in detections.iterrows():  
                    x1, y1, x2, y2, conf, cls = row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence'], row['class']  
                    print(x1, y1, x2, y2, conf, cls)
                    label = f'{model.names[int(cls)]}: {conf:.2f}'
                    print('label',label)
                    start_point = (int(x1), int(y1))
                    end_point = (int(x2), int(y2))
                    cv2.rectangle(frame, start_point, end_point,(0, 0, 255), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    process.stdin.write(frame.tobytes())  # Write raw frame to stdin pipe.



    except KeyboardInterrupt:  
        print("Terminating due to KeyboardInterrupt")  
  
    finally:  
    # Properly close FFmpeg subprocess  
        if process.stdin:  
            process.stdin.close()  # Close stdin pipe  
        process.wait()  # Wait for FFmpeg sub-process to finish
        #ffplay_process.kill()  # Forcefully close FFplay sub-process
  
    # Release OpenCV resources  
        cap.release()  
        cv2.destroyAllWindows()  
  
        print("Resources have been released.")


  
def main():  
    # Create the parser  
    parser = argparse.ArgumentParser(description="A simple example script")  
  
    # Add arguments  
    parser.add_argument('--filename', type=str, help='The filename to process')  
    parser.add_argument('--verbose', '-v', action='store_true', help='Increase output verbosity')  
  
    # Parse the arguments  
    args = parser.parse_args()  
  
    # Use the arguments  
    if args.verbose:  
        print(f"Verbose mode is on.")
    if args.filename:
        print(f"Processing file: {args.filename}")

    command = [ffmpeg_cmd,
           '-re',
           '-f', 'rawvideo',  # Apply raw video as input - it's more efficient than encoding each frame to PNG
           '-s', f'{img_width}x{img_height}',
           '-pixel_format', 'bgr24',
           '-r', f'{fps}',
           '-i', '-',
           output_file,
           '-pix_fmt', 'yuv420p',
           '-c:v', 'libx264',
           '-bufsize', '64M',
           '-maxrate', '4M'
           #'-rtsp_transport', 'tcp',
           #'-f', 'rtsp',
           #'-muxdelay', '0.1',
           #rtsp_server
        ]
#ffmpeg -re -i "13.mp4" -c:v copy -c:a aac -ar 44100 -ac 1 -f flv rtmp://127.0.0.1/live/stream
    process = sp.Popen(command, stdin=sp.PIPE)  # Execute FFmpeg sub-process for RTSP streaming    

    inference(process)
  
if __name__ == '__main__':  
    main()