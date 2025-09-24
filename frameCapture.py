import cv2
import time
import os

# create folder for saved images if doesn't exist
saveFolder = "calibrationFrames"
os.makedirs(saveFolder, exist_ok = True)

# opens webcam
capture = cv2.VideoCapture(0)

# checks for camera connection
if not capture.isOpened():
    print("Cannot open camera")
    exit()

# initializes variable that tracks number of frames saved
frameCount = 0

# gives users instructions for how the process operates
print("Press:")
print("  'j' : save image as JPG")
print("  'p' : save image as PNG")
print("  'e' : to end process")

# runs while camera is active
while True:
    returnVal, frame = capture.read()

    # ends program if the returned value doesn't exist
    if not returnVal:
        print("Cannot get frame.. Closing...")
        break

    # shows frame
    cv2.imshow('Webcam video', frame)

    # allows for input from user
    key = cv2.waitKey(1) & 0xFF
    
    # saves frame as JPG if user inputs 'j'
    if key == ord('j'):
        filename = f"{saveFolder}/calibrationFrame_{frameCount}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved JPG: {filename}")
        frameCount += 1
    
    # saves frame as PNG if user inputs 'p'
    elif key == ord('p'):
        filename = f"{saveFolder}/frame_{timestamp}_{frameCount}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved PNG: {filename}")
        frameCount += 1
    
    # breaks while loop
    elif key == ord('e'):
        break
        

# end program
capture.release()
cv2.destroyAllWindows()