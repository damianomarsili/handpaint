import cv2
import time
import math
import mediapipe as mp
import pygame
import sys

HEIGHT = 600
WIDTH = 600
WHITE = (255,255,255)
BLACK = (0,0,0)

def main():
    # Initialize canvas and background
    pygame.init()
    WINDOW = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    pygame.draw.rect(WINDOW, WHITE, (0, 0, WIDTH, HEIGHT))
    
    # Initialize hand capture
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        
        pen = False
        prev_time = time.time()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            success, image = cap.read()
            
            # Ignore empty/failed frames
            if not success:
                print("Empty camera frame ignored")
                continue
            
            # Flip image across y-axis for selfie view and set as pass-by-value to improve performance
            image  = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = hands.process(image)
            
            # Set image back to BGR and pass-by-reference to draw hand mesh
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    #  Draw hand mesh onto webcam feed
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract fingers
                    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    ring_finger = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    
                    # Calculate thumb & ring finger distance to adjust pen on/off
                    thumb_ring_distance = math.sqrt((thumb.x - ring_finger.x)**2 + (thumb.y - ring_finger.y)**2)
                    
                    # Ensure a second has passed to avoid repeatedly turning pen on/off
                    if thumb_ring_distance <= 0.15 and time.time() - prev_time >= 1.0:
                        pen = not pen;
                        prev_time = time.time()
                    
                    # Draw to canvas if pen is on
                    if pen:
                        pygame.draw.circle(WINDOW, BLACK, (index_finger.x * WIDTH, index_finger.y * HEIGHT), 7)
            
            cv2.imshow('HandArt Tracer', image)
            pygame.display.update()
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

main()
