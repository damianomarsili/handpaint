import cv2
import time
import math
import mediapipe as mp
import pygame
import sys

HEIGHT = 600
WIDTH = 600
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Circle:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Canvas:
    
    def __init__(self, surface):
        self.surface = surface
        self.circles = []
        self.surface.fill(WHITE)
        pygame.display.update()

    def draw_circles(self):
        for circle in self.circles:
            pygame.draw.circle(self.surface, BLACK, (circle.x, circle.y), 7)

    def draw_cursor(self, x, y):
        pygame.draw.circle(self.surface, RED, (x, y), 4)

    def draw_canvas(self, cursor_x, cursor_y):
        self.surface.fill(WHITE)
        self.draw_circles()
        self.draw_cursor(cursor_x, cursor_y)
        pygame.display.update()

class Game:
    
    def __init__(self):
        #Initialize canvas and background
        pygame.init()
        surface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        self.canvas = Canvas(surface)

    def main(self):
        canvas = self.canvas
        
        #Initialize hand capture
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            
            pen = False
            prev_press_time = time.time()

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
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = hands.process(image)

                # Set image back to BGR and pass-by-reference to draw hand mesh
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand mesh onto webcam feed
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Extract fingers
                        index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        ring_finger = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                        
                        # Add circles to canvas
                        canvas.circles.append(Circle(index_finger.x * WIDTH, index_finger.y * HEIGHT))
                        canvas.draw_canvas(index_finger.x * WIDTH, index_finger.y * HEIGHT)
                        
                cv2.imshow('HandPaint Tracker', image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()

game = Game()
game.main()


