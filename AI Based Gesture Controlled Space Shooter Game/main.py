import cv2
import mediapipe as mp
import pygame
import random
import math

pygame.init()

WIDTH = 800
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Space Shooter")

# Load Images
background = pygame.image.load("assets/space background.png")
background = pygame.transform.scale(background,(WIDTH,HEIGHT))

player_img = pygame.image.load("assets/spaceship.png").convert_alpha()
player_img = pygame.transform.scale(player_img,(120,120))

enemy_img = pygame.image.load("assets/enemy.png").convert_alpha()
enemy_img = pygame.transform.scale(enemy_img,(100,100))

bullet_img = pygame.image.load("assets/bullet.png").convert_alpha()
bullet_img = pygame.transform.scale(bullet_img,(25,50))

explosion_img = pygame.image.load("assets/explosion.png").convert_alpha()
explosion_img = pygame.transform.scale(explosion_img,(120,120))

# Sounds
laser_sound = pygame.mixer.Sound("assets/laser.wav")
explosion_sound = pygame.mixer.Sound("assets/explosion.mp3")

# Player
player_x = WIDTH//2
player_y = 450
player_speed = 12
player_health = 3

# Bullet
bullet_x = 0
bullet_y = player_y
bullet_state = "ready"
bullet_speed = 15

# Enemy list
enemy_list = []
enemy_health = []

for i in range(4):
    enemy_list.append([random.randint(0,WIDTH-100), random.randint(20,150)])
    enemy_health.append(3)

enemy_speed = 3

score = 0
high_score = 0
game_over = False

font = pygame.font.SysFont("Arial",30)

# Camera
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw = mp.solutions.drawing_utils


def shoot_bullet(x,y):
    screen.blit(bullet_img,(x+45,y))


def show_score():
    score_text = font.render("Score: "+str(score),True,(255,255,255))
    screen.blit(score_text,(10,10))

    high_text = font.render("High Score: "+str(high_score),True,(255,255,0))
    screen.blit(high_text,(10,40))


def show_health():
    health_text = font.render("Health: "+str(player_health),True,(255,0,0))
    screen.blit(health_text,(650,10))


def show_game_over():
    over_text = font.render("GAME OVER",True,(255,0,0))
    screen.blit(over_text,(WIDTH//2-100,HEIGHT//2))

    restart_text = font.render("Press R to Restart",True,(255,255,255))
    screen.blit(restart_text,(WIDTH//2-120,HEIGHT//2+40))


def collision(ex,ey,bx,by):
    dist = math.sqrt((ex-bx)**2+(ey-by)**2)
    return dist < 60


running = True

while running:

    screen.blit(background,(0,0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if game_over:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    running = False

    success,img = cap.read()
    if not success:
        continue

    img = cv2.flip(img,1)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    shoot = False

    if result.multi_hand_landmarks and not game_over:

        for hand in result.multi_hand_landmarks:

            lm = hand.landmark

            h,w,c = img.shape

            index_x = int(lm[8].x*w)
            thumb_x = int(lm[4].x*w)

            # Move player
            if index_x < 250:
                player_x -= player_speed

            elif index_x > 400:
                player_x += player_speed

            # Pinch shoot
            pinch_distance = abs(index_x-thumb_x)

            if pinch_distance < 40:
                shoot = True

            draw.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)

    player_x = max(0,min(player_x,WIDTH-120))

    # Bullet logic
    if shoot and bullet_state == "ready" and not game_over:
        bullet_x = player_x
        bullet_state = "fire"
        laser_sound.play()

    if bullet_state == "fire":
        shoot_bullet(bullet_x,bullet_y)
        bullet_y -= bullet_speed

    if bullet_y < 0:
        bullet_y = player_y
        bullet_state = "ready"

    # Enemy loop
    if not game_over:

        for i in range(len(enemy_list)):

            enemy_x,enemy_y = enemy_list[i]

            enemy_y += enemy_speed

            if enemy_y > HEIGHT:
                enemy_y = 50
                enemy_x = random.randint(0,WIDTH-100)

            # Player hit detection
            if abs(enemy_x-player_x) < 70 and abs(enemy_y-player_y) < 70:
                player_health -= 1
                enemy_y = 50
                enemy_x = random.randint(0,WIDTH-100)

                if player_health <= 0:
                    game_over = True

            # Bullet collision
            if collision(enemy_x,enemy_y,bullet_x,bullet_y):

                bullet_y = player_y
                bullet_state = "ready"

                enemy_health[i] -= 1

                if enemy_health[i] <= 0:

                    screen.blit(explosion_img,(enemy_x,enemy_y))
                    pygame.display.update()
                    pygame.time.delay(120)

                    enemy_x = random.randint(0,WIDTH-100)
                    enemy_y = 50

                    enemy_health[i] = 3

                    score += 1
                    explosion_sound.play()

            # Health bar
            pygame.draw.rect(screen,(255,0,0),(enemy_x,enemy_y-10,100,5))
            pygame.draw.rect(screen,(0,255,0),(enemy_x,enemy_y-10,100*(enemy_health[i]/3),5))

            screen.blit(enemy_img,(enemy_x,enemy_y))

            enemy_list[i] = [enemy_x,enemy_y]

    if score > high_score:
        high_score = score

    if not game_over:
        screen.blit(player_img,(player_x,player_y))
    else:
        show_game_over()

    show_score()
    show_health()

    pygame.display.update()

    cv2.imshow("Camera",img)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

pygame.quit()
