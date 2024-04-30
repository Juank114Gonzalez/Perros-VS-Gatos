import pygame as pg
import sys
from random import randrange
from scipy import signal
from tools import getAudio, melspec_lr
import numpy as np
import pyaudio
import pickle

vec2 = pg.math.Vector2


class Snake:
    def __init__(self, game):
        self.game = game
        self.size = game.TILE_SIZE
        self.rect = pg.rect.Rect([0, 0, game.TILE_SIZE - 2, game.TILE_SIZE - 2])
        self.range = self.size // 2, self.game.WINDOW_SIZE - self.size // 2, self.size
        self.rect.center = self.get_random_position()
        self.direction = vec2(0, 0)
        self.step_delay = 100  # milliseconds
        self.time = 0
        self.length = 1
        self.segments = []
        self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 1, pg.K_d: 1}

    def control(self, command):
        if command == "up" and self.directions[pg.K_w]:
            self.direction = vec2(0, -self.size)
            self.directions = {pg.K_w: 1, pg.K_s: 0, pg.K_a: 1, pg.K_d: 1}

        if command == "down" and self.directions[pg.K_s]:
            self.direction = vec2(0, self.size)
            self.directions = {pg.K_w: 0, pg.K_s: 1, pg.K_a: 1, pg.K_d: 1}

        if command == "left" and self.directions[pg.K_a]:
            self.direction = vec2(-self.size, 0)
            self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 1, pg.K_d: 0}

        if command == "right" and self.directions[pg.K_d]:
            self.direction = vec2(self.size, 0)
            self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 0, pg.K_d: 1}

    def delta_time(self):
        time_now = pg.time.get_ticks()
        if time_now - self.time > self.step_delay:
            self.time = time_now
            return True
        return False

    def get_random_position(self):
        return [randrange(*self.range), randrange(*self.range)]

    def check_borders(self):
        if self.rect.left < 0 or self.rect.right > self.game.WINDOW_SIZE:
            self.game.new_game()
        if self.rect.top < 0 or self.rect.bottom > self.game.WINDOW_SIZE:
            self.game.new_game()

    def check_food(self):
        if self.rect.center == self.game.food.rect.center:
            self.game.food.rect.center = self.get_random_position()
            self.length += 1

    def check_selfeating(self):
        if len(self.segments) != len(set(segment.center for segment in self.segments)):
            self.game.new_game()

    def move(self):
        if self.delta_time():
            self.rect.move_ip(self.direction)
            self.segments.append(self.rect.copy())
            self.segments = self.segments[-self.length:]

    def update(self):
        self.check_selfeating()
        self.check_borders()
        self.check_food()
        self.move()

    def draw(self):
        [pg.draw.rect(self.game.screen, 'green', segment) for segment in self.segments]


class Food:
    def __init__(self, game):
        self.game = game
        self.size = game.TILE_SIZE
        self.rect = pg.rect.Rect([0, 0, game.TILE_SIZE - 2, game.TILE_SIZE - 2])
        self.rect.center = self.game.snake.get_random_position()

    def draw(self):
        pg.draw.rect(self.game.screen, 'red', self.rect)


class Game:
    def __init__(self, clf, clases, p, fs, duracion, umbral):
        pg.init()
        self.clf = clf
        self.umbral = umbral
        self.clases = clases
        self.fs = fs
        self.p = p
        self.duracion = duracion
        self.WINDOW_SIZE = 1000
        self.TILE_SIZE = 50
        self.screen = pg.display.set_mode([self.WINDOW_SIZE] * 2)
        self.clock = pg.time.Clock()
        self.new_game()

    def draw_grid(self):
        [pg.draw.line(self.screen, [50] * 3, (x, 0), (x, self.WINDOW_SIZE))
                                             for x in range(0, self.WINDOW_SIZE, self.TILE_SIZE)]
        [pg.draw.line(self.screen, [50] * 3, (0, y), (self.WINDOW_SIZE, y))
                                             for y in range(0, self.WINDOW_SIZE, self.TILE_SIZE)]

    def new_game(self):
        self.snake = Snake(self)
        self.food = Food(self)

    def update(self):
        self.snake.update()
        pg.display.flip()
        self.clock.tick(60)

    def draw(self):
        self.screen.fill('black')
        self.draw_grid()
        self.food.draw()
        self.snake.draw()

    def check_event(self):
        b = signal.firwin(128, 20 / (self.fs / 2), window="hamming", pass_zero=True)
        x = getAudio(self.p, RATE=self.fs, RECORD_SECONDS=self.duracion)
        x = x / np.max(abs(x))
        s = signal.hilbert(x)
        Eh = np.abs(s)
        Eh = signal.lfilter(b, 1, Eh)
        # tomar solo las partes mayores a 0.2, xs es la senal segmentada
        xs = x[Eh > self.umbral]
        ## calcular espectro de mel
        M = int(self.fs * 0.025)
        H = int(self.fs * 0.010)
        mel = melspec_lr(xs, self.fs, hop_length=H, win_length=M, n_mels=27)
        s = np.cov(mel)
        feats = s[np.triu_indices(s.shape[0])]
        # Convert the list to a NumPy array
        my_array = np.array(feats)
        # Check if any element is NaN
        contains_nan = np.any(np.isnan(my_array))
        if not contains_nan: 
            probability = self.clf.predict_proba(feats.reshape(1, -1))
            print(probability)
            if max(probability[0])>=0.7:
                # 4. realizar prediccon usando el modelo
                prediction = self.clf.predict(feats.reshape(1, -1))
                command = self.clases[prediction][0]
                print(f"[INFO] comando reconocido: {command}")
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        sys.exit()
                    # snake control
                self.snake.control(command)
            else: 
                print("No se reconoce nigun comando")
        else: print("NaN")

    def run(self):
        while True:
            self.check_event()
            self.update()
            self.draw()


if __name__ == '__main__':
    # 1. Crear objeto de pyAudio
    p = pyaudio.PyAudio()
    fs = 16000  # Hertz
    duracion = 1  # cuantos segundos por audio?

    ##2. cargar el modelo y las etiquetas
    fh = open("clasificador.pkl", "rb")
    clf = pickle.load(fh)
    fh.close()

    # etiquetas
    fh = open("clases.pkl", "rb")
    clases = pickle.load(fh)
    fh.close()
    game = Game(clf, clases, p, fs, duracion, 0.2)
    game.run()