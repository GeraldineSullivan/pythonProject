# https://www.youtube.com/watch?v=bwNgyjQr6H4 Bar Visualiser
# https://www.youtube.com/watch?v=xL6cEdUGirE&t=1179s Drawing the bars

import pygame
import random
import numpy as np
import librosa
import librosa.feature
import librosa.display
import matplotlib.pyplot as plt
import math
from io import BytesIO

pygame.init()

# Setting Variables for the Pygame Window
WIDTH = 1200
HEIGHT = 600
colourA = (0, 0, 0)
colourB = (255, 202, 96)
colourC = (255, 255, 255)

# Create buttons for Starting the Visualiser and Exiting the Application
# Added button for Mel Spectrogram
button_default = pygame.image.load("ButtonBlack.png")
button_hover = pygame.image.load("ButtonHover.png")
button_quit = pygame.image.load("ButtonQuit.png")
button_spectro = pygame.image.load("ButtonSpectro.png")
start_button = button_default.get_rect(center=(WIDTH // 2, HEIGHT // 2))
quit_button = button_quit.get_rect(center=(1050, 550))
spectro_button = button_spectro.get_rect(center=(150, 550))


''' Loading mp3 file and extracting the audio data using librosa library.
The short time fourier tranform(stft) and the fast fourier transform(fft) are both used here.
The value n_fft = 2048 is a default value in librosa documentation as is hop_length = 512,
and we convert our amplitude spectrogram to decibels as the amplitude array was quite difficult to work with
'''
filename = 'Mischief.wav'
y, sr = librosa.load(filename, sr=None)
n_fft = 2048*2
time_series, sample_rate = librosa.load(filename)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=n_fft))
spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=n_fft)
seconds = librosa.core.frames_to_time(np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=n_fft)
seconds_index_ratio = len(seconds) / seconds[len(seconds) - 1]
fft_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


# This function converts the audio data to decibels
def get_decibel(target_time, freq):
    return spectrogram[int(freq * fft_index_ratio)][int(target_time * seconds_index_ratio)]


# This class updates our moving circle using frequency data
class CircleVisualizer:

    def __init__(self, coord_x, coord_y, frequency, colour, start_radius=150, final_radius=300,start_decibel=-80,limit_decibel=0):
            self.coord_x,self.coord_y = coord_x, coord_y
            self.frequency = frequency
            self.colour = colour
            self.start_radius,self.final_radius =start_radius,final_radius
            self.radius = start_radius
            self.start_decibel,self.limit_decibel = start_decibel,limit_decibel
            self.decibel_radius_ratio = (final_radius - start_radius)/ (limit_decibel - start_decibel)

    def  change(self,et,decibel):
         ideal_radius = decibel * self.decibel_radius_ratio + self.final_radius
         speed = (ideal_radius - self.radius) /0.2
         self.radius += speed * et * 7
         self.radius = max(self.start_radius, min(self.final_radius,self.radius))
         self.color = colourC

    def draw(self,display):
        pygame.draw.circle(display,self.colour,(int(self.coord_x), int(self.coord_y)),int(self.radius))


# Setting up variables for our bar array
n = 120
array = random.sample(range(1, n+1), n)

# This function draws the bars and changes the bars heights based on the decibels
def draw_bars(display, default_height = 5):
    n = len(array)
    bar_width = WIDTH / n

    for i in range(n):
        decibel = get_decibel(pygame.mixer.music.get_pos() / 1000.0, frequencies[i])
        bar_height = default_height * ((decibel + 100)/2)
        x = bar_width * i
        y = HEIGHT / 2 - bar_height / 2
        bar = (x, y, bar_width, bar_height)
        pygame.draw.rect(display, colourB, bar)


# This function draws the decorative sine waves on the pygame display
def draw_waves(display):
    plotPointsA = []
    for x in range(0, WIDTH):
        y = int(math.sin(x/WIDTH * 4 * math.pi) * 200 + HEIGHT/2)
        plotPointsA.append([x, y])

    pygame.draw.lines(display, [255, 255, 255], False, plotPointsA, 10)

    plotPointsB = []
    for x in range(0, WIDTH):
        y = int(-math.sin((x+160)/WIDTH * 4 * math.pi) * 200 + HEIGHT/2)
        plotPointsB.append([x, y])

    pygame.draw.lines(display, [255, 255, 255], False, plotPointsB, 4)

    plotPointsC = []
    for x in range(0, WIDTH):
        y = int(math.sin((x+160) / WIDTH * 4 * math.pi) * 200 + HEIGHT / 2)
        plotPointsC.append([x, y])

    pygame.draw.lines(display, [255, 255, 255], False, plotPointsC, 4)

    plotPointsD = []
    for x in range(0, WIDTH):
        y = int(-math.sin(x / WIDTH * 4 * math.pi) * 200 + HEIGHT / 2)
        plotPointsD.append([x, y])

    pygame.draw.lines(display, [255, 255, 255], False, plotPointsD, 10 )

# this function is for displaying the Mel Spectrogram
def display_spectrogram(filename):
    # load our audio file using librosa
    y, sr = librosa.load(filename)

    # librosa.feature is used to extract the mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048)
    # convert the power spectrogram to decibels
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    # creates a byte buffer to store binary image data for our mel spectrogram
    img_buff = BytesIO()

    # matplotlib is used here (plt) to  get a visual representation (plt.figure) of the
    # Mel Spectrogram using librosa.display.specshow, plt.colour bar to add a colour bar,
    # plt.savefig to store image from BitesIO. It treats binary image data like a file.
    
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel', sr=sr, hop_length=512, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.savefig(img_buff, format='png')
    plt.close()
    img_buff.seek(0)
    return img_buff

# Plays the MP3 File in the background and count the ticks
pygame.mixer.init()
ticks = pygame.time.get_ticks()
get_ticks_last_frame = ticks
pygame.mixer.music.load(filename)


# Set up Visualiser Window
pygame.display.init()
display = pygame.display.set_mode((WIDTH, HEIGHT))
display.fill(colourA)
pygame.display.flip()
pygame.display.set_caption('SAMHLÚ')
display.blit(button_spectro, spectro_button)


# Variables for rendering elements in main loop
circle_visualizer = CircleVisualizer(WIDTH // 2, HEIGHT // 2, 1000, (255, 255, 255))
logo = pygame.image.load('Samhlu_Logo.png')


# Pygame main Application Loop
running = True
# setting a flag for whether spectrogram will be shown
show_spectrogram = False


while running:
    ticks = pygame.time.get_ticks()
    elapsed_time = (ticks - get_ticks_last_frame) / 1000.0
    get_ticks_last_frame = ticks

    #This section of the loop checks for the start button event to start the visualiser
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if start_button.collidepoint(event.pos):
                pygame.mixer.music.play()

            elif spectro_button.collidepoint(event.pos):
                # this should show or hide the spectrogram
                show_spectrogram=not show_spectrogram


            # This section of the loop updates the display of the visualiser
            while running:
                    pygame.display.update()
                    display.fill(colourA)
                    draw_waves(display)
                    draw_bars(display)
                    circle_visualizer.change(elapsed_time, get_decibel(pygame.mixer.music.get_pos() / 1000.0, circle_visualizer.frequency))
                    circle_visualizer.draw(display)
                    display.blit(logo, (487, 250))
                    display.blit(button_quit, quit_button)

                    # if the boolean flag is set, this will show the spectrogram
                    if show_spectrogram:
                        spectro_pic=display_spectrogram(filename)
                        spectro_show=pygame.image.load_extended(spectro_pic)
                        display.blit(spectro_show, (15, 15))

                    pygame.display.flip()


                    # This section of the loop reads for the quit button event
                    for event_inner in pygame.event.get():
                        if event_inner.type == pygame.QUIT:
                            running = False
                        elif event_inner.type == pygame.MOUSEBUTTONDOWN and event_inner.button == 1:
                            if quit_button.collidepoint(event_inner.pos):
                                running = False

# This part of the loop renders the start screen of the pygame application
    if start_button.collidepoint(pygame.mouse.get_pos()):
        pygame.display.update()
        draw_waves(display)
        pygame.draw.circle(display, colourC, (600,300), 170)
        display.blit(button_hover, start_button)
    else:
        pygame.display.update()
        draw_waves(display)
        pygame.draw.circle(display, colourC, (600, 300), 170)
        display.blit(button_default, start_button)

pygame.quit()
