import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import lfilter
from scipy.signal import hilbert
import random
import pyaudio
from helpAudio import byte_to_float
from wav_rw import wavwrite
import os
from glob import glob
from scipy.signal import get_window
import librosa


##############
def calFeatures(x, fs, M, H, NFFT):
    xf, _ = enframe(x, window="hamming", M=M, H=H)
    Fc = []
    HH = []
    Er = []
    F1 = []
    F2 = []
    # bandas
    band1 = np.zeros(NFFT // 2 + 1)
    band1[0:50] = 1
    band2 = np.zeros(NFFT // 2 + 1)
    band2[50:100] = 1

    for frame in xf:
        X = np.abs(np.fft.rfft(frame, NFFT) / NFFT) ** 2
        # calcular centroide
        f = np.linspace(0, fs / 2, X.size)
        fc = (X * f).sum() / X.sum()
        Fc.append(fc)
        # calcular entropia
        P = X
        p = P / sum(P)
        h = -1 * (p * np.log(p)).sum()
        HH.append(h)
        # calcular relacion de energia
        E1 = np.dot(band1, X)
        E2 = np.dot(band2, X)
        Er.append(E2 / E1)
        # formantes
        a, g = solve_lpc(frame, 10)
        a = np.hstack(([1], -a))
        r = np.roots(a)
        angle = np.angle(r)
        angle = np.sort(angle[angle > 0])
        F1.append(angle[0])
        F2.append(angle[1])

    # calcular promedios
    Fc = np.mean(Fc)
    HH = np.mean(HH)
    Er = np.mean(Er)
    F1 = np.mean(F1)
    F2 = np.mean(F2)

    return Fc, HH, Er, F1, F2


########################


########################## get melspectrogram
def melspec_lr(s, fs, hop_length=100, win_length=400, n_mels=27, n_fft=512):
    # s   = librosa.effects.preemphasis(s)
    S = librosa.feature.melspectrogram(
        y=s,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
    )
    feats = librosa.power_to_db(S, ref=np.max)
    feats_norm = (
        (feats.T - feats.mean(axis=1)) / (feats.std(axis=1) + np.finfo(float).eps)
    ).T
    return feats_norm


######### configurar py audio
# funcion que abre el microfono, graba durante RECORD_SECONDS y
# retorno la muestras del audio
# CHUNK tamano del buffer
# FORMAT = pyaudio.paInt16
# CHANNELS = cuantos canales?
# RATE  frecuencia de muestreo
# RECORD_SECONDS duracion total en segundos


def getAudio(
    objPyaudio,
    CHUNK=1024,
    FORMAT=pyaudio.paInt16,
    CHANNELS=1,
    RATE=16000,
    RECORD_SECONDS=3,
):

    stream = objPyaudio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print(
        f"[INFO] Escuchando  durante {RECORD_SECONDS} segundos..."
    )  # cuando vea el mensaje debe hablar
    audio = []
    # concatenar los CHUNK que se reciben
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        data = byte_to_float(data)
        audio.append(data)
    print("[INFO] finalizado!")  # cuando vea el mensaje debe dejar de hablar
    audio = np.hstack(audio)
    # detener y cerrar el stream
    stream.stop_stream()
    stream.close()

    return audio


#######################ENFRAME
def enframe(x=None, window="rectangular", M=512, H=512):
    """
    retorna una matriz de ventanas en tiempo corto usando M como longitud
    y H para incrementos
    """
    if x is None:  # raise error if there is no input signal
        raise ValueError("there is no input signal")
    x = np.squeeze(x)
    w = get_window(window, M)  # compute analysis window
    # w    = w / sum(w)                       # normalize analysis window
    # W=W/sqrt(sum(W(1:INC:NW).^2));      % normalize window
    if x.ndim != 1:
        raise TypeError("enframe input must be a 1-dimensional array.")
    n_frames = 1 + int(np.floor((len(x) - M) / float(H)))
    xf = np.zeros((n_frames, M))
    for ii in range(n_frames):
        xf[ii] = x[ii * H : ii * H + M] * w
    next_w = (ii + 1) * H
    excess = np.array([])
    if next_w < len(x):
        excess = x[next_w:]
    return xf, excess


####################LPC###############
def add_overlapping_blocks(B, H):
    [count, nw] = B.shape
    step = H

    n = (count - 1) * step + nw

    x = np.zeros((n,))

    for i in range(count):
        offset = i * step
        x[offset : nw + offset] += B[i, :]

    return x


#################### usando la senal x, generar una matriz para el sistema de ecuaciones
def make_matrix_X(x, p):
    n = len(x)
    # [x_n, ..., x_1, 0, ..., 0]
    xz = np.concatenate([x[::-1], np.zeros(p)])

    X = np.zeros((n - 1, p))
    for i in range(n - 1):
        offset = n - 1 - i
        X[i, :] = xz[offset : offset + p]
    return X


#################### resolver y retornar los coeficientes y la varianza del ruido
def solve_lpc(x, p):
    b = x[1:]

    X = make_matrix_X(x, p)

    a = np.linalg.lstsq(X, b.T, rcond=None)[0]

    e = b - np.dot(X, a)
    g = np.var(e)

    return [a, g]


def run_source_filter(a, g, block_size):
    src = np.sqrt(g) * np.random.randn(block_size, 1)  # noise

    b = np.concatenate([np.array([-1]), a])

    x_hat = signal.lfilter([1], b.T, src.T).T

    # convert Nx1 matrix into a N vector
    return np.squeeze(x_hat)


def lpc_decode(A, G, w, H):

    [ne, n] = G.shape
    nw = len(w)
    [p, _] = A.shape

    B_hat = np.zeros((n, nw))

    for i in range(n):
        B_hat[i, :] = run_source_filter(A[:, i], G[:, i], nw)

    # recover signal from blocks
    x_hat = add_overlapping_blocks(B_hat, H)

    return x_hat


def lpc_encode(x, p, window, M, H):

    B = enframe(x, window, M, H)
    [nb, nw] = B.shape

    A = np.zeros((p, nb))
    G = np.zeros((1, nb))

    for i in range(nb):
        [a, g] = solve_lpc(B[i, :], p)

        A[:, i] = a
        G[:, i] = g

    return [A, G]
