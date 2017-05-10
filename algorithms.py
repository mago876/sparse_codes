import numpy as np
from numpy import linalg as la

def soft_thres(x, theta):
    return np.sign(x) * np.fmax(0,np.abs(x)-theta)

def ISTA(Wd, X, alpha, L=None, maxiter=15000, eps=1e-6):
    Z0 = np.zeros(Wd.shape[1])
    if L==None:  # Si no se provee, se calcula mayor valor propio de A'*A
        L = np.max(la.eigh(Wd.T.dot(Wd))[0])
    
    k = 0  # Contador de iteraciones
    seguir = True  # Flag para criterio de parada
    Z = [Z0]  # Historial de iteraciones
    
    while seguir:
        k += 1
        Zk = soft_thres(Z0 - (1/L)*Wd.T.dot(Wd.dot(Z0)-X), alpha/L)
        seguir = (k < maxiter) & (la.norm(Zk-Z0) > eps)
        Z.append(Zk)
        Z0 = Zk
    
    return Z


def FISTA(Wd, X, alpha, L=None, maxiter=5000, eps=1e-6):
    Z0 = np.zeros(Wd.shape[1])
    if L==None:  # Si no se provee, se calcula mayor valor propio de A'*A
        L = np.max(la.eigh(Wd.T.dot(Wd))[0])
    
    k = 0  # Contador de iteraciones
    seguir = True  # Flag para criterio de parada
    Z = [Z0]  # Historial de iteraciones
    Y = Z0  # Variable auxiliar introducida en FISTA
    t = 1   # Variable auxiliar introducida en FISTA
    
    while seguir:
        k += 1
        Zk = soft_thres(Y - (1/L)*Wd.T.dot(Wd.dot(Y)-X), alpha/L)
        seguir = (k < maxiter) & (la.norm(Zk-Z0) > eps)
        tk = (1 + np.sqrt(1+4*t**2)) / 2
        Y = Zk + (t-1)/tk * (Zk - Z0)
        Z.append(Zk)
        Z0 = Zk
        t = tk
    
    return Z


def LISTA_fprop(X, We, S, theta, T):
    ''' Algoritmo 3 del paper de Gregor/LeCun '''
    B = We.dot(X)
    Z = [soft_thres(B, theta)]
    C = []
    
    for t in xrange(T):
        C.append(B + S.dot(Z[t]))
        Z.append(soft_thres(C[t], theta))
        
    cache = (X, Z, We, S, theta, T, C, B)  # Para backpropagation
    
    return Z, cache


def gradient_soft_thres(X, t):
    ''' Gradiente del soft-thresholding '''
    return (X>t) + (X<-t)


def LISTA_bprop(Zopt, cache):
    ''' Algoritmo 4 del paper de Gregor/LeCun '''
    (X, Z, We, S, theta, T, C, B) = cache  # Calculados en LISTA_fprop
    dB = np.zeros_like(B)
    dS = np.zeros_like(S)    
    dtheta = np.zeros_like(B)
    dZ = Z[T] - Zopt
    
    for t in reversed(xrange(T)):
        dC = gradient_soft_thres(C[t], theta) * dZ
        dtheta -= np.sign(C[t]) * dC
        dB += dC
        dS += np.tensordot(dC, Z[t], axes=0)
        dZ = S.T.dot(dC)
    
    dB += gradient_soft_thres(B, theta) * dZ
    dtheta -= np.sign(B) * gradient_soft_thres(B, theta) * dZ
    dWe = np.tensordot(dB, X, axes=0)
    # dX = We.T.dot(dB)  # No es necesario optimizar en X
    
    return dWe, dS, dtheta