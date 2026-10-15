import numpy as np
import matplotlib.pyplot as plt




B=20.0
I0 = 400.0
ALPHA=0.9
DELTA=5.2

x = np.linspace(-5,5,161)

E_X = np.sin(ALPHA*x)/(ALPHA*x)
E_X =np.where(x==0,1.0, E_X)**2
C_X =np.cos(DELTA*x)**2

GAMMA_X = B + I0 * E_X * C_X

if(np.all(np.isfinite(GAMMA_X))):
    if (np.all(GAMMA_X>=0)):
      if (np.allclose(GAMMA_X, GAMMA_X[::1])):
         if(GAMMA_X[80] == B+I0): 
            plt.figure(figsize=(10,6))
            plt.plot(x,GAMMA_X,label=r'$\mu(x)$ (Modello atteso)', color='black', lw=2)
            plt.grid(True)
            plt.axhline(B, color='blue', linestyle='--', label=r'Fondo $B$')
            plt.plot(x, B + I0 * E_X, color='red', linestyle=':', label=r'Inviluppo superiore $B + I_0 E(x)$')
            plt.xlabel('Posizione $x$ (mm)')
            plt.ylabel('Conteggi per bin')
            plt.title('Attività 1: Modello di Intensità (Interferenza e Diffrazione)')
            plt.legend()
plt.show()



