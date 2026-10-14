import matplotlib.pyplot as plt
import numpy as np

# --- PARAMETRI INIZIALI ---
B = 20.0        # Fondo uniforme (conteggi/bin)
I0 = 400.0      # Scala del segnale (conteggi/bin)
alpha = 0.90    # Larghezza dell'inviluppo (mm^-1)
delta = 5.20    # Frequenza spaziale delle frange (mm^-1)

# --- DOMINIO SPAZIALE ---
# 161 punti equidistanti nell'intervallo [-5.0, 5.0] mm
x = np.linspace(-5.0, 5.0, 161)

# --- MODELLO FISICO ---
# Sfruttiamo la definizione di np.sinc di NumPy: np.sinc(z) = sin(pi*z)/(pi*z)[cite: 1]
# Pertanto l'inviluppo è E(x) = [sin(alpha*x) / (alpha*x)]^2 = np.sinc(alpha * x / np.pi)**2[cite: 1]
E_x = np.sinc(alpha * x / np.pi) ** 2
C_x = np.cos(delta * x) ** 2

mu_x = B + I0 * E_x * C_x

# --- VERIFICHE RICHIESTE ---
if(np.all(np.isfinite(mu_x))):
    if(np.all(mu_x >= 0)):
        if(np.allclose(mu_x, mu_x[::-1])):

            # Verifica x = 0 (compatibilità con B + I0)
            mu_0_teorica = B + I0
            mu_0_calcolata = mu_x[np.argmin(np.abs(x))] # punto più vicino a x=0
            print(f"Mu(0) atteso: {mu_0_teorica}, Mu(0) calcolato: {mu_0_calcolata}")

            # --- RAPPRESENTAZIONE GRAFICA ---
            plt.figure(figsize=(10, 6))
            plt.plot(x, mu_x, label=r'$\mu(x)$ (Modello completo)', color='black', lw=1.5)
            plt.axhline(B, color='gray', linestyle='--', label=f'Fondo $B$ = {B}')
            plt.plot(x, B + I0 * E_x, color='red', linestyle=':', label=r'Inviluppo superiore $B + I_0 E(x)$')
            plt.xlabel('Posizione $x$ (mm)')
            plt.ylabel('Conteggi per bin')
            plt.title('Attività 1: Costruzione del Modello Fisico')
            plt.legend()
            plt.grid(True, alpha=0.3)
plt.show()