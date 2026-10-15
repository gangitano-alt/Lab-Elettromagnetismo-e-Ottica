import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ATTIVITÀ 1: Costruzione del modello fisico
# ==========================================

# Parametri iniziali suggeriti[cite: 2]
B = 20.0          # Fondo uniforme (conteggi per bin)[cite: 2]
I0 = 400.0        # Scala del segnale (conteggi per bin)[cite: 2]
alpha = 0.90      # Larghezza dell'inviluppo (mm^-1)[cite: 2]
delta = 5.20      # Frequenza spaziale delle frange (mm^-1)[cite: 2]

# Creazione dell'array delle posizioni: 161 punti equidistanti in [-5.0, 5.0] mm[cite: 2]
x = np.linspace(-5.0, 5.0, 161)

# Implementazione delle componenti del modello usando np.sinc di NumPy
# Nota: np.sinc(z) = sin(pi*z) / (pi*z), quindi l'inviluppo si calcola come sinc(alpha*x / pi)**2[cite: 2]
E_x = np.sin(alpha * x) / (alpha * x) # Alternativa: np.sinc(alpha * x / np.pi)
E_x = np.where(x == 0, 1.0, E_x)**2  # Gestione nativa del limite in x=0 senza correzioni manuali forzate post-divisione

C_x = np.cos(delta * x)**2

# Modello medio atteso mu(x)[cite: 2]
mu_x = B + I0 * E_x * C_x

# Verifiche richieste[cite: 3]
if(np.all(np.isfinite(mu_x))):
    if(np.all(mu_x >= 0)):
        if(np.allclose(mu_x, mu_x[::-1])):
            if(np.all(mu_x[80]== B + I0)):

                # Rappresentazione grafica del modello[cite: 3]
                plt.figure(figsize=(10, 6))
                plt.plot(x, mu_x, label=r'$\mu(x)$ (Modello atteso)', color='black', lw=2)
                plt.axhline(B, color='blue', linestyle='--', label=r'Fondo $B$')
                plt.plot(x, B + I0 * E_x, color='red', linestyle=':', label=r'Inviluppo superiore $B + I_0 E(x)$')
                plt.xlabel('Posizione $x$ (mm)')
                plt.ylabel('Conteggi per bin')
                plt.title('Attività 1: Modello di Intensità (Interferenza e Diffrazione)')
                plt.legend()
                plt.grid(True)
plt.show()


# ==========================================
# ATTIVITÀ 2: Simulazione di un'acquisizione
# ==========================================

# Inizializzazione di un generatore di numeri casuali locale (Best practice con NumPy moderno)[cite: 3]
rng = np.random.default_rng(seed=42)

# Generazione dei conteggi indipendenti secondo una distribuzione di Poisson[cite: 3]
n_i = rng.poisson(mu_x)

# Definizione della regione centrale |x_i| < 1.2 mm[cite: 3]
central_mask = np.abs(x) < 1.2

# Calcolo dei conteggi nella regione centrale e del valore atteso corrispondente[cite: 3]
Nc = np.sum(n_i[central_mask])
mu_c = np.sum(mu_x[central_mask])

print("\n--- Simulazione Singola Acquisizione ---")
print(f"Conteggi osservati nella regione centrale (Nc): {Nc}")
print(f"Valore atteso nella regione centrale (mu_c): {mu_c:.2f}")
print(f"Differenza (Nc - mu_c): {Nc - mu_c:.2f}")
print(f"Incertezza attesa sqrt(mu_c): {np.sqrt(mu_c):.2f}")
print(f"Fluttuazione in unità di deviazione standard: {(Nc - mu_c) / np.sqrt(mu_c):.2f} sigma")

# Grafico della singola acquisizione con barre d'errore[cite: 3]
plt.figure(figsize=(10, 6))
# Per la sola rappresentazione grafica usiamo un errore minimo pari a 1 nei bin vuoti[cite: 3]
errors = np.sqrt(n_i)
errors[errors == 0] = 1.0

plt.errorbar(x, n_i, yerr=errors, fmt='o', color='purple', ecolor='gray', elinewidth=1, capsize=2, label='Dati simulati ($n_i$ Poisson)')
plt.plot(x, mu_x, color='black', lw=1.5, label=r'Curva attesa $\mu(x)$')
plt.xlabel('Posizione $x$ (mm)')
plt.ylabel('Conteggi')
plt.title('Attività 2: Simulazione di una singola acquisizione')
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# ATTIVITÀ 3: Distribuzione dello stimatore
# ==========================================

M = 2000  # Numero di pseudo-esperimenti[cite: 4]

# Generazione della matrice delle simulazioni: ogni riga è un esperimento completo[cite: 4]
# shape: (M, len(x))
simulations_matrix = rng.poisson(mu_x, size=(M, len(x)))

# Calcolo di Nc per ciascuno pseudo-esperimento[cite: 4]
Nc_array = np.sum(simulations_matrix[:, central_mask], axis=1)

# Statistiche sulla distribuzione di Nc[cite: 4]
mean_Nc = np.mean(Nc_array)
std_Nc = np.std(Nc_array, ddof=1)  # Deviazione standard con dof=1[cite: 4]
err_mean_Nc = std_Nc / np.sqrt(M)  # Errore sulla media delle simulazioni[cite: 4]
rel_diff = (mean_Nc - mu_c) / mu_c  # Scarto relativo tra media simulata e mu_c[cite: 4]

print("\n--- Distribuzione dello Stimatore (M = 2000) ---")
print(f"Media simulata di Nc: {mean_Nc:.2f}")
print(f"Valore atteso teorico (mu_c): {mu_c:.2f}")
print(f"Deviazione standard della distribuzione: {std_Nc:.2f}")
print(f"Confronto con sqrt(mu_c): {np.sqrt(mu_c):.2f}")
print(f"Errore sulla media delle simulazioni: {err_mean_Nc:.4f}")
print(f"Scarto relativo: {rel_diff:.5f}")

# Rappresentazione dell'istogramma di Nc[cite: 4]
plt.figure(figsize=(10, 6))
plt.hist(Nc_array, bins=30, density=True, edgecolor='black', alpha=0.7, color='skyblue', label='Distribuzione di $N_c$')
plt.axvline(mean_Nc, color='red', linestyle='-', linewidth=2, label=f'Media simulata ({mean_Nc:.1f})')
plt.axvline(mu_c, color='green', linestyle='--', linewidth=2, label=f'Valore atteso $\mu_c$ ({mu_c:.1f})')
plt.xlabel(r'Valore dello stimatore $N_c$')
plt.ylabel('Densità di probabilità')
plt.title('Attività 3: Distribuzione dello stimatore $N_c$ su 2000 pseudo-esperimenti')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()