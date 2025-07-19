import numpy as np
import pandas as pd
from scipy.optimize import nnls
import os

np.random.seed(42)  # Для повторяемости

# Диапазон длин волн
wavelengths = np.arange(380, 1101, 1)  # 380–1100 нм, шаг 1 нм

# === 1. Генерация эталонного спектра ===
# Составим "правдоподобный" спектр как сумму 3 Гауссиан
def generate_reference_spectrum():
    s = (
        1.2 * np.exp(-((wavelengths - 450) / 40) ** 2) +
        1.0 * np.exp(-((wavelengths - 550) / 50) ** 2) +
        0.8 * np.exp(-((wavelengths - 800) / 100) ** 2)
    )
    return s

ref_spectrum = generate_reference_spectrum()

# === 2. Генерация ламп (профили-Гауссы) ===
def generate_lamp_profiles(n_lamps):
    centers = np.linspace(400, 1000, n_lamps)
    lamps = []
    for c in centers:
        sigma = np.random.uniform(30, 80)
        amplitude = np.random.uniform(0.5, 1.2)
        profile = amplitude * np.exp(-((wavelengths - c) / sigma) ** 2)
        lamps.append(profile)
    return np.array(lamps)

n_lamps = 5
lamp_profiles = generate_lamp_profiles(n_lamps)

# === 3. Подбор коэффициентов смешивания ===
coeffs, _ = nnls(lamp_profiles.T, ref_spectrum)

# === 4. Проверка восстановления ===
reconstructed = np.dot(coeffs, lamp_profiles)
mape = np.mean(np.abs((reconstructed - ref_spectrum) / ref_spectrum)) * 100
print(f"MAPE: {mape:.2f}% — отклонение от эталона")

# === 5. Сохранение ламп в .txt ===
os.makedirs("lamps/lamps", exist_ok=True)

for i, lamp in enumerate(lamp_profiles):
    filename = f"lamps/lamp{i+1}.txt"
    with open(filename, "w") as f:
        f.write(f"https://example.com/lamp{i+1}\n")
        for wl, intensity in zip(wavelengths, lamp):
            f.write(f"{wl}\t{intensity:.6f}\n")

# === 6. Сохранение эталонного спектра в .csv ===
df = pd.DataFrame({
    "wavelength": wavelengths,
    "intensity": ref_spectrum
})
df.to_csv("waves.csv", index=False)