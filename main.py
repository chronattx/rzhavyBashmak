import numpy as np
import pandas as pd
import glob
from scipy.optimize import nnls
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_spectrum(df, title="Спектр", label="Спектральная интенсивность"):
    plt.figure(figsize=(10, 5))
    plt.plot(df['wavelength'], df['intensity'], label=label, color='orange')
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Интенсивность (W/m²/nm)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_deficit(wavelengths, deficit, threshold=0.05):
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, deficit, label="Недостаток спектра", color='red')
    plt.axhline(threshold, color='gray', linestyle='--', label="Порог чувствительности")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Абсолютная ошибка (W/m²/nm)")
    plt.title("Разность между эталонным и синтезированным спектром")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_reference_spectrum_csv(file_path):
    df = pd.read_csv(file_path, sep=',')
    df = df.iloc[:, :2]  # Только первые две колонки
    df.columns = ['wavelength', 'intensity']
    return df


def load_lamp_spectra(folder_path):
    spectra = {}
    for file in glob.glob(f"{folder_path}/*.txt"):
        name = file.split("/")[-1].split("\\")[-1].split(".")[0]  # Для Windows путей
        df = pd.read_csv(file, sep=" ", skiprows=1, header=None)  # Пропускаем первую строку
        df.columns = ['wavelength', 'intensity']
        spectra[name] = df['intensity'].values  # Уже интерполировано → только интенсивность
    return spectra


def relative_error(synth, target):
    return np.mean(np.abs(synth - target) / (target + 1e-10))


def find_best_combination(ref_spec, lamp_specs, error_threshold=0.15, max_lamps=14):
    ref_int = ref_spec['intensity'].values
    lamp_names = list(lamp_specs.keys())
    lamp_matrix = {name: lamp_specs[name] for name in lamp_names}

    best_result = None

    for r in range(1, min(len(lamp_names), max_lamps) + 1):
        for combo in tqdm(combinations(lamp_names, r), desc=f"Комбинации по {r}"):
            M = np.vstack([lamp_matrix[name] for name in combo]).T
            coeffs, _ = nnls(M, ref_int)
            synthesized = M @ coeffs
            error = relative_error(synthesized, ref_int)

            if error <= error_threshold:
                if best_result is None or len(combo) < len(best_result['lamps']) or \
                   (len(combo) == len(best_result['lamps']) and error < best_result['error']):
                    best_result = {
                        'lamps': combo,
                        'coeffs': coeffs,
                        'error': error,
                        'synthesized': synthesized
                    }

    return best_result


def main():
    ref_spec = load_reference_spectrum_csv("waves.csv")
    lamp_spectra = load_lamp_spectra("lamps")  # Папка с .txt-файлами ламп

    result = find_best_combination(ref_spec, lamp_spectra, error_threshold=0.15)

    if result:
        print("✅ Лучшая комбинация ламп:")
        for name, coeff in zip(result['lamps'], result['coeffs']):
            print(f" - {name}: коэффициент {coeff:.3f}")
        print(f"Средняя ошибка: {result['error']:.2%}")

        # График эталонного и синтезированного спектра
        plt.figure(figsize=(12, 5))
        plt.plot(ref_spec['wavelength'], ref_spec['intensity'], label="Эталонный спектр", color='orange')
        plt.plot(ref_spec['wavelength'], result['synthesized'], label="Синтезированный спектр", color='blue')
        plt.xlabel("Длина волны (нм)")
        plt.ylabel("Интенсивность (W/m²/nm)")
        plt.title("Сравнение эталонного и синтезированного спектров")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Анализ дефицита
        deficit = np.abs(ref_spec['intensity'].values - result['synthesized'])
    else:
        print("❌ Не удалось подобрать подходящие лампы.")
        print("💡 Попробуйте загрузить больше спектров или изменить порог ошибки.")

        # Построим график дефицита между эталоном и "нулевым" спектром
        zero_spectrum = np.zeros_like(ref_spec['intensity'].values)
        deficit = np.abs(ref_spec['intensity'].values - zero_spectrum)

    # Показываем дефицит в любом случае
    plot_deficit(ref_spec['wavelength'], deficit)

    # Сохраняем CSV с дефицитом
    deficit_df = pd.DataFrame({
        'wavelength': ref_spec['wavelength'],
        'deficit': deficit
    })
    deficit_df.to_csv("deficit_report.csv", index=False)
    print("📁 Файл с дефицитом по длинам волн сохранён: deficit_report.csv")

    # Печатаем ТОП-10 дефицитных участков
    print("\n🔍 ТОП-10 длин волн с наибольшим дефицитом:")
    top_def = deficit_df.sort_values(by="deficit", ascending=False).head(10)
    for _, row in top_def.iterrows():
        print(f" - {int(row['wavelength'])} нм: недостача {row['deficit']:.4f} W/m²/nm")


if __name__ == "__main__":
    main()
