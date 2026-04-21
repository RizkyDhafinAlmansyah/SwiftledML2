# ============================================================
# Generate Document Assets - C400 & Skripsi (Bab 3 & Bab 4)
# ============================================================
# Output: gambar-gambar siap pakai untuk dokumen akademik
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# --- Config ---
DATA_PATH = r"c:\SwiftledML\data\ml_training_with_external.csv"
OUTPUT_DIR = r"c:\SwiftledML\evaluation_output\document_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Font setup for professional look
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# --- Load Data ---
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]
df['recorded_at'] = pd.to_datetime(df['recorded_at'], errors='coerce')

# Split real vs external
real = df[df['recorded_at'].dt.year >= 2025].copy()
ext = df[df['recorded_at'].dt.year < 2025].copy()

print(f"Total: {len(df):,} | Real: {len(real):,} | External: {len(ext):,}")
print(f"Output dir: {OUTPUT_DIR}")

# ============================================================
# 1. TABEL STRUKTUR DATASET (Bab 3)
# ============================================================
print("\n[1] Generating: Struktur Dataset...")

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')

table_data = [
    ['No', 'Nama Kolom', 'Tipe Data', 'Satuan', 'Deskripsi'],
    ['1', 'recorded_at', 'DateTime', '-', 'Waktu pencatatan data sensor'],
    ['2', 'temperature_c', 'Float', '°C', 'Suhu udara dalam ruangan'],
    ['3', 'humidity_rh', 'Float', '%RH', 'Kelembaban relatif udara'],
    ['4', 'nh3_ppm', 'Float', 'ppm', 'Konsentrasi gas amonia (NH₃)'],
]

colors_header = ['#2C3E50'] * 5
colors_rows = [['#F8F9FA'] * 5, ['#FFFFFF'] * 5, ['#F8F9FA'] * 5, ['#FFFFFF'] * 5]

table = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc='center',
    loc='center',
    colWidths=[0.06, 0.18, 0.12, 0.08, 0.42],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)

# Style header
for j in range(5):
    cell = table[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=11)

# Style rows
for i in range(1, 5):
    for j in range(5):
        cell = table[i, j]
        cell.set_facecolor('#F8F9FA' if i % 2 == 1 else '#FFFFFF')
        cell.set_edgecolor('#DEE2E6')

ax.set_title('Struktur Dataset Sensor Sarang Walet', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_struktur_dataset.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  [OK] 01_struktur_dataset.png")


# ============================================================
# 2. SAMPLE DATA (Bab 3) — 10 baris contoh
# ============================================================
print("\n[2] Generating: Sample Data...")

sample = real.sort_values('recorded_at').head(10).copy()
sample['recorded_at'] = sample['recorded_at'].dt.strftime('%Y-%m-%d %H:%M')
sample = sample[['recorded_at', 'temperature_c', 'humidity_rh', 'nh3_ppm']]

fig, ax = plt.subplots(figsize=(11, 5))
ax.axis('off')

headers = ['No', 'Waktu Pencatatan', 'Suhu (°C)', 'Kelembaban (%)', 'NH₃ (ppm)']
rows = []
for idx, (_, row) in enumerate(sample.iterrows(), 1):
    rows.append([
        str(idx),
        row['recorded_at'],
        f"{row['temperature_c']:.2f}",
        f"{row['humidity_rh']:.2f}",
        f"{row['nh3_ppm']:.3f}",
    ])

table = ax.table(
    cellText=rows,
    colLabels=headers,
    cellLoc='center',
    loc='center',
    colWidths=[0.06, 0.28, 0.16, 0.18, 0.16],
)
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1.0, 1.7)

for j in range(5):
    cell = table[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=11)

for i in range(1, len(rows) + 1):
    for j in range(5):
        cell = table[i, j]
        cell.set_facecolor('#F8F9FA' if i % 2 == 1 else '#FFFFFF')
        cell.set_edgecolor('#DEE2E6')

ax.set_title('Contoh Data Sensor (10 Sampel)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_sample_data.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  [OK] 02_sample_data.png")


# ============================================================
# 3. STATISTIK DESKRIPTIF (Bab 3 / Bab 4)
# ============================================================
print("\n[3] Generating: Statistik Deskriptif...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Compute stats
params = ['temperature_c', 'humidity_rh', 'nh3_ppm']
labels = ['Suhu (°C)', 'Kelembaban (%RH)', 'NH₃ (ppm)']

headers = ['Parameter', 'Sumber', 'Jumlah', 'Min', 'Max', 'Mean', 'Median', 'Std Dev']
rows = []
for param, label in zip(params, labels):
    for source, sdf, color_tag in [('Data Real', real, 'real'), ('Data External', ext, 'ext'), ('Gabungan', df, 'all')]:
        col = sdf[param]
        rows.append([
            label if source == 'Data Real' else '',
            source,
            f'{len(col):,}',
            f'{col.min():.2f}',
            f'{col.max():.2f}',
            f'{col.mean():.2f}',
            f'{col.median():.2f}',
            f'{col.std():.2f}',
        ])

table = ax.table(
    cellText=rows,
    colLabels=headers,
    cellLoc='center',
    loc='center',
    colWidths=[0.15, 0.12, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.6)

for j in range(8):
    cell = table[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=10.5)

# Alternate shading per parameter group
group_colors = [
    ('#E8F4FD', '#D1ECF9', '#BFE3F5'),  # Suhu - biru muda
    ('#E8F8E8', '#D1F0D1', '#BFE8BF'),  # RH - hijau muda
    ('#FFF3E0', '#FFE8CC', '#FFDDB8'),  # NH3 - oranye muda
]
for i in range(len(rows)):
    group = i // 3
    sub = i % 3
    for j in range(8):
        cell = table[i + 1, j]
        cell.set_facecolor(group_colors[group][sub])
        cell.set_edgecolor('#DEE2E6')
        if sub == 2:  # "Gabungan" row - bold
            cell.set_text_props(fontweight='bold')

ax.set_title('Statistik Deskriptif Dataset Training', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_statistik_deskriptif.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  [OK] 03_statistik_deskriptif.png")


# ============================================================
# 4. DISTRIBUSI DATA — HISTOGRAM (Bab 4)
# ============================================================
print("\n[4] Generating: Distribusi Data (Histogram)...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (param, label, unit, color_r, color_e) in enumerate([
    ('temperature_c', 'Suhu', '°C', '#3498DB', '#E74C3C'),
    ('humidity_rh', 'Kelembaban', '%RH', '#2ECC71', '#E67E22'),
    ('nh3_ppm', 'NH₃', 'ppm', '#9B59B6', '#F39C12'),
]):
    ax = axes[idx]
    ax.hist(real[param], bins=40, alpha=0.7, color=color_r, label=f'Real ({len(real):,})', edgecolor='white', linewidth=0.5)
    ax.hist(ext[param], bins=40, alpha=0.6, color=color_e, label=f'External ({len(ext):,})', edgecolor='white', linewidth=0.5)
    ax.set_xlabel(f'{label} ({unit})', fontsize=12)
    ax.set_ylabel('Frekuensi', fontsize=12)
    ax.set_title(f'Distribusi {label}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

plt.suptitle('Distribusi Data Sensor: Real vs External', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_distribusi_histogram.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  [OK] 04_distribusi_histogram.png")


# ============================================================
# 5. BOX PLOT PERBANDINGAN (Bab 4)
# ============================================================
print("\n[5] Generating: Box Plot Perbandingan...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for idx, (param, label, unit) in enumerate([
    ('temperature_c', 'Suhu', '°C'),
    ('humidity_rh', 'Kelembaban', '%RH'),
    ('nh3_ppm', 'NH₃', 'ppm'),
]):
    ax = axes[idx]
    data_to_plot = [real[param].dropna(), ext[param].dropna()]
    bp = ax.boxplot(data_to_plot, labels=['Data\nReal', 'Data\nExternal'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='#E74C3C', linewidth=2))
    
    colors_box = ['#3498DB', '#E67E22']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(f'{label} ({unit})', fontsize=12)
    ax.set_title(f'{label}', fontsize=13, fontweight='bold')

    # Add mean annotation
    for i, d in enumerate(data_to_plot, 1):
        ax.text(i, d.mean(), f'μ={d.mean():.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

plt.suptitle('Perbandingan Distribusi: Data Real vs External (GAMS)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_boxplot_perbandingan.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  [OK] 05_boxplot_perbandingan.png")


# ============================================================
# 6. KOMPOSISI DATASET (Bab 3) — Improved
# ============================================================
print("\n[6] Generating: Komposisi Dataset...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: Dataset composition
sizes = [len(real), len(ext)]
labels_pie = [f'Data Real\n({len(real):,} baris)', f'Data External GAMS\n({len(ext):,} baris)']
colors_pie = ['#3498DB', '#E67E22']
explode = (0.03, 0.03)

wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
    startangle=90, explode=explode, textprops={'fontsize': 11},
    pctdistance=0.75, labeldistance=1.15
)
for at in autotexts:
    at.set_fontweight('bold')
    at.set_fontsize(12)
axes[0].set_title('Komposisi Dataset Training', fontsize=13, fontweight='bold')

# Right: Timeline / source info as a clean info card
axes[1].axis('off')
info_text = (
    f"Ringkasan Dataset Training\n"
    f"{'─' * 40}\n\n"
    f"Total data          :  {len(df):,} baris\n"
    f"Data Real (sensor)  :  {len(real):,} baris (70.8%)\n"
    f"Data External       :  {len(ext):,} baris (29.2%)\n\n"
    f"{'─' * 40}\n\n"
    f"Sumber Real         :  Sensor IoT sarang walet\n"
    f"Sumber External     :  GAMS Indoor Air Quality\n"
    f"Jumlah fitur        :  3 sensor + 8-9 engineered\n"
    f"Periode real        :  Nov - Des 2025\n"
    f"Periode external    :  2017 - 2020\n\n"
    f"{'─' * 40}\n\n"
    f"Tujuan external     :  Memperluas cakupan\n"
    f"                       distribusi data untuk\n"
    f"                       kondisi ekstrem (NH₃ tinggi,\n"
    f"                       suhu rendah)"
)
axes[1].text(0.05, 0.95, info_text, transform=axes[1].transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_komposisi_dataset.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  [OK] 06_komposisi_dataset.png")


# ============================================================
# 7. TABEL HASIL MODEL (Bab 4)
# ============================================================
print("\n[7] Generating: Tabel Hasil Model...")

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis('off')

headers = ['Model', 'Algoritma', 'Metrik Utama', 'Nilai', 'CV Score', 'Keterangan']
rows = [
    ['Grade\nPrediction', 'LightGBM\nClassifier', 'F1-Macro', '0.8067', '0.5244 ± 0.17', '3 kelas:\nbagus/sedang/buruk'],
    ['Pump State\nPrediction', 'LightGBM\nClassifier', 'F1-Score', '0.7710', '0.9152 ± 0.07', 'Binary:\nON/OFF'],
    ['Pump Duration\nPrediction', 'LightGBM\nRegressor', 'MAE', '0.78 detik', 'R² = 0.782', 'Range:\n5 - 30 detik'],
    ['Anomaly\nDetection', 'Isolation\nForest', 'Anomaly Rate', '2.0%', '-', '684 anomali\ndari 34,190'],
]

table = ax.table(
    cellText=rows,
    colLabels=headers,
    cellLoc='center',
    loc='center',
    colWidths=[0.14, 0.12, 0.11, 0.10, 0.14, 0.16],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 2.2)

for j in range(6):
    cell = table[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=10.5)

row_colors = ['#E8F4FD', '#E8F8E8', '#FFF3E0', '#F3E8FF']
for i in range(len(rows)):
    for j in range(6):
        cell = table[i + 1, j]
        cell.set_facecolor(row_colors[i])
        cell.set_edgecolor('#DEE2E6')

ax.set_title('Ringkasan Performa Model Machine Learning', fontsize=14, fontweight='bold', pad=25)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_tabel_hasil_model.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  [OK] 07_tabel_hasil_model.png")


# ============================================================
# 8. PERBANDINGAN SEBELUM vs SESUDAH (Bab 4)
# ============================================================
print("\n[8] Generating: Perbandingan Before/After (improved)...")

fig, ax = plt.subplots(figsize=(11, 6))

metrics = ['Grade\nF1-Macro', 'Pump State\nF1-Score', 'Duration\nR²']
before = [0.3283, 0.4702, 0.924]
after = [0.8067, 0.9152, 0.782]
improvement = ['+145.7%', '+94.6%', '-15.4%']

x = np.arange(len(metrics))
width = 0.32

bars1 = ax.bar(x - width/2, before, width, label='Sebelum Optimasi\n(Real Data Only, RandomForest)',
               color='#E74C3C', alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
bars2 = ax.bar(x + width/2, after, width, label='Sesudah Optimasi\n(+ External + LightGBM + Optuna)',
               color='#2ECC71', alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)

ax.set_ylabel('Score', fontsize=13)
ax.set_title('Perbandingan Performa Model: Sebelum vs Sesudah Optimasi',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax.set_ylim(0, 1.25)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# Value labels + improvement
for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvement)):
    ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.02,
            f'{bar1.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#C0392B')
    ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.02,
            f'{bar2.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#27AE60')
    # Improvement arrow
    mid_x = (bar1.get_x() + bar2.get_x() + bar2.get_width()) / 2
    top_y = max(bar1.get_height(), bar2.get_height()) + 0.10
    color_imp = '#27AE60' if imp.startswith('+') else '#E74C3C'
    ax.text(mid_x, top_y, imp, ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=color_imp, bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFCC', edgecolor=color_imp, alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_perbandingan_before_after.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  [OK] 08_perbandingan_before_after.png")


# ============================================================
# 9. TABEL FITUR (Bab 3)
# ============================================================
print("\n[9] Generating: Tabel Fitur Model...")

fig, ax = plt.subplots(figsize=(13, 8))
ax.axis('off')

headers = ['No', 'Nama Fitur', 'Tipe', 'Deskripsi', 'Dipakai Oleh']
rows = [
    ['1',  'temperature',            'Sensor',      'Suhu udara (°C)',                    'Grade, Pump, Duration'],
    ['2',  'humidity',               'Sensor',      'Kelembaban relatif (%RH)',            'Grade, Pump, Duration'],
    ['3',  'ammonia',                'Sensor',      'Konsentrasi NH₃ (ppm)',               'Grade, Pump, Duration'],
    ['4',  'temp_avg_1h',            'Rolling',     'Rata-rata suhu 1 jam terakhir',       'Grade, Pump, Duration'],
    ['5',  'humid_avg_1h',           'Rolling',     'Rata-rata kelembaban 1 jam',          'Grade, Pump, Duration'],
    ['6',  'nh3_avg_1h',             'Rolling',     'Rata-rata NH₃ 1 jam',                'Grade'],
    ['7',  'humid_delta_1h',         'Delta',       'Perubahan kelembaban (%)',             'Pump, Duration'],
    ['8',  'comfort_index',          'Derived',     'Indeks kenyamanan (0-100)',            'Grade, Duration'],
    ['9',  'is_daytime',             'Temporal',    'Siang (1) atau malam (0)',             'Grade'],
    ['10', 'hour_of_day',            'Temporal',    'Jam dalam sehari (0-23)',              'Pump, Duration'],
    ['11', 'day_of_week',            'Temporal',    'Hari dalam minggu (0-6)',              'Grade, Pump'],
    ['12', 'temp_humid_interaction', 'Interaction', 'Suhu × Kelembaban / 100',             'Grade, Pump, Duration'],
    ['13', 'nh3_rate_of_change',     'Delta',       'Kecepatan perubahan NH₃',             'Pump, Duration'],
    ['14', 'temp_rolling_std',       'Rolling',     'Volatilitas suhu (std dev)',           'Pump, Duration'],
    ['15', 'time_norm',              'Temporal',    'Waktu ternormalisasi (0.0 - 1.0)',    'Grade, Pump, Duration'],
]

table = ax.table(
    cellText=rows,
    colLabels=headers,
    cellLoc='center',
    loc='center',
    colWidths=[0.04, 0.22, 0.10, 0.32, 0.20],
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.0, 1.55)

for j in range(5):
    cell = table[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=10)

type_colors = {
    'Sensor': '#E8F4FD',
    'Rolling': '#E8F8E8',
    'Delta': '#FFF3E0',
    'Derived': '#F3E8FF',
    'Temporal': '#FDE8E8',
    'Interaction': '#E8F0FF',
}
for i in range(len(rows)):
    ftype = rows[i][2]
    bg = type_colors.get(ftype, '#FFFFFF')
    for j in range(5):
        cell = table[i + 1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor('#DEE2E6')

ax.set_title('Daftar Fitur yang Digunakan dalam Model ML', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_tabel_fitur.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  [OK] 09_tabel_fitur.png")


# ============================================================
# 10. TIMELINE DATA (Bab 3)  
# ============================================================
print("\n[10] Generating: Timeline Data Collection...")

fig, ax = plt.subplots(figsize=(12, 4))

# Real data time distribution
real_sorted = real.sort_values('recorded_at')
ax.scatter(real_sorted['recorded_at'], [1]*len(real_sorted), alpha=0.05, s=5, color='#3498DB', label='Data Real')
ax.scatter(ext.sort_values('recorded_at')['recorded_at'], [0.5]*len(ext), alpha=0.05, s=5, color='#E67E22', label='Data External (GAMS)')

ax.set_yticks([0.5, 1.0])
ax.set_yticklabels(['External\n(GAMS)', 'Real\n(Sensor)'], fontsize=11)
ax.set_ylim(0, 1.5)
ax.set_xlabel('Waktu', fontsize=12)
ax.set_title('Timeline Pengumpulan Data', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')

# Add annotations
ax.annotate(f'{len(real):,} baris', xy=(real_sorted['recorded_at'].median(), 1.0),
            xytext=(real_sorted['recorded_at'].median(), 1.25),
            fontsize=10, ha='center', fontweight='bold', color='#2980B9',
            arrowprops=dict(arrowstyle='->', color='#2980B9'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_timeline_data.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  [OK] 10_timeline_data.png")


# ============================================================
# DONE
# ============================================================
print(f"""
{'='*60}
DOCUMENT ASSETS GENERATED
{'='*60}

Output: {OUTPUT_DIR}

Untuk Bab 3 (Metodologi):
  01. struktur_dataset.png       - Struktur tabel database
  02. sample_data.png            - 10 baris contoh data
  03. statistik_deskriptif.png   - Min/Max/Mean/Std per parameter
  06. komposisi_dataset.png      - Pie chart + info ringkasan
  09. tabel_fitur.png            - Daftar 15 fitur ML
  10. timeline_data.png          - Timeline pengumpulan data

Untuk Bab 4 (Hasil):
  04. distribusi_histogram.png   - Histogram Real vs External
  05. boxplot_perbandingan.png   - Box plot perbandingan
  07. tabel_hasil_model.png      - Ringkasan performa 4 model
  08. perbandingan_before_after.png - Before vs After optimization

Ditambah file yang SUDAH ADA di evaluation_output/:
  - confusion_matrix_grade.png
  - confusion_matrix_pump.png
  - feature_importance_grade.png
  - feature_importance_pump.png
  - confidence_distribution_grade.png
  - cross_validation_scores.png

{'='*60}
""")
