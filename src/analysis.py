import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

BASE = "/data01/giacomelli/data"

train_ba = pd.read_csv(f"{BASE}/RHPE_Annotations/RHPE_Boneage_train.csv")
val_ba   = pd.read_csv(f"{BASE}/RHPE_Annotations/RHPE_Boneage_val.csv")
test_ba  = pd.read_csv(f"{BASE}/RHPE_Annotations/RHPE_Gender_Chronological_test.csv")

# ── 2. Statistiche base ──────────────────────────────────────────────────────
for name, df in [("TRAIN", train_ba), ("VAL", val_ba), ("TEST", test_ba)]:
    print(f"\n=== {name} ({len(df)} immagini) ===")
    print(f"Maschi: {df['Male'].sum()} ({df['Male'].mean()*100:.1f}%)")
    print(f"Femmine: {(~df['Male']).sum()} ({(~df['Male']).mean()*100:.1f}%)")
    if 'Boneage' in df.columns:
        print(f"Bone age — media: {df['Boneage'].mean():.1f} mesi, "
              f"std: {df['Boneage'].std():.1f}, "
              f"min: {df['Boneage'].min()}, max: {df['Boneage'].max()}")
    print(f"Età cronologica — media: {df['Chronological'].mean():.1f} mesi, "
          f"std: {df['Chronological'].std():.1f}, "
          f"min: {df['Chronological'].min()}, max: {df['Chronological'].max()}")

# ── 3. Dimensioni immagini (campione da train) ───────────────────────────────
train_dir = os.path.join(BASE, "RHPE_train")
sizes = []
for f in os.listdir(train_dir)[:1000]:
    try:
        img = Image.open(os.path.join(train_dir, f))
        sizes.append(img.size)
    except:
        pass

widths  = [s[0] for s in sizes]
heights = [s[1] for s in sizes]
print(f"\n=== Dimensioni immagini (campione 1000) ===")
print(f"Width  — min: {min(widths)}, max: {max(widths)}, media: {sum(widths)/len(widths):.0f}")
print(f"Height — min: {min(heights)}, max: {max(heights)}, media: {sum(heights)/len(heights):.0f}")

# ── 4. Grafici ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for i, (name, df) in enumerate([("Train", train_ba), ("Val", val_ba), ("Test", test_ba)]):
    # Bone age
    if 'Boneage' in df.columns:
        sns.histplot(df['Boneage'], ax=axes[0][i], kde=True, color='steelblue')
        axes[0][i].set_title(f"{name} — Bone Age")
        axes[0][i].set_xlabel("mesi")
    # Età cronologica
    sns.histplot(df['Chronological'], ax=axes[1][i], kde=True, color='coral')
    axes[1][i].set_title(f"{name} — Età Cronologica")
    axes[1][i].set_xlabel("mesi")

plt.tight_layout()
plt.savefig("/data01/giacomelli/src/distribuzione_eta.png", dpi=150)
print("\nGrafico salvato in src/distribuzione_eta.png")