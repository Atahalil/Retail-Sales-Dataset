import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
import shutil
from pathlib import Path
import os
import matplotlib.pyplot as plt



# CSV dosyasını yükle
df = pd.read_csv('c:/Veri/retail_sales_dataset.csv')


# Ek Fonksiyonlar ve Gelişmiş Analizler

sns.set(style='whitegrid')

def load_and_clean(path):
    df = pd.read_csv(path)
    # Kolon isimlerini düzelt
    df.columns = df.columns.str.strip()
    # Tarih parse
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Numerik dönüşümler (varsa)
    for col in ['Age', 'Quantity', 'Price per Unit', 'Total Amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Boş değerler raporu
    na_counts = df.isna().sum()
    print('Eksik değerler (kolon:eksik):')
    print(na_counts[na_counts>0])
    # Önemli eksikler: Transaction ID veya Total Amount yoksa satırları çıkar
    key_cols = [c for c in ['Transaction ID','Total Amount'] if c in df.columns]
    if key_cols:
        before = len(df)
        df = df.dropna(subset=key_cols)
        print(f"Dropped {before - len(df)} rows due to missing {key_cols}")
    # Basit doldurmalar
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].fillna('Unknown')
    # Duplicate kontrol (Transaction ID varsa)
    if 'Transaction ID' in df.columns:
        dup_count = df.duplicated(subset=['Transaction ID']).sum()
        print(f"Duplicate Transaction ID count: {dup_count}")
        df = df.drop_duplicates(subset=['Transaction ID'])
    else:
        dup_count = df.duplicated().sum()
        print(f"Duplicate full-row count: {dup_count}")
        df = df.drop_duplicates()
    for num in ['Total Amount','Quantity']:
        if num in df.columns:
            q1 = df[num].quantile(0.25)
            q3 = df[num].quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
            outliers = df[(df[num] < low) | (df[num] > high)]
            print(f"{num}: {len(outliers)} outliers (IQR method)")
    return df

def summary_statistics(df):
    print('\\n=== Genel Özet ===')
    print('Satır sayısı:', len(df))
    print('Sütun sayısı:', len(df.columns))
    print('\\nVeri tipleri:')
    print(df.dtypes)
    print('\\nSayısal özet:')
    print(df.select_dtypes(include=[np.number]).describe().round(2))
    # Kategori bazlı örnek: Product Category varsa
    if 'Product Category' in df.columns and 'Total Amount' in df.columns:
        cat = df.groupby('Product Category', observed=True)['Total Amount'].agg(['count','sum','mean']).sort_values('sum', ascending=False)
        print('\\nKategori bazlı satış (ilk 10):')
        print(cat.head(10).round(2))
    # Müşteri sayısı
    if 'Customer ID' in df.columns:
        print('\\nBenzersiz müşteri sayısı:', df['Customer ID'].nunique())

def correlation_analysis(df, save_plot=True):
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        print('Korelasyon için yeterli sayısal kolon yok.')
        return
    corr = num_df.corr()
    print('\\n=== Korelasyon Matrisi ===')
    print(corr.round(2))
    # Kaydet ve görselleştir
    if save_plot:
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0)
        plt.title('Korelasyon Matrisi (Numeric Columns)')
        plt.tight_layout()
        out = plots_dir / 'correlation_matrix.png'
        plt.savefig(out)
        plt.close()
        print('Korelasyon ısı haritası kaydedildi:', out.resolve())

# Örnek kullanım
if __name__ == '__main__':
    csv_path = r'c:/Veri/retail_sales_dataset.csv'   # dosya yolunu gerektiği gibi güncelleyin
    df = load_and_clean(csv_path)
    summary_statistics(df)
    correlation_analysis(df)
    # Temizlenmiş veriyi kaydetmek isterseniz:
    Path('plots').mkdir(exist_ok=True)
    df.to_csv(Path('plots') / 'cleaned_retail_sales.csv', index=False)
    print('Temizlenmiş veri kaydedildi:', (Path('plots') / 'cleaned_retail_sales.csv').resolve())



# Temel bilgiler
print("=" * 60)
print("VERİ İSTATİSTİKLERİ - RETAIL SALES DATASET")
print("=" * 60)

# 1. Veri Seti Özeti
print("\n📊 VERİ SETİ ÖZET BİLGİLERİ:")
print(f"Toplam Satış İşlemi: {len(df)}")
print(f"Toplam Sütun Sayısı: {len(df.columns)}")
print(f"Veri Türleri:\n{df.dtypes}")

# 2. Sayısal Değerlerin İstatistikleri
print("\n📈 SAYISAL VERİLERİN İSTATİSTİKLERİ:")
print(df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].describe().round(2))

# 3. Kategori Analizi
print("\n🏪 ÜRÜN KATEGORİLERİ ANALİZİ:")
category_stats = df.groupby('Product Category').agg({
    'Transaction ID': 'count',
    'Total Amount': ['sum', 'mean', 'min', 'max']
}).round(2)
category_stats.columns = ['İşlem Sayısı', 'Toplam Satış', 'Ortalama Satış', 'Min Satış', 'Max Satış']
print(category_stats)

# 4. Cinsiyete Göre Analiz
print("\n👥 CİNSİYETE GÖRE ANALİZ:")
gender_stats = df.groupby('Gender').agg({
    'Transaction ID': 'count',
    'Total Amount': ['sum', 'mean'],
    'Age': 'mean'
}).round(2)
gender_stats.columns = ['İşlem Sayısı', 'Toplam Satış', 'Ortalama Satış', 'Ort. Yaş']
print(gender_stats)

# 5. Yaş Gruplarına Göre Analiz
print("\n📋 YAŞ GRUPLAARINA GÖRE ANALİZ:")
df['Age Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 65], 
                         labels=['18-25', '26-35', '36-50', '51+'])
age_group_stats = df.groupby('Age Group', observed=True).agg({
    'Transaction ID': 'count',
    'Total Amount': ['sum', 'mean']
}).round(2)
age_group_stats.columns = ['İşlem Sayısı', 'Toplam Satış', 'Ortalama Satış']
print(age_group_stats)

# 6. Genel Finansal Özet
print("\n💰 GENEL FİNANSAL ÖZET:")
print(f"Toplam Satış Tutarı: ${df['Total Amount'].sum():,.2f}")
print(f"Ortalama İşlem Tutarı: ${df['Total Amount'].mean():,.2f}")
print(f"Medyan İşlem Tutarı: ${df['Total Amount'].median():,.2f}")
print(f"En Yüksek İşlem: ${df['Total Amount'].max():,.2f}")
print(f"En Düşük İşlem: ${df['Total Amount'].min():,.2f}")
print(f"Standart Sapma: ${df['Total Amount'].std():,.2f}")

# 7. Müşteri Bilgileri
print("\n👤 MÜŞTERİ BİLGİLERİ:")
print(f"Toplam Benzersiz Müşteri: {df['Customer ID'].nunique()}")
print(f"Ortalama Müşteri Yaşı: {df['Age'].mean():.1f}")
print(f"En Genç Müşteri: {df['Age'].min()} yaş")
print(f"En Yaşlı Müşteri: {df['Age'].max()} yaş")

# 8. Miktara Göre Analiz
print("\n📦 MİKTAR ANALİZİ:")
print(f"Toplam Satılan Ürün Miktarı: {df['Quantity'].sum()} adet")
print(f"Ortalama Ürün Miktarı: {df['Quantity'].mean():.2f} adet")
print(f"En Fazla Satılan Miktar: {df['Quantity'].max()} adet")
print(f"En Az Satılan Miktar: {df['Quantity'].min()} adet")

# 9. Fiyat Analizi
print("\n💵 FİYAT ANALİZİ:")
print(f"Ortalama Birim Fiyatı: ${df['Price per Unit'].mean():,.2f}")
print(f"En Yüksek Birim Fiyatı: ${df['Price per Unit'].max():,.2f}")
print(f"En Düşük Birim Fiyatı: ${df['Price per Unit'].min():,.2f}")

# 10. Tarih Aralığı
print("\n📅 TARİH BİLGİSİ:")
df['Date'] = pd.to_datetime(df['Date'])
print(f"Veri Başlangıç Tarihi: {df['Date'].min().date()}")
print(f"Veri Bitiş Tarihi: {df['Date'].max().date()}")
print(f"Veri Süresi: {(df['Date'].max() - df['Date'].min()).days} gün")

print("\n" + "=" * 60)

# -- Görselleştirmeler ve VS Code görüntüleme yardımcıları --
sns.set(style='whitegrid', palette='muted')

def create_and_open_plots(df):
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # Hazırlık: Date ve Age Group
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()

    if 'Age' in df.columns:
        df['Age Group'] = pd.cut(df['Age'], bins=[0,25,35,50,65,120], labels=['0-25','26-35','36-50','51-65','66+'])

    saved = []

    # 1) Kategori başına satış dağılımı
    if 'Product Category' in df.columns and 'Total Amount' in df.columns:
        cat = df.groupby('Product Category', observed=True)['Total Amount'].sum().sort_values(ascending=False)
        plt.figure(figsize=(8,5))
        sns.barplot(x=cat.values, y=cat.index, palette='tab10')
        plt.title('Toplam Satış - Ürün Kategorisi')
        plt.xlabel('Toplam Satış')
        plt.tight_layout()
        out1 = plots_dir / 'sales_by_category.png'
        plt.savefig(out1)
        plt.close()
        saved.append(out1)

    # 2) Yaş gruplarına göre harcama trendi (aylık)
    if 'Age Group' in df.columns and 'Month' in df.columns and 'Total Amount' in df.columns:
        pivot = df.groupby(['Month','Age Group'], observed=True)['Total Amount'].sum().reset_index()
        plt.figure(figsize=(10,6))
        sns.lineplot(data=pivot, x='Month', y='Total Amount', hue='Age Group', marker='o')
        plt.title('Yaş Gruplarına Göre Aylık Harcama Trendi')
        plt.xlabel('Ay')
        plt.ylabel('Toplam Harcama')
        plt.xticks(rotation=45)
        plt.legend(title='Age Group')
        plt.tight_layout()
        out2 = plots_dir / 'agegroup_monthly_trend.png'
        plt.savefig(out2)
        plt.close()
        saved.append(out2)

    # 3) Aylık satış performansı
    if 'Month' in df.columns and 'Total Amount' in df.columns:
        monthly = df.groupby('Month', observed=True)['Total Amount'].sum().sort_index()
        plt.figure(figsize=(10,5))
        sns.lineplot(x=monthly.index, y=monthly.values, marker='o')
        plt.title('Aylık Toplam Satış')
        plt.xlabel('Ay')
        plt.ylabel('Toplam Satış')
        plt.xticks(rotation=45)
        plt.tight_layout()
        out3 = plots_dir / 'monthly_sales.png'
        plt.savefig(out3)
        plt.close()
        saved.append(out3)

    # Açma: önce `code` CLI varsa VS Code'da aç, yoksa Windows için os.startfile ile açmayı dene
    if not saved:
        print('Görüntülenecek grafik bulunamadı. Lütfen gerekli kolonların mevcut olduğunu doğrulayın.')
        return

    code_cli = shutil.which('code')
    if code_cli:
        try:
            subprocess.run([code_cli] + [str(p.resolve()) for p in saved], check=False)
            print('Grafikler VS Code ile açıldı (code CLI kullanıldı).')
            return
        except Exception as e:
            print('VS Code açılırken hata:', e)

    # Fallback: Windows için os.startfile (VS Code değilse sistem görüntüleyicisi açılır)
    if os.name == 'nt':
        for p in saved:
            try:
                os.startfile(p.resolve())
            except Exception:
                pass
        print('Grafikler sistem varsayılan görüntüleyicisinde açıldı (veya yollar yazdırıldı).')
    else:
        print('`code` komutu bulunamadı ve otomatik açma desteklenmiyor; grafik dosyaları:')
        for p in saved:
            print('-', p.resolve())


# Kullanım bağlamında çağır
if __name__ == '__main__':
    try:
        create_and_open_plots(df)
    except Exception as e:
        print('Görselleştirme çalıştırılamadı:', e)



