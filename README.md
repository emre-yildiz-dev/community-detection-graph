# Community Detection: Classical Algorithms vs Graph Neural Networks

Bu proje, **Çizge Teorisi**'nin teorik temellerini modern **Makine Öğrenmesi** uygulamaları ile birleştiren kapsamlı bir topluluk tespiti (community detection) çalışmasıdır. Neo4j ve PyTorch kullanarak klasik algoritmaları Graph Neural Networks ile karşılaştırır.

## 🎯 Proje Hedefleri

- **Klasik Algoritmalar**: Louvain ve Label Propagation algoritmalarını Neo4j GDS ile uygulama
- **Modern Yaklaşımlar**: GCN, GraphSAGE ve GAT modellerini PyTorch Geometric ile eğitme
- **Karşılaştırmalı Analiz**: Tüm yöntemlerin performansını değerlendirme
- **Görselleştirme**: İnteraktif grafikler ve kapsamlı dashboard'lar

## 🏗️ Proje Yapısı

```
community-detection/
├── src/
│   ├── __init__.py
│   ├── config.py              # Konfigürasyon ayarları
│   ├── neo4j_manager.py       # Neo4j veritabanı yönetimi
│   ├── data_loader.py         # Cora verisetini yükleme
│   ├── gnn_models.py          # Graph Neural Network modelleri
│   └── visualization.py      # Görselleştirme araçları
├── notebooks/
│   └── community_detection_analysis.ipynb  # İnteraktif analiz
├── data/                      # Veri dosyaları (otomatik oluşturulur)
├── results/                   # Sonuçlar ve grafikler
├── docker-compose.yml         # Neo4j Docker konfigürasyonu
├── requirements.txt           # Python bağımlılıkları
├── .env                       # Çevre değişkenleri
├── main.py                    # Ana uygulama
└── README.md
```

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler

- Python 3.8+
- Docker ve Docker Compose
- CUDA (isteğe bağlı, GPU desteği için)

### 2. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### 3. Neo4j'yi Docker ile Başlatma

```bash
docker-compose up -d
```

Neo4j arayüzüne http://localhost:7474 adresinden erişebilirsiniz.
- Kullanıcı adı: `neo4j`
- Şifre: `password123`

### 4. Projeyi Çalıştırma

#### Ana Uygulama
```bash
python main.py
```

#### Jupyter Notebook ile İnteraktif Analiz
```bash
jupyter notebook notebooks/community_detection_analysis.ipynb
```

## 📊 Kullanılan Veriset

**Cora Citation Network**: Makine öğrenmesi alanında 2708 bilimsel makale ve 5429 alıntı ilişkisi içeren klasik bir çizge verisettidir.

- **Düğümler**: Bilimsel makaleler
- **Kenarlar**: Alıntı ilişkileri
- **Özellikler**: 1433 boyutlu kelime vektörleri (binary)
- **Etiketler**: 7 farklı konu sınıfı

## 🔬 Uygulanan Yöntemler

### Klasik Algoritmalar (Neo4j GDS)

1. **Louvain Algoritması**
   - Modülerlik optimizasyonuna dayalı
   - Hızlı ve etkili
   - Hiyerarşik topluluk yapısı

2. **Label Propagation Algorithm (LPA)**
   - Etiket yayılımı prensibi
   - Basit ve hızlı
   - Deterministik olmayan

### Graph Neural Networks (PyTorch Geometric)

1. **Graph Convolutional Network (GCN)**
   - Spektral graph convolution
   - Yerel komşuluk bilgisi

2. **GraphSAGE**
   - Sampling ve aggregation
   - Büyük graphlar için ölçeklenebilir

3. **Graph Attention Network (GAT)**
   - Attention mechanism
   - Adaptif komşuluk ağırlıkları

## 📈 Değerlendirme Metrikleri

- **Adjusted Rand Index (ARI)**: Kümeleme kalitesi
- **Normalized Mutual Information (NMI)**: Bilgi teorisi tabanlı
- **Modularity**: Ağ yapısına uygunluk
- **Community Count**: Tespit edilen topluluk sayısı

## 🎨 Görselleştirme Özellikleri

- Veriset istatistikleri
- Topluluk boyut dağılımları
- Performans karşılaştırma grafikleri
- İnteraktif ağ görselleştirmeleri
- Kapsamlı dashboard'lar

## 📁 Çıktılar

Proje sonrasında `results/` klasöründe şunlar oluşturulur:

- **Grafikler**: PNG formatında statik görselleştirmeler
- **İnteraktif Grafikler**: HTML formatında etkileşimli dashboardlar
- **Sonuç Özetleri**: JSON ve CSV formatında detaylı sonuçlar
- **Log Dosyaları**: Tüm işlemlerin kaydı

## 🔧 Konfigürasyon

`.env` dosyasından temel ayarları değiştirebilirsiniz:

```env
# Neo4j Ayarları
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123

# Model Ayarları
HIDDEN_CHANNELS=64
NUM_LAYERS=2
LEARNING_RATE=0.01
EPOCHS=200
```

## 📚 Teorik Arkaplan

### Çizge Teorisi Kavramları

- **Modülerlik**: Bir ağın topluluk yapısının kalitesini ölçen metrik
- **Merkezilik**: Düğümlerin ağdaki önemini belirleyen ölçütler
- **Spektral Analiz**: Çizge matrislerinin özdeğer analizi

### Machine Learning Yaklaşımları

- **Unsupervised Learning**: Etiket olmadan öğrenme
- **Semi-supervised Learning**: Kısmi etiketli veri ile öğrenme
- **Graph Representation Learning**: Düğüm ve çizge temsillerini öğrenme

## 🚀 Gelişmiş Kullanım

### Farklı Verisetleri Kullanma

```python
from src.data_loader import CoraDataLoader

# Kendi verisetinizi yüklemek için data_loader.py'yi modifiye edin
```

### Model Parametrelerini Değiştirme

```python
from src.config import MODEL_CONFIG

MODEL_CONFIG["hidden_channels"] = 128
MODEL_CONFIG["num_layers"] = 3
```

### Yeni GNN Modelleri Ekleme

`src/gnn_models.py` dosyasına yeni model sınıfları ekleyebilirsiniz.

## 📖 Kaynaklar

- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Community Detection Algorithms](https://arxiv.org/abs/0906.0612)
- [Graph Neural Networks](https://arxiv.org/abs/1901.00596)

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👥 İletişim

Proje ile ilgili sorularınız için issue açabilir veya doğrudan iletişime geçebilirsiniz.

---

**Not**: Bu proje yüksek lisans düzeyinde bir ders için hazırlanmış olup, hem teorik altyapı hem de pratik uygulama içermektedir. Graph Theory ve Modern Machine Learning tekniklerini birleştiren kapsamlı bir örnektir. 