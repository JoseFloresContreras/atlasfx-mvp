# Dónde Almacenar los Datos - Análisis de Opciones

**Fecha:** 20 de Octubre, 2025  
**Versión:** 1.0  
**Propósito:** Evaluar las 3 mejores opciones para almacenar datos del proyecto AtlasFX MVP

---

## Contexto del Proyecto

AtlasFX es un sistema de trading algorítmico que utiliza:
- **Datos Level 1 tick data** de Dukascopy (forex)
- **Pipeline de datos** que procesa, limpia, agrega y genera features
- **Modelos de ML** (VAE, TFT, SAC) que requieren acceso rápido a datos procesados
- **Experimentos ML** que necesitan versionado de datos
- **Backtesting** que requiere acceso histórico completo

### Requerimientos de Almacenamiento

1. **Volumen de Datos:**
   - Tick data crudo: ~100GB+ por año (7 pares de forex)
   - Datos agregados: ~10GB por año
   - Features procesadas: ~5GB por año
   - Modelos entrenados: ~1-5GB

2. **Patrones de Acceso:**
   - Lectura secuencial para entrenamiento (alto throughput)
   - Lectura aleatoria para backtesting (baja latencia)
   - Escritura por lotes (pipeline de datos)
   - Versionado de datos y modelos

3. **Requisitos No Funcionales:**
   - Reproducibilidad (versiones inmutables)
   - Costo-efectivo para desarrollo MVP
   - Escalable para producción futura
   - Compatible con Python/PyTorch ecosystem
   - Backup y recuperación

---

## Opción 1: DVC + Almacenamiento en la Nube (S3/Google Cloud Storage)

### Descripción

**DVC (Data Version Control)** es una herramienta de versionado de datos diseñada para proyectos de ML. Funciona como Git pero para archivos grandes, almacenando metadata en Git y datos reales en almacenamiento remoto (S3, GCS, Azure Blob).

### Arquitectura Propuesta

```
Local Development
├── data/                    # DVC-tracked data (metadata in Git)
│   ├── raw/                 # Tick data from Dukascopy
│   ├── processed/           # Aggregated time-series
│   ├── features/            # Feature matrices
│   └── splits/              # Train/val/test splits
├── models/                  # DVC-tracked model checkpoints
├── .dvc/                    # DVC configuration
└── .git/                    # Git repository

Remote Storage (S3/GCS)
└── atlasfx-data/
    ├── data/                # Actual data files
    └── models/              # Actual model files
```

### Ventajas ✅

1. **Versionado Nativo:**
   - Cada versión de datos tiene un hash único
   - Puedes volver a cualquier versión histórica
   - Ideal para reproducibilidad científica

2. **Integración con Git:**
   - Los cambios en datos se trackean junto con código
   - PRs pueden incluir cambios de datos y código
   - Workflow familiar para desarrolladores

3. **Escalabilidad:**
   - Almacenamiento en la nube prácticamente ilimitado
   - S3/GCS son altamente disponibles y durables (99.999999999%)
   - Fácil escalar de MVP a producción

4. **Costo-Efectivo:**
   - S3 Standard: ~$0.023/GB/mes (primeros 50TB)
   - S3 Glacier (archival): ~$0.004/GB/mes para datos antiguos
   - GCS Standard: ~$0.020/GB/mes
   - Solo pagas por lo que usas

5. **Ecosystem ML:**
   - Compatibilidad nativa con MLflow, Weights & Biases
   - Soporte para Parquet, HDF5, CSV
   - Pipelines de datos reproducibles

6. **Backup Automático:**
   - Los datos en S3/GCS están replicados automáticamente
   - Versionado inmutable (no puedes borrar por accidente)

### Desventajas ⚠️

1. **Latencia de Red:**
   - Primer acceso requiere download de datos
   - Necesitas buena conexión a internet
   - Cache local ayuda pero ocupa disco

2. **Costos de Transferencia:**
   - Download desde S3: ~$0.09/GB (después de 100GB/mes gratis)
   - GCS: ~$0.12/GB
   - Puede acumularse con experimentos frecuentes

3. **Complejidad Inicial:**
   - Curva de aprendizaje de DVC
   - Setup de credenciales AWS/GCP
   - Configuración de permisos (IAM)

4. **Dependencia de Terceros:**
   - Requiere cuenta AWS/GCP
   - Dependes de la disponibilidad del servicio
   - Costos pueden cambiar con el tiempo

### Implementación

```bash
# Instalar DVC
pip install dvc[s3]  # o dvc[gs] para Google Cloud

# Inicializar DVC
dvc init

# Configurar remote storage
dvc remote add -d storage s3://atlasfx-data/dvc-store
dvc remote modify storage region us-east-1

# Trackear datos
dvc add data/raw/eurusd_2024.parquet
git add data/raw/eurusd_2024.parquet.dvc .gitignore
git commit -m "Add raw EUR/USD data"

# Push a remote
dvc push

# Pull en otra máquina
dvc pull
```

### Costo Estimado (Mensual)

- **Almacenamiento:** 120GB × $0.023 = $2.76/mes
- **Transferencia:** 50GB/mes × $0.09 = $4.50/mes
- **Total:** ~$7-10/mes para MVP

### Recomendación

⭐⭐⭐⭐⭐ **ALTAMENTE RECOMENDADO**

**Caso de Uso Ideal:**
- Proyectos ML que requieren reproducibilidad
- Equipos distribuidos (o trabajo en múltiples máquinas)
- Escalabilidad de MVP a producción
- Presupuesto moderado ($10-50/mes)

---

## Opción 2: Almacenamiento Local Puro

### Descripción

**Almacenamiento completamente local** en tu PC dedicado, sin sincronización con cloud. Los datos viven únicamente en discos locales con backups manuales a discos externos.

### Especificaciones del Sistema

**Hardware Disponible:**
- 1TB de espacio exclusivo para el proyecto
- SSD o HDD (TBD - importante para performance)

**Estimaciones de Datos:**
- **Raw tick data:** ~100GB (7 pares + 3 instrumentos, 4 años)
- **Aggregated k-lines:** ~10-20GB (1-10 min intervals)
- **Features procesadas:** ~5-10GB (200-500 features estimadas)
- **Modelos entrenados:** ~5GB (VAE, TFT, SAC checkpoints)
- **Experiments logs:** ~10GB (MLflow/W&B artifacts)
- **Total estimado:** ~130-145GB

**Espacio disponible después:** ~855-870GB (suficiente para crecer 6-7x)

### Arquitectura Propuesta

```
/home/user/atlasfx-data/          # Directorio raíz (1TB disk)
├── raw/                           # Raw tick data (~100GB)
│   ├── forex/
│   │   ├── EURUSD/
│   │   ├── GBPUSD/
│   │   └── ...
│   └── instruments/
│       ├── XAUUSD/  # Gold
│       └── ...
├── processed/                     # Aggregated k-lines (~10-20GB)
│   ├── 1min/
│   ├── 5min/
│   └── 10min/
├── features/                      # Feature matrices (~5-10GB)
│   ├── train/
│   ├── val/
│   └── test/
├── models/                        # Trained models (~5GB)
│   ├── vae/
│   ├── tft/
│   └── sac/
├── experiments/                   # MLflow logs (~10GB)
│   └── mlruns/
└── backups/                       # Manual backups
    └── weekly_snapshots/
```

### Ventajas ✅

1. **Costo Cero**
   - No hay costos mensuales de cloud
   - Hardware ya disponible
   - Ideal para MVP con presupuesto limitado

2. **Velocidad de Acceso**
   - Latencia ultra-baja (especialmente con SSD)
   - No dependes de ancho de banda de internet
   - I/O directo sin API calls

3. **Privacidad Total**
   - Datos no salen de tu máquina
   - No requieres cuenta AWS/GCP
   - No hay riesgos de leak de datos

4. **Simplicidad**
   - No requiere setup de DVC, S3, credenciales
   - File system normal (cp, mv, rsync)
   - Menos moving parts = menos cosas que romper

5. **Sin Límites de Transferencia**
   - No pagas por download/upload
   - Puedes iterar libremente sin preocuparte por costos
   - Ideal para fase de experimentación intensiva

6. **Patrón de Acceso Ideal**
   - ✅ Raw data se accede **UNA VEZ** durante pipeline
   - ✅ Features procesadas se acceden durante entrenamiento
   - ✅ Después del pipeline, raw data queda en "cold storage"
   - ✅ No necesitas acceso frecuente después de generar features

### Desventajas ❌

1. **No Hay Versionado Nativo**
   - Sin DVC, no tienes versiones automáticas de datasets
   - Debes implementar versionado manual (timestamps, git tags)
   - Reproducibilidad requiere disciplina

2. **Riesgo de Pérdida de Datos**
   - Si el disco se daña, pierdes todo
   - Requiere estrategia de backup robusta
   - Single point of failure

3. **No Colaborativo**
   - Solo tú tienes acceso a los datos
   - Si trabajas en otra máquina, debes copiar manualmente
   - No hay sincronización automática

4. **No Escalable a Producción**
   - En producción necesitarás cloud storage
   - Migración futura será necesaria
   - No hay path directo de local a prod

5. **Backup Manual**
   - Debes recordar hacer backups
   - Proceso tedioso (rsync, cp a disco externo)
   - No hay snapshots automáticos

### Implementación

#### 1. Setup del Sistema de Archivos

```bash
# Crear estructura de directorios
mkdir -p ~/atlasfx-data/{raw,processed,features,models,experiments,backups}

# Crear subdirectorios para raw data
mkdir -p ~/atlasfx-data/raw/forex/{EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,NZDUSD}
mkdir -p ~/atlasfx-data/raw/instruments/{XAUUSD,WTI,ES}

# Crear subdirectorios para processed data
mkdir -p ~/atlasfx-data/processed/{1min,5min,10min}

# Crear subdirectorios para features
mkdir -p ~/atlasfx-data/features/{train,val,test}
```

#### 2. Versionado Manual con Git Tags

```bash
# Después de generar features
cd ~/atlasfx-data/features

# Crear snapshot con hash
SNAPSHOT_HASH=$(find . -type f -exec md5sum {} + | sort | md5sum | cut -d' ' -f1)
echo $SNAPSHOT_HASH > VERSION.txt

# Tag en el repo de código
cd ~/atlasfx-mvp
git tag -a "data-v1.0-${SNAPSHOT_HASH:0:8}" -m "Features for experiment v1.0"
git push --tags

# Documentar en experiment config
echo "data_version: v1.0-${SNAPSHOT_HASH:0:8}" >> configs/experiment.yaml
```

#### 3. Estrategia de Backup

**Opción A: Backup a Disco Externo (Semanal)**
```bash
#!/bin/bash
# backup_weekly.sh

SOURCE_DIR=~/atlasfx-data
BACKUP_DIR=/mnt/external-drive/atlasfx-backups
DATE=$(date +%Y-%m-%d)

# Crear backup con rsync
rsync -avh --progress \
  --exclude 'experiments/*' \
  --exclude '.cache/*' \
  $SOURCE_DIR/ $BACKUP_DIR/backup-$DATE/

# Mantener solo últimas 4 semanas
ls -t $BACKUP_DIR | tail -n +5 | xargs -I {} rm -rf $BACKUP_DIR/{}

echo "Backup completado: $BACKUP_DIR/backup-$DATE/"
```

**Opción B: Backup Selectivo (Solo Importantes)**
```bash
#!/bin/bash
# backup_important.sh

# Solo backupear datos críticos (features y modelos)
IMPORTANT_DIRS=(
  "features"
  "models"
  "processed"  # Optional, se puede regenerar
)

for dir in "${IMPORTANT_DIRS[@]}"; do
  tar -czf ~/backups/atlasfx-${dir}-$(date +%Y%m%d).tar.gz \
    ~/atlasfx-data/$dir
done
```

#### 4. Reproducibilidad sin DVC

```python
# src/atlasfx/utils/versioning.py
import hashlib
import json
from pathlib import Path
from typing import Dict

class LocalDataVersioning:
    """Manual data versioning for local storage."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.manifest_path = data_dir / "MANIFEST.json"
    
    def compute_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def create_manifest(self) -> Dict[str, str]:
        """Create manifest with hashes of all files."""
        manifest = {}
        for file_path in self.data_dir.rglob("*.parquet"):
            rel_path = file_path.relative_to(self.data_dir)
            manifest[str(rel_path)] = self.compute_hash(file_path)
        
        # Save manifest
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        return manifest
    
    def verify_manifest(self) -> bool:
        """Verify data integrity against manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError("Manifest not found")
        
        with open(self.manifest_path) as f:
            manifest = json.load(f)
        
        for rel_path, expected_hash in manifest.items():
            file_path = self.data_dir / rel_path
            if not file_path.exists():
                print(f"❌ Missing: {rel_path}")
                return False
            
            actual_hash = self.compute_hash(file_path)
            if actual_hash != expected_hash:
                print(f"❌ Corrupted: {rel_path}")
                return False
        
        print("✅ All files verified")
        return True
```

#### 5. Uso en Experiments

```python
# scripts/train_vae.py
from atlasfx.utils.versioning import LocalDataVersioning
from pathlib import Path

# Verificar integridad de datos antes de entrenar
data_dir = Path("~/atlasfx-data/features").expanduser()
versioning = LocalDataVersioning(data_dir)

if versioning.manifest_path.exists():
    print("Verificando integridad de datos...")
    if not versioning.verify_manifest():
        raise ValueError("Data integrity check failed!")
else:
    print("Creando manifest de datos...")
    versioning.create_manifest()

# Proceder con entrenamiento
train_vae(data_dir)
```

### Análisis de Performance

#### SSD vs. HDD

**Para Raw Tick Data (100GB):**
- **HDD (7200 RPM):**
  - Sequential read: ~150 MB/s
  - Tiempo de carga: ~11 minutos
  - ✅ Suficientemente rápido (loading se hace una vez)

- **SSD (SATA):**
  - Sequential read: ~500 MB/s
  - Tiempo de carga: ~3 minutos
  - ⭐ Ideal pero no crítico

- **NVMe SSD:**
  - Sequential read: ~3500 MB/s
  - Tiempo de carga: ~30 segundos
  - 🚀 Overkill para este proyecto

**Recomendación:**
- **Raw data:** Puede estar en HDD (se accede raramente)
- **Processed features:** Mejor en SSD (acceso frecuente durante training)
- **Models & experiments:** SSD (I/O constante)

**Setup Ideal:**
```
HDD 1TB:
  └── raw/          # Cold storage, acceso infrecuente

SSD 500GB:
  ├── processed/    # Warm storage
  ├── features/     # Hot storage
  ├── models/       # Hot storage
  └── experiments/  # Hot storage
```

### Costo Estimado

**One-time:**
- SSD 500GB: $50-70 (si no tienes)
- Disco externo 1TB (backups): $50

**Mensual:**
- Electricidad: ~$2-3/mes (PC encendida 8h/día)
- **Total:** ~$2-3/mes

**Comparado con cloud:**
- DVC + S3: ~$7-10/mes
- **Ahorro:** ~$5-7/mes (~$60-84/año)

### Recomendación

⭐⭐⭐⭐ **RECOMENDADO PARA MVP**

**Caso de Uso Ideal:**
- ✅ Trabajo individual (no colaboración)
- ✅ Presupuesto cero para MVP
- ✅ Datos se acceden una vez (pipeline) y luego poco
- ✅ Tienes 1TB disponible (espacio suficiente)
- ✅ Hardware ya disponible

**Por qué es ideal para tu caso:**

1. **Patrón de Acceso:**
   - Raw data (100GB) se procesa UNA VEZ
   - Después solo accedes a features procesadas (~10GB)
   - 90% del espacio (raw data) queda en "cold storage"

2. **Capacidad Suficiente:**
   - 1TB >> 145GB necesarios (7x margen)
   - Espacio para crecer a 500+ features

3. **MVP Scope:**
   - No necesitas colaboración aún
   - Reproducibilidad manual es suficiente
   - Puedes migrar a cloud post-MVP

4. **Costo-Efectivo:**
   - $0/mes vs. $7-10/mes S3
   - Ahorro de $60-84/año para MVP

**Cuándo migrar a cloud:**
- Cuando empieces a colaborar con otros
- Cuando necesites acceder desde múltiples máquinas
- Cuando vayas a producción
- Cuando el costo de tu tiempo > $10/mes

---

## Opción 3: Git LFS (Large File Storage)

### Descripción

**Git LFS (Large File Storage)** es una extensión de Git que maneja archivos grandes. Los archivos grandes se almacenan en un servidor LFS (GitHub, GitLab, Bitbucket) y Git solo guarda punteros.

### Arquitectura Propuesta

```
Repository
├── data/                    # Git LFS tracked
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/                  # Git LFS tracked
├── .gitattributes           # LFS configuration
└── .git/
    └── lfs/                 # LFS cache
```

### Ventajas ✅

1. **Simplicidad:**
   - Git workflow normal
   - No requiere setup de cloud
   - Una sola herramienta (Git)

2. **Integración GitHub:**
   - Los datos viven en el mismo repo
   - Clone = código + datos
   - Fácil para colaboradores nuevos

3. **Costo Inicial Bajo:**
   - GitHub: 1GB gratis, $5/mes por 50GB
   - GitLab: 10GB gratis en plan free
   - Bitbucket: 1GB gratis

4. **Sin Dependencias Externas:**
   - No necesitas cuenta AWS/GCP
   - Todo en GitHub/GitLab
   - Menos servicios que administrar

### Desventajas ❌

1. **Limitaciones de Tamaño:**
   - GitHub LFS: Límite de 2GB por archivo
   - 1GB de ancho de banda gratis/mes
   - Muy limitado para datos de trading (>100GB)

2. **Costo Escalable:**
   - GitHub: $5/mes por 50GB adicionales
   - Para 200GB: ~$20/mes (más caro que S3)
   - Bandwidth charges pueden ser altos

3. **Performance:**
   - Más lento que S3 para archivos grandes
   - Clone inicial puede tardar horas
   - No optimizado para ML workloads

4. **No Es Verdadero Versionado:**
   - Git LFS puede tener conflictos
   - Eliminar versiones antiguas es complejo
   - No está diseñado para datasets grandes

5. **Límites de GitHub:**
   - Repository size max: 100GB (recomendado <5GB)
   - Warnings después de 1GB
   - No escalable para producción

### Implementación

```bash
# Instalar Git LFS
git lfs install

# Trackear archivos grandes
git lfs track "*.parquet"
git lfs track "*.h5"
git lfs track "*.pth"

# Agregar a Git
git add .gitattributes
git add data/raw/eurusd_2024.parquet
git commit -m "Add data with LFS"
git push
```

### Costo Estimado (Mensual)

- **Almacenamiento:** 120GB → $15/mes (GitHub packs)
- **Bandwidth:** 50GB/mes → $5/mes
- **Total:** ~$20/mes (más caro que S3)

### Recomendación

⭐⭐ **NO RECOMENDADO PARA ESTE PROYECTO**

**Caso de Uso Ideal:**
- Proyectos pequeños (<50GB)
- Equipos que solo usan GitHub
- Datos que cambian poco
- Prototipado rápido

**Por qué no para AtlasFX:**
- Los datos de tick superan los límites de GitHub LFS
- Más caro que S3 para volúmenes grandes
- No optimizado para ML workflows
- Escalabilidad limitada

---

## Opción 4: Base de Datos Especializada (TimescaleDB/InfluxDB)

### Descripción

Usar una **base de datos de series temporales** optimizada para datos financieros. TimescaleDB (PostgreSQL) o InfluxDB están diseñadas para manejar millones de ticks con queries eficientes.

### Arquitectura Propuesta

```
Database Server (Local o Cloud)
└── TimescaleDB
    ├── tick_data              # Raw ticks (hypertable)
    │   ├── timestamp
    │   ├── symbol
    │   ├── bid
    │   ├── ask
    │   └── volume
    ├── aggregated_ohlc        # Pre-aggregated (hypertable)
    └── features               # Processed features

Application Layer
├── data-pipeline/
│   └── db_loader.py          # ETL to database
├── models/
│   └── data_loader.py        # Query features from DB
└── experiments/
    └── mlflow/               # Model versioning with MLflow
```

### Ventajas ✅

1. **Performance Optimizado:**
   - Queries por rango temporal son ultra-rápidas
   - Índices automáticos en timestamp
   - Compresión de datos (TimescaleDB: 90%+ compression)
   - Agregaciones nativas (OHLC, VWAP)

2. **Queries SQL Potentes:**
   ```sql
   -- Obtener datos de entrenamiento
   SELECT * FROM features 
   WHERE timestamp BETWEEN '2024-01-01' AND '2024-06-30'
   AND symbol = 'EURUSD'
   ORDER BY timestamp;
   
   -- Calcular VWAP on-the-fly
   SELECT time_bucket('5 minutes', timestamp) AS bucket,
          symbol,
          SUM(price * volume) / SUM(volume) AS vwap
   FROM tick_data
   GROUP BY bucket, symbol;
   ```

3. **Versionado con Timestamps:**
   - Datos inmutables (insert-only)
   - Puedes recrear estado en cualquier momento
   - Auditoría completa de cambios

4. **Escalabilidad Vertical:**
   - TimescaleDB escala a billones de filas
   - Puede manejar todo el histórico de Dukascopy
   - Particionamiento automático (time-based)

5. **Integración Python:**
   - SQLAlchemy, pandas, psycopg2
   - Direct SQL queries desde PyTorch DataLoader
   - Streaming de datos para entrenamiento

6. **Costo-Efectivo (Local):**
   - PostgreSQL/TimescaleDB es open-source
   - Puedes correr en tu propia máquina
   - No hay costos de cloud (si es local)

### Desventajas ⚠️

1. **Setup Complejo:**
   - Requiere administrar un servidor de BD
   - Configuración de índices y particiones
   - Backups manuales necesarios

2. **No Es Versionado Nativo:**
   - No puedes volver a "versiones" como DVC
   - Necesitas snapshots manuales
   - Reproducibilidad requiere disciplina

3. **Backup Crítico:**
   - Si la BD se corrompe, pierdes todo
   - Necesitas estrategia de backup robusta
   - Disaster recovery más complejo

4. **Memoria y Disco:**
   - Base de datos puede crecer mucho
   - Requiere SSD rápido para performance
   - Tuning de PostgreSQL necesario

5. **Limitaciones para ML:**
   - No integración nativa con MLflow/W&B
   - Feature store requiere desarrollo custom
   - Versionado de features no es trivial

6. **Escalabilidad Horizontal Limitada:**
   - Scaling a múltiples máquinas es complejo
   - No es serverless (requiere servidor running)

### Implementación

```python
# Conexión a TimescaleDB
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:pass@localhost/atlasfx')

# Cargar tick data
df = pd.read_parquet('eurusd_ticks.parquet')
df.to_sql('tick_data', engine, if_exists='append', index=False)

# Query para entrenamiento
query = """
SELECT * FROM features 
WHERE timestamp >= '2024-01-01'::timestamp
AND timestamp < '2024-07-01'::timestamp
ORDER BY timestamp
"""
train_data = pd.read_sql(query, engine)

# Crear hypertable (TimescaleDB)
engine.execute("""
    SELECT create_hypertable('tick_data', 'timestamp',
                             chunk_time_interval => INTERVAL '1 day');
""")
```

### Costo Estimado (Mensual)

**Opción Local:**
- Hardware: $0 (usa tu máquina)
- Almacenamiento: SSD de 512GB (~$50 one-time)
- Electricidad: ~$5/mes
- **Total:** ~$5/mes

**Opción Cloud (TimescaleDB Cloud):**
- Starter: $50/mes (25GB storage, 0.5 CPU)
- Growth: $200/mes (100GB storage, 2 CPU)
- **Total:** $50-200/mes (muy caro para MVP)

### Recomendación

⭐⭐⭐ **RECOMENDADO PARA CIERTOS CASOS**

**Caso de Uso Ideal:**
- Sistemas de trading en producción (latencia crítica)
- Necesitas queries complejas sobre ticks
- Acceso en tiempo real a datos históricos
- Infraestructura local (no cloud)

**Por qué podría no ser ideal para AtlasFX MVP:**
- Complejidad de setup vs. DVC
- No tiene versionado nativo de datasets
- Backup y disaster recovery más difícil
- Mejor para producción que para investigación

**Cuándo considerarlo:**
- Después del MVP, para deployment
- Si necesitas backtesting ultra-rápido
- Si tienes experiencia con bases de datos

---

## Comparación Final

| Criterio | DVC + S3/GCS | Local Puro | Git LFS | TimescaleDB |
|----------|-------------|------------|---------|-------------|
| **Costo (MVP)** | $7-10/mes | $2-3/mes | $20/mes | $5/mes (local) |
| **Escalabilidad** | ⭐⭐⭐⭐⭐ Excelente | ⭐⭐ Limitada | ⭐⭐ Limitada | ⭐⭐⭐⭐ Muy buena |
| **Versionado** | ⭐⭐⭐⭐⭐ Nativo | ⭐⭐ Manual | ⭐⭐⭐ Básico | ⭐⭐ Manual |
| **Performance** | ⭐⭐⭐⭐ Muy bueno | ⭐⭐⭐⭐⭐ Excelente | ⭐⭐⭐ Bueno | ⭐⭐⭐⭐⭐ Excelente |
| **Facilidad Setup** | ⭐⭐⭐⭐ Fácil | ⭐⭐⭐⭐⭐ Muy fácil | ⭐⭐⭐⭐⭐ Muy fácil | ⭐⭐ Complejo |
| **Reproducibilidad** | ⭐⭐⭐⭐⭐ Perfecta | ⭐⭐⭐ Requiere disciplina | ⭐⭐⭐ Buena | ⭐⭐⭐ Requiere disciplina |
| **Colaboración** | ⭐⭐⭐⭐⭐ Excelente | ⭐ Limitada | ⭐⭐⭐⭐ Buena | ⭐⭐ Media |
| **ML Ecosystem** | ⭐⭐⭐⭐⭐ Perfecto | ⭐⭐⭐ Aceptable | ⭐⭐⭐ Aceptable | ⭐⭐⭐ Custom integration |
| **Backup** | ⭐⭐⭐⭐⭐ Automático | ⭐⭐ Manual | ⭐⭐⭐⭐ Incluido | ⭐⭐ Manual |
| **Complejidad** | ⭐⭐⭐ Media | ⭐⭐⭐⭐⭐ Muy baja | ⭐⭐⭐⭐ Baja | ⭐⭐ Alta |
| **Acceso Infrec.** | ⭐⭐⭐ OK (latencia) | ⭐⭐⭐⭐⭐ Perfecto | ⭐⭐⭐ OK | ⭐⭐⭐⭐ Muy bueno |

---

## Recomendación Final

### Para MVP: Dos Opciones Válidas

Dependiendo de tus prioridades, hay **dos opciones recomendadas** para el MVP:

---

### 🥇 **Opción A: Almacenamiento Local Puro (Recomendado para MVP)**

#### Justificación

Para el **MVP de AtlasFX**, **almacenamiento local** es la mejor opción inicial porque:

1. **Patrón de Acceso Ideal:**
   - Raw data (100GB) se procesa **UNA VEZ** durante el pipeline
   - Después solo accedes a features procesadas (~10GB)
   - 90% del espacio queda en "cold storage" (no se usa frecuentemente)
   - Cita del problema statement: *"no es que los tengamos que usar constantemente ya que una vez que generemos el df del data pipeline ya no creo que los usemos mucho"*

2. **Capacidad Más que Suficiente:**
   - 1TB disponible >> 145GB necesarios (factor 7x)
   - Espacio para crecer features de 200 a 500+ sin problemas
   - Margen para experimentos, checkpoints, logs

3. **Costo Cero:**
   - No hay costos mensuales ($0 vs. $7-10/mes S3)
   - Hardware ya disponible
   - Ahorro de ~$60-84/año durante desarrollo MVP

4. **Simplicidad:**
   - No requiere setup de DVC, AWS, credenciales
   - File system normal (cp, mv, rsync)
   - Menos moving parts = menos cosas que romper
   - Más tiempo para enfocarte en algoritmos

5. **Performance Excelente:**
   - Latencia ultra-baja (I/O local)
   - No dependes de internet
   - Especialmente con SSD para features/models

6. **Viable para el Alcance del MVP:**
   - Trabajo individual (sin colaboración aún)
   - Reproducibilidad manual es suficiente
   - Path de migración claro a cloud post-MVP

#### Implementación Recomendada (MVP)

```bash
# Setup estructura local
~/atlasfx-data/
├── raw/              # 100GB - HDD OK (cold storage)
├── processed/        # 10-20GB - SSD recomendado
├── features/         # 5-10GB - SSD recomendado
├── models/           # 5GB - SSD recomendado
└── experiments/      # 10GB - SSD recomendado

# Versionado manual con manifests
python scripts/create_data_manifest.py

# Backup semanal a disco externo
rsync -avh ~/atlasfx-data/ /mnt/external/atlasfx-backups/
```

#### Cuándo Migrar a Cloud

Migrar a **DVC + S3** cuando:
- ✅ MVP completado y validado
- ✅ Necesites colaborar con otros
- ✅ Trabajes desde múltiples máquinas
- ✅ Vayas a producción
- ✅ Tu tiempo vale más que $10/mes

---

### 🥈 **Opción B: DVC + AWS S3 (Alternativa Profesional)**

#### Justificación

Si prefieres setup más profesional desde el inicio, **DVC + S3** sigue siendo excelente porque:

1. **Versionado Automático:**
   - Cada versión de datos tiene hash único
   - Puedes volver a cualquier versión histórica
   - Reproducibilidad perfecta sin esfuerzo manual

2. **Estándares de ML Research:**
   - Integración nativa con MLflow, W&B
   - Compatible con PyTorch DataLoaders
   - Usado por proyectos de nivel doctoral

3. **Escalabilidad:**
   - De MVP a producción sin cambios de arquitectura
   - S3 puede manejar petabytes si es necesario
   - Alta disponibilidad y durabilidad (99.999999999%)

4. **Backup Automático:**
   - Datos replicados automáticamente
   - No dependes de una sola máquina
   - Disaster recovery incluido

5. **Bajo Riesgo:**
   - Pérdida de datos es prácticamente imposible
   - No preocuparte por backups manuales

#### Costo Estimado

- **MVP (primeros 6 meses):** ~$50-60 ($7-10/mes)
- **Post-MVP (año 1):** ~$100-200/año

#### Implementación Recomendada

```bash
# Semana 1: Setup DVC
pip install dvc[s3]
dvc init
dvc remote add -d storage s3://atlasfx-mvp-data/dvc-store

# Semana 2: Migrar datos existentes
dvc add data/raw/
dvc add data/processed/
dvc push
git commit -m "Add data versioning with DVC"

# Workflow normal
python data-pipeline/pipeline.py
dvc add data/features/v1.0.parquet
git add data/features/v1.0.parquet.dvc
git commit -m "Add features v1.0"
dvc push
```

---

### 🎯 **Recomendación Personalizada para Tu Caso**

Basado en la información del problema statement:

**✅ USAR ALMACENAMIENTO LOCAL PURO PARA MVP**

**Razones específicas para tu caso:**

1. **Tienes hardware adecuado:**
   - 1TB exclusivo para proyecto ✅
   - Suficiente para 100GB raw + expansión

2. **Patrón de acceso confirma viabilidad:**
   - Raw data se usa una vez (pipeline) ✅
   - Después acceso mínimo ✅
   - No justifica costos de cloud

3. **Alcance del MVP:**
   - Trabajo individual (sin colaboración) ✅
   - Presupuesto limitado ✅
   - Foco en algoritmos, no infraestructura

4. **Path de migración claro:**
   - Empiezas local (rápido, barato)
   - Migras a DVC + S3 post-MVP si necesario
   - No pierdes tiempo/dinero en setup prematuro

**Plan de Implementación:**

```
MVP (Semanas 1-18):
  ✅ Almacenamiento local puro
  ✅ Versionado manual con manifests
  ✅ Backups semanales a disco externo
  
Post-MVP (Semanas 19+):
  Evaluar si migrar a DVC + S3 basado en:
    - ¿Necesitas colaboración?
    - ¿Múltiples máquinas?
    - ¿Producción?
    
  Si respuesta es NO → Mantener local
  Si respuesta es SÍ → Migrar a DVC + S3
```

---

### Arquitectura Híbrida (Futuro Post-MVP)

Cuando el proyecto pase a producción, la mejor estrategia es **híbrida**:

1. **Mantener Local para:**
   - Desarrollo e iteración rápida
   - Experiments que requieren I/O intensivo
   - Cache de datos frecuentemente usados

2. **Usar DVC + S3 para:**
   - Versionado de datasets de entrenamiento
   - Compartir datos con colaboradores
   - Backup y disaster recovery
   - Reproducibilidad científica

3. **Agregar TimescaleDB para:**
   - Datos en tiempo real (tick stream en producción)
   - Queries de backtesting rápidos
   - Monitoreo de performance en vivo

**Arquitectura Híbrida:**
```
Research/Training:
  ├── Local SSD (hot data, fast iteration)
  └── DVC + S3 (versioning, backup, sharing)

Production/Live:
  ├── TimescaleDB (real-time ticks, low latency)
  └── S3 (backup, archival)

Archival:
  └── S3 Glacier (cold storage, old experiments)
```

---

### Resumen de Decisión

| Fase | Recomendación | Razón |
|------|--------------|-------|
| **MVP (ahora)** | 🏆 **Local Puro** | Costo $0, simplicidad, hardware disponible, patrón de acceso ideal |
| **Post-MVP** | Evaluar migración a DVC + S3 | Si necesitas colaboración, reproducibilidad avanzada |
| **Producción** | Híbrido (Local + DVC + TimescaleDB) | Mejor de todos los mundos |

---

### Alternativa Económica (Si Prefieres Cloud desde el Inicio)

Si quieres empezar con cloud pero presupuesto es limitado:

**Opción Intermedia: DVC + Google Drive**
- Google Drive: 15GB gratis (suficiente para features procesadas)
- DVC soporta Google Drive como remote
- Mantén raw data local (100GB)
- Versionas solo datos procesados (~10GB)

```bash
# Setup DVC con Google Drive
pip install dvc[gdrive]
dvc remote add -d storage gdrive://folder-id
dvc remote modify storage gdrive_acknowledge_abuse true

# Solo versionar features, no raw data
dvc add data/features/
dvc push
```

**Costo:** $0/mes (hasta 15GB)

---

## Conclusión

**Para AtlasFX MVP, la recomendación es:**

### ✅ Implementar: Almacenamiento Local Puro

**Razones:**
- Hardware disponible (1TB) ✅
- Patrón de acceso ideal (una vez) ✅
- Costo $0 vs. $7-10/mes ✅
- Simplicidad máxima ✅
- Performance excelente ✅
- Path de migración claro ✅

**Next Steps:**
1. Crear estructura de directorios local
2. Implementar sistema de manifests para versioning manual
3. Setup script de backup automático (semanal)
4. Documentar convenciones de naming/versioning
5. Proceder con desarrollo del MVP

**Migración Futura (Post-MVP):**
- Evaluar necesidad basada en colaboración/producción
- Si necesario, migrar a DVC + S3
- Proceso de migración es straightforward (DVC facilita)

---

### 📊 Comparación Costo Total (3 años)

| Opción | Setup | Año 1 | Año 2 | Año 3 | Total 3 años |
|--------|-------|-------|-------|-------|--------------|
| **Local Puro** | $100 (SSD+backup disk) | $30 | $30 | $30 | **$190** |
| **DVC + S3** | $0 | $120 | $150 | $200 | **$470** |
| **Git LFS** | $0 | $240 | $300 | $360 | **$900** |
| **TimescaleDB Cloud** | $0 | $600 | $2400 | $2400 | **$5400** |

**Ahorro de Local vs. DVC:** $280 en 3 años  
**Ahorro de Local vs. Git LFS:** $710 en 3 años

**ROI:** Para MVP, el ahorro de $280 es significativo. Post-MVP, cuando generes value, $10/mes de S3 es despreciable.

---

**Autor:** Análisis actualizado para AtlasFX MVP  
**Fecha:** 20 de Octubre, 2025  
**Versión:** 2.0 (incluye análisis de almacenamiento local)

---

**Autor:** Análisis actualizado para AtlasFX MVP  
**Fecha:** 20 de Octubre, 2025  
**Versión:** 2.0 (incluye análisis de almacenamiento local)
