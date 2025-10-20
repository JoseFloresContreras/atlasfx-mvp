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

## Opción 2: Almacenamiento Local + Git LFS

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

## Opción 3: Almacenamiento Local + Base de Datos Especializada (TimescaleDB/InfluxDB)

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

| Criterio | DVC + S3/GCS | Git LFS | TimescaleDB |
|----------|-------------|---------|-------------|
| **Costo (MVP)** | $7-10/mes | $20/mes | $5/mes (local) |
| **Escalabilidad** | ⭐⭐⭐⭐⭐ Excelente | ⭐⭐ Limitada | ⭐⭐⭐⭐ Muy buena |
| **Versionado** | ⭐⭐⭐⭐⭐ Nativo | ⭐⭐⭐ Básico | ⭐⭐ Manual |
| **Performance** | ⭐⭐⭐⭐ Muy bueno | ⭐⭐⭐ Bueno | ⭐⭐⭐⭐⭐ Excelente |
| **Facilidad Setup** | ⭐⭐⭐⭐ Fácil | ⭐⭐⭐⭐⭐ Muy fácil | ⭐⭐ Complejo |
| **Reproducibilidad** | ⭐⭐⭐⭐⭐ Perfecta | ⭐⭐⭐ Buena | ⭐⭐⭐ Requiere disciplina |
| **ML Ecosystem** | ⭐⭐⭐⭐⭐ Perfecto | ⭐⭐⭐ Aceptable | ⭐⭐⭐ Custom integration |
| **Backup** | ⭐⭐⭐⭐⭐ Automático | ⭐⭐⭐⭐ Incluido | ⭐⭐ Manual |
| **Complejidad** | ⭐⭐⭐ Media | ⭐⭐⭐⭐ Baja | ⭐⭐ Alta |

---

## Recomendación Final

### 🥇 **Ganador: DVC + AWS S3 (o Google Cloud Storage)**

#### Justificación

Para el MVP de AtlasFX, **DVC + S3** es la mejor opción porque:

1. **Reproducibilidad Científica:**
   - El proyecto requiere estándares de nivel doctoral
   - DVC provee versionado inmutable de datasets
   - Puedes volver a cualquier experimento histórico

2. **Costo-Efectivo:**
   - ~$7-10/mes es muy económico para MVP
   - Escala linealmente (pagas solo lo que usas)
   - No hay sorpresas en la factura

3. **Ecosystem ML:**
   - Integración nativa con MLflow, Weights & Biases
   - Compatible con PyTorch DataLoaders
   - Soporte para Parquet, HDF5 (formatos eficientes)

4. **Escalabilidad:**
   - De MVP a producción sin cambios de arquitectura
   - S3 puede manejar petabytes si es necesario
   - Alta disponibilidad y durabilidad

5. **Bajo Riesgo:**
   - Backup automático (99.999999999% durability)
   - No dependes de una sola máquina
   - Fácil colaboración (múltiples desarrolladores)

### Implementación Recomendada (Fase 1)

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

# Semana 3+: Workflow normal
# 1. Procesar datos
python data-pipeline/pipeline.py

# 2. Versionar resultados
dvc add data/features/v1.0.parquet
git add data/features/v1.0.parquet.dvc
git commit -m "Add features v1.0"
dvc push

# 3. Entrenar modelo
python train_vae.py --data-version v1.0

# 4. Versionar modelo
dvc add models/vae_best.pth
git add models/vae_best.pth.dvc
git commit -m "Add trained VAE"
dvc push
```

### Migración Futura (Post-MVP)

Cuando el proyecto pase a producción:
1. **Mantener DVC + S3** para:
   - Versionado de datasets de entrenamiento
   - Reproducibilidad de experimentos
   - Backup de modelos

2. **Agregar TimescaleDB** para:
   - Datos en tiempo real (tick stream)
   - Queries de backtesting rápidos
   - Monitoreo de performance en producción

3. **Arquitectura Híbrida:**
   ```
   Research/Training: DVC + S3 (versionado)
   Production/Live: TimescaleDB (latencia baja)
   Backup: S3 Glacier (archival)
   ```

### Alternativa Económica (Si presupuesto es $0)

Si no puedes pagar $10/mes:
1. **Usar almacenamiento local** con DVC (sin remote)
2. **Backups manuales** a disco externo
3. **Git LFS** solo para datos críticos (<5GB)
4. **Migrar a S3** cuando tengas presupuesto

---

## Conclusión

**Para AtlasFX MVP, la recomendación es:**

### ✅ Implementar: DVC + AWS S3

**Razones:**
- Mejor balance costo/beneficio/complejidad
- Alineado con estándares de ML research
- Escalable de 120GB a terabytes
- Integración con herramientas modernas de ML
- Reproducibilidad garantizada

**Next Steps:**
1. Crear cuenta AWS (free tier: 5GB gratis)
2. Setup S3 bucket con lifecycle policies
3. Instalar y configurar DVC
4. Migrar datos actuales
5. Documentar workflow en README

**Costo Total:**
- **MVP (primeros 6 meses):** ~$50-60
- **Producción (año 1):** ~$100-200/año

**ROI:** El tiempo ahorrado en debugging y reproducibilidad vale mucho más que $10/mes.

---

**Autor:** Análisis para AtlasFX MVP  
**Fecha:** Octubre 2025  
**Versión:** 1.0
