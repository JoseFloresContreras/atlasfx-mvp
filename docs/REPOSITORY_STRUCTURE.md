# AtlasFX - Análisis de Estructura del Repositorio

**Versión:** 1.0  
**Fecha:** 20 de Octubre, 2025  
**Propósito:** Evaluar la estructura ideal del repositorio y decidir si conviene adoptar la estructura propuesta

---

## Executive Summary

Este documento evalúa dos estructuras de repositorio:
1. **Estructura Actual** (post-cleanup, orientada a MVP)
2. **Estructura Propuesta** (tu documentación guardada)

**Recomendación:** ✅ **Adoptar estructura propuesta con modificaciones incrementales**

La estructura propuesta es más profesional, escalable y alineada con estándares de proyectos ML en producción. Sin embargo, se recomienda migrar de forma **incremental** durante las fases del MVP en vez de hacerlo todo de una vez.

---

## Estructura Actual (Post-Cleanup)

```
atlasfx-mvp/
├── data-pipeline/              # Pipeline de procesamiento de datos
│   ├── pipeline.py            # Orquestación
│   ├── merge.py               # Merging de ticks
│   ├── clean.py               # Limpieza
│   ├── aggregate.py           # Agregación temporal
│   ├── featurize.py           # Feature engineering
│   ├── featurizers.py         # Calculadores de features
│   ├── aggregators.py         # Funciones de agregación
│   ├── normalize.py           # Normalización
│   ├── split.py               # Train/val/test split
│   ├── winsorize.py           # Manejo de outliers
│   ├── visualize.py           # Visualización
│   ├── logger.py              # Logging
│   ├── pipeline.yaml          # Configuración
│   └── requirements.txt       # Dependencias
├── docs/                       # Documentación
│   ├── ARCHITECTURE.md
│   ├── AUDIT_REPORT.md
│   ├── FEATURES.md
│   ├── MVP_ACTION_PLAN.md
│   ├── NEXT_STEPS.md
│   ├── DONDE_ALMACENAR_DATOS.md
│   └── BACKLOG.md            # ✅ Nuevo
├── tests/                      # Tests
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── README.md
├── REFACTORING_SUMMARY.md
└── pyproject.toml
```

### Fortalezas ✅

1. **Simple y Enfocado**
   - Todo relacionado con el pipeline está en un lugar
   - Fácil de navegar para el MVP
   - Bajo overhead de estructura

2. **Documentación Completa**
   - Carpeta `docs/` bien organizada
   - Decision records claros

3. **Testing Infrastructure**
   - Tests separados por tipo (unit/integration)
   - Fixtures compartidas en conftest.py

### Debilidades ⚠️

1. **No Escalable**
   - `data-pipeline/` va a crecer mucho cuando agreguemos VAE, TFT, SAC
   - Todo mezclado en raíz (no hay separación clara)

2. **No Es un Paquete Instalable**
   - No hay `src/atlasfx/` → No puedes hacer `pip install .`
   - Dificulta imports relativos
   - No puedes instalar en otro proyecto

3. **Falta Modularización**
   - No hay separación entre:
     - Data loading
     - Model training
     - Model evaluation
     - Deployment/Execution

4. **No Sigue PEP Standards**
   - PEP 517/518 recomienda `src/` layout
   - Protege contra accidental imports de tests

---

## Estructura Propuesta (De Tu Documentación)

```
atlasfx/
├── atlasfx/                    # Paquete principal
│   ├── data/                  # Módulo de gestión de datos
│   │   ├── loader.py          # Carga de datos desde múltiples formatos
│   │   ├── preprocessor.py    # Preprocesamiento de datos
│   │   └── validator.py       # Validación de datos de tick
│   ├── models/                # Módulo de modelos
│   │   ├── base.py            # Clase base para modelos
│   │   └── ml_model.py        # Modelo de ML (Random Forest)
│   ├── training/              # Módulo de entrenamiento
│   │   └── trainer.py         # Entrenador de modelos
│   ├── evaluation/            # Módulo de evaluación
│   │   ├── evaluator.py       # Evaluador de modelos
│   │   └── metrics.py         # Métricas de trading
│   └── execution/             # Módulo de ejecución
│       └── executor.py        # Ejecutor de estrategias
├── tests/                     # Tests unitarios
├── config/                    # Archivos de configuración
├── schemas/                   # Esquemas de datos (YAML)
├── data/                      # Directorio de datos (no versionado)
├── models/                    # Modelos guardados (no versionado)
└── pyproject.toml            # Configuración del proyecto
```

### Análisis de la Propuesta

#### Fortalezas ✅

1. **Modular y Organizado**
   - Separación clara por responsabilidad
   - Cada módulo tiene un propósito único

2. **Escalable**
   - Fácil agregar nuevos modelos en `models/`
   - Fácil agregar nuevas métricas en `evaluation/`

3. **Profesional**
   - Sigue convenciones de proyectos ML
   - Similar a scikit-learn, PyTorch Lightning, etc.

4. **Instalable**
   - Puedes hacer `pip install -e .`
   - Imports limpios: `from atlasfx.models import VAE`

#### Debilidades ⚠️

1. **Ambigüedad en Nombres**
   - `ml_model.py` es genérico (¿qué modelo?)
   - `executor.py` podría confundirse con execution de código

2. **Falta de Contexto**
   - No menciona dónde van scripts de entrenamiento
   - No menciona notebooks exploratorios
   - No menciona experiments/logs

3. **Muy Genérica**
   - Diseñada para cualquier proyecto ML
   - No específica para trading/RL/time-series

---

## Estructura Recomendada (Híbrida Optimizada)

Combinando lo mejor de ambas + mejores prácticas de la industria:

```
atlasfx-mvp/
├── src/                           # Source code (PEP 517/518 layout)
│   └── atlasfx/                  # Paquete principal instalable
│       ├── __init__.py
│       ├── data/                 # Módulo de gestión de datos
│       │   ├── __init__.py
│       │   ├── loaders.py        # Carga desde Parquet, CSV, DVC
│       │   ├── processors.py     # Merge, clean, aggregate
│       │   ├── validators.py     # Validación de tick data
│       │   ├── featurizers.py    # Feature engineering
│       │   └── splitters.py      # Train/val/test splits
│       ├── models/               # Módulo de modelos DL
│       │   ├── __init__.py
│       │   ├── base.py           # Clase base AbstractModel
│       │   ├── vae.py            # Variational Autoencoder
│       │   ├── tft.py            # Temporal Fusion Transformer
│       │   └── sac.py            # Soft Actor-Critic agent
│       ├── training/             # Módulo de entrenamiento
│       │   ├── __init__.py
│       │   ├── trainers.py       # Trainer genérico
│       │   ├── vae_trainer.py    # VAE-specific trainer
│       │   ├── tft_trainer.py    # TFT-specific trainer
│       │   └── sac_trainer.py    # SAC-specific trainer
│       ├── evaluation/           # Módulo de evaluación
│       │   ├── __init__.py
│       │   ├── metrics.py        # Sharpe, drawdown, win rate
│       │   ├── backtester.py     # Backtesting engine
│       │   └── visualizers.py    # Plots y reportes
│       ├── environments/         # Trading environments (Gym)
│       │   ├── __init__.py
│       │   ├── trading_env.py    # Main trading environment
│       │   └── rewards.py        # Reward functions
│       ├── agents/               # RL agent wrappers
│       │   ├── __init__.py
│       │   └── sac_agent.py      # SAC wrapper para deployment
│       ├── utils/                # Utilidades compartidas
│       │   ├── __init__.py
│       │   ├── logging.py        # Logging utilities
│       │   ├── reproducibility.py # Seed fixing
│       │   └── io.py             # File I/O helpers
│       └── config/               # Configuration schemas
│           ├── __init__.py
│           └── schemas.py        # Pydantic configs
├── tests/                         # Tests (mirrors src/ structure)
│   ├── unit/
│   │   ├── data/
│   │   ├── models/
│   │   ├── training/
│   │   └── evaluation/
│   ├── integration/
│   │   ├── test_data_pipeline.py
│   │   └── test_training_loop.py
│   └── conftest.py               # Shared fixtures
├── scripts/                       # Scripts ejecutables
│   ├── run_data_pipeline.py      # Ejecutar pipeline de datos
│   ├── train_vae.py              # Entrenar VAE
│   ├── train_tft.py              # Entrenar TFT
│   ├── train_sac.py              # Entrenar SAC agent
│   ├── backtest.py               # Backtesting
│   └── deploy.py                 # Deployment a producción
├── notebooks/                     # Jupyter notebooks (exploratory)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_debugging.ipynb
├── configs/                       # Archivos de configuración YAML
│   ├── data_pipeline.yaml
│   ├── vae.yaml
│   ├── tft.yaml
│   └── sac.yaml
├── experiments/                   # MLflow / W&B logs
│   └── .gitkeep
├── data/                          # Datos (DVC tracked)
│   ├── raw/                      # Raw tick data
│   ├── processed/                # Cleaned & aggregated
│   ├── features/                 # Feature matrices
│   └── splits/                   # Train/val/test
├── models/                        # Modelos guardados (.pth)
│   ├── vae/
│   ├── tft/
│   └── sac/
├── docs/                          # Documentación
│   ├── ARCHITECTURE.md
│   ├── FEATURES.md
│   ├── MVP_ACTION_PLAN.md
│   ├── DONDE_ALMACENAR_DATOS.md
│   ├── BACKLOG.md               # ✅ Nuevo
│   ├── REPOSITORY_STRUCTURE.md  # ✅ Este documento
│   └── api/                      # Sphinx API docs (auto-generated)
├── .github/                       # GitHub workflows
│   └── workflows/
│       ├── ci.yml                # CI/CD pipeline
│       └── deploy.yml            # Deployment workflow
├── .gitignore
├── .pre-commit-config.yaml       # Pre-commit hooks
├── pyproject.toml                # Poetry config + tool configs
├── README.md
└── LICENSE
```

---

## Justificación de la Estructura Recomendada

### 1. `src/atlasfx/` Layout (PEP 517/518)

**Por qué:**
- Evita accidental imports de root
- Fuerza a instalar el paquete (`pip install -e .`)
- Protege contra import conflicts
- Permite imports limpios: `from atlasfx.models import VAE`

**Referencia:** [Python Packaging Guide - src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)

---

### 2. Separación `data/` vs. `models/` vs. `training/` vs. `evaluation/`

**Por qué:**
- **Single Responsibility Principle:** Cada módulo hace una cosa
- **Reusabilidad:** `models/` puede usarse sin `training/`
- **Testing:** Tests más focalizados y rápidos
- **Colaboración:** Equipos pueden trabajar en módulos diferentes

**Ejemplo:**
```python
# Limpio y modular
from atlasfx.data import load_tick_data, preprocess
from atlasfx.models import VAE
from atlasfx.training import VAETrainer

# vs. estructura actual (todo mezclado)
from data_pipeline.featurize import featurize  # ¿Dónde está VAE?
```

---

### 3. `scripts/` para Entry Points

**Por qué:**
- Separa lógica (en `src/`) de ejecución (en `scripts/`)
- Scripts son simples: parsean args, llaman a funciones
- Lógica compleja vive en `src/` y está testeada

**Ejemplo:**
```python
# scripts/train_vae.py
from atlasfx.training import VAETrainer
from atlasfx.config import VAEConfig

def main():
    config = VAEConfig.from_yaml("configs/vae.yaml")
    trainer = VAETrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
```

---

### 4. `configs/` Separado

**Por qué:**
- Configs son "datos" no código
- Fácil versionar diferentes experimentos
- No mezclados con código

**Estructura:**
```yaml
# configs/vae.yaml
model:
  input_dim: 200
  latent_dim: 64
  hidden_dims: [128, 64]
  beta: 1.0

training:
  learning_rate: 0.001
  batch_size: 256
  epochs: 100
```

---

### 5. `notebooks/` para Exploración

**Por qué:**
- Notebooks son para análisis exploratorio
- No son parte del código productivo
- Gitignore outputs, versionar solo `.ipynb`

**Convención:**
- Nombres con prefijo numérico: `01_`, `02_`, etc.
- Documentar propósito en primera celda

---

### 6. `experiments/` para MLflow/W&B

**Por qué:**
- Logs de experimentos separados del código
- Gitignore esta carpeta (muy grande)
- Solo versionar configs y resultados finales

---

### 7. `tests/` que Espeja `src/`

**Por qué:**
- Fácil encontrar tests de un módulo
- Estructura clara: `tests/unit/data/test_loaders.py` ↔ `src/atlasfx/data/loaders.py`

---

## Plan de Migración Incremental

### Fase 1: Setup Básico (Semana 1-2 del MVP)
```bash
# Crear estructura base
mkdir -p src/atlasfx/{data,models,training,evaluation,environments,agents,utils,config}
mkdir -p scripts configs notebooks experiments

# Mover pyproject.toml config
# Configurar Poetry con src/ layout
```

**Deliverables:**
- [ ] Estructura de carpetas creada
- [ ] `pyproject.toml` actualizado
- [ ] `src/atlasfx/__init__.py` creado

---

### Fase 2: Migrar Data Pipeline (Semana 3-4 del MVP)
```bash
# Mover módulos de data-pipeline/ a src/atlasfx/data/
mv data-pipeline/merge.py src/atlasfx/data/loaders.py  # refactor
mv data-pipeline/clean.py src/atlasfx/data/processors.py  # refactor
mv data-pipeline/featurize.py src/atlasfx/data/featurizers.py
# ... etc
```

**Deliverables:**
- [ ] Todo en `data-pipeline/` migrado a `src/atlasfx/data/`
- [ ] Imports actualizados
- [ ] Tests actualizados
- [ ] Script `scripts/run_data_pipeline.py` creado

---

### Fase 3: Agregar Modelos (Semana 5-8 del MVP)
```bash
# Implementar VAE y TFT
touch src/atlasfx/models/{base.py,vae.py,tft.py}
touch src/atlasfx/training/{trainers.py,vae_trainer.py,tft_trainer.py}
```

**Deliverables:**
- [ ] VAE implementado en `src/atlasfx/models/vae.py`
- [ ] TFT implementado en `src/atlasfx/models/tft.py`
- [ ] Trainers implementados
- [ ] Scripts de entrenamiento en `scripts/`

---

### Fase 4: Agregar SAC y Environment (Semana 9-10 del MVP)
```bash
touch src/atlasfx/models/sac.py
touch src/atlasfx/environments/trading_env.py
touch src/atlasfx/agents/sac_agent.py
```

**Deliverables:**
- [ ] SAC implementado
- [ ] Trading environment implementado
- [ ] Agent wrapper implementado
- [ ] Script de entrenamiento SAC

---

### Fase 5: Evaluation & Backtesting (Semana 11-12 del MVP)
```bash
touch src/atlasfx/evaluation/{metrics.py,backtester.py,visualizers.py}
touch scripts/backtest.py
```

**Deliverables:**
- [ ] Métricas implementadas
- [ ] Backtester implementado
- [ ] Visualizaciones implementadas
- [ ] Script de backtesting

---

## Comparación Final

| Aspecto | Actual | Propuesta Original | Recomendada |
|---------|--------|-------------------|-------------|
| **Modularidad** | ⭐⭐ Baja | ⭐⭐⭐⭐ Alta | ⭐⭐⭐⭐⭐ Muy Alta |
| **Escalabilidad** | ⭐⭐ Limitada | ⭐⭐⭐⭐ Buena | ⭐⭐⭐⭐⭐ Excelente |
| **Instalabilidad** | ❌ No | ✅ Sí | ✅ Sí |
| **Standards** | ⭐⭐ Parcial | ⭐⭐⭐ Bueno | ⭐⭐⭐⭐⭐ PEP compliant |
| **Claridad** | ⭐⭐⭐ OK | ⭐⭐⭐⭐ Buena | ⭐⭐⭐⭐⭐ Muy Clara |
| **Testing** | ⭐⭐⭐ Básico | ⭐⭐⭐ Básico | ⭐⭐⭐⭐⭐ Estructurado |
| **Documentación** | ⭐⭐⭐⭐ Muy Buena | ⭐⭐ Limitada | ⭐⭐⭐⭐⭐ Completa |
| **Complejidad Inicial** | ⭐⭐⭐⭐⭐ Simple | ⭐⭐⭐ Media | ⭐⭐⭐ Media |

---

## Recomendación Final

### ✅ ADOPTAR ESTRUCTURA RECOMENDADA

**Razones:**

1. **Profesionalismo:**
   - Sigue estándares de industria (PEP 517/518)
   - Similar a proyectos open-source exitosos

2. **Escalabilidad:**
   - Fácil agregar nuevos modelos, agentes, metrics
   - No requiere reestructurar al escalar

3. **Colaboración:**
   - Estructura clara facilita onboarding
   - Separación de responsabilidades

4. **Mantenibilidad:**
   - Código modular es más fácil de testear
   - Cambios localizados (no afectan todo)

5. **Deployment:**
   - Paquete instalable (`pip install atlasfx`)
   - Imports limpios
   - Fácil integración con CI/CD

### 🎯 Estrategia: Migración Incremental

**NO** hacer big-bang rewrite. **SÍ** migrar por fases:
- Fase 1-2: Setup estructura
- Fase 3-5: Migrar módulos uno a uno
- Mantener ambas estructuras temporalmente
- Deprecar old structure al finalizar MVP

**Timeline:** Se alinea perfectamente con el MVP Action Plan (12-18 semanas)

---

## Recursos y Referencias

### Best Practices
- [Python Packaging Guide](https://packaging.python.org/)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [ML Project Template](https://github.com/ashleve/lightning-hydra-template)

### Ejemplos de Proyectos Bien Estructurados
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn)

---

## Próximos Pasos

1. **Revisión:** Obtener approval de la estructura recomendada
2. **Setup:** Crear estructura básica (Fase 1)
3. **Migración:** Seguir plan incremental (Fases 2-5)
4. **Validación:** Asegurar que tests pasan después de cada fase
5. **Documentación:** Actualizar README con nueva estructura

---

**Autor:** AtlasFX Architecture Team  
**Fecha:** Octubre 2025  
**Versión:** 1.0  
**Status:** Propuesta para Aprobación
