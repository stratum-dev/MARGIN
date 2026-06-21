![](./assets/logo.png)

[English](../README.MD) | [中文](./README.zh.MD) | [Français](./README.fr.MD) | [Español](./README.es.MD) | [Русский](./README.ru.MD) | [العربية](./README.ar.MD)

**M**argin-**A**ware **R**egularized **G**eometry for **I**mbalance Vulnerability Detectio**N**

Implementación oficial del artículo. Este repositorio proporciona los pipelines de entrenamiento y evaluación para su reproducción.

## Resumen

![](./assets/banner.png)

La detección de vulnerabilidades de software es fundamental para garantizar la seguridad y fiabilidad del software. A pesar de los avances recientes en aprendizaje profundo, los conjuntos de datos de vulnerabilidades del mundo real sufren dos graves desafíos: desequilibrio de frecuencia y desequilibrio de dificultad. Reinterpretamos estos desafíos desde una perspectiva de geometría de representación, observando que dichos desequilibrios inducen distorsiones geométricas en el espacio de representación hiperesférico. Para abordar este problema, proponemos MARGIN, un marco basado en aprendizaje métrico que aprende representaciones discriminativas de vulnerabilidades mediante aprendizaje métrico con margen adaptativo y modelado de prototipos hiperesféricos. MARGIN ajusta dinámicamente la regularización geométrica según la estructura de distribución estimada por la concentración de von Mises-Fisher, alineando la masa de probabilidad de las distribuciones de representación con sus correspondientes celdas de Voronoi, reduciendo así la distorsión geométrica y produciendo fronteras de decisión más estables. Experimentos exhaustivos en conjuntos de datos públicos de vulnerabilidades muestran que MARGIN supera consistentemente a los modelos de referencia sólidos, logrando mejoras notables en clasificación y detección, especialmente en conjuntos de datos desafiantes y desequilibrados. Análisis adicionales demuestran que MARGIN produce geometrías de representación más estructuradas, mejorando la robustez, interpretabilidad y generalización.

## Requisitos

- **Memoria GPU**: ≥ 24 GB
- **CUDA**: runtime 12.4 (controlador ≥ 525 también puede funcionar por compatibilidad hacia atrás)
- **CPU**: x86-64, ≥ 4 núcleos
- **RAM**: ≥ 8 GB
- **Espacio en disco**: ≥ 2 GB para el conjunto de datos, ≥ 5 GB para los modelos backbone
- **Sistema operativo**: Linux (probado en Ubuntu 22.04 / 24.04)

## Instalación

### 1. Crear el entorno conda

```bash
conda env create -f environment.yml
conda activate margin
```

### 2. Instalar PyTorch

(Ruedas CUDA 12.4 — alojadas en pytorch.org, no en PyPI)

```bash
pip install torch==2.6.0+cu124          \
            torchvision==0.21.0+cu124   \
            torchaudio==2.6.0+cu124     \
            --index-url https://download.pytorch.org/whl/cu124
```

### 3. Instalar el CLI de HuggingFace

```bash
pip install -U "huggingface_hub"
```

## Descarga

El conjunto de datos y los modelos backbone están alojados en HuggingFace. Puede descargarlos manualmente mediante las herramientas CLI.

### Conjunto de datos

```bash
hf download codemetic/MARGIN --repo-type=dataset
```

### Modelos Backbone

```bash
hf download Salesforce/codet5-base
```

> **Para usuarios en China continental:** configure el endpoint espejo de HuggingFace antes de cualquier comando `hf download`:
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

## Entrenamiento

Ejecute `python main.py --help` para ver todos los parámetros disponibles. Los hiperparámetros relevantes se proporcionan en el artículo.

Ejemplo de comando de entrenamiento:

```bash
python  main.py                                           \
        --dataset_subset          bigvul                  \
        --backbone_name           Salesforce/codet5-base  \
        --max_epochs              200                     \
        --max_checkpoints         3                       \
        --early_stopping_patience 30                      \
        --base_scale              20                      \
        --device                  cuda:0                  \
        --seed                    42                      \
        --batch_size              16
```

**NOTA:** Los resultados de entrenamiento y evaluación pueden diferir ligeramente del artículo original debido a variaciones en la arquitectura de la plataforma, hardware y versiones de software.

## Evaluación

```bash
python eval.py
```

CLI interactiva — use `↑` / `↓` para seleccionar una ejecución de entrenamiento y un punto de control, luego configure el tamaño de lote, la semilla aleatoria y el dispositivo antes de que comience la evaluación.

## Cita

```bibtex
@misc{zhang2026MARGIN,
      title={MARGIN: Margin-Aware Regularized Geometry for Imbalanced Vulnerability Detection}, 
      author={Yuteng Zhang and Huifang Ma and Jiahui Wei and Qingqing Li and Yafei Yang},
      year={2026},
      eprint={2605.10240},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2605.10240}, 
}
```

## Licencia

El código fuente y la implementación de programación en este repositorio están bajo la licencia **CC BY 4.0**.

Los conjuntos de datos referenciados o utilizados por este proyecto **NO** están incluidos bajo esta licencia. Todos estos conjuntos de datos siguen siendo propiedad intelectual de sus respectivos propietarios y se distribuyen bajo sus propias licencias, términos de uso y condiciones de acceso. Los usuarios son responsables de cumplir con los requisitos de licencia aplicables de los proveedores de conjuntos de datos originales al descargar, usar o redistribuir estos conjuntos de datos.
