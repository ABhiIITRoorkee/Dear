
# DEAR: Distillation-Calibrated Evidence Routing for Mashup–Web API Recommendation

Official implementation of **DEAR: Distillation-Calibrated Evidence Routing with Asymmetric Query–Passage Encoding for Mashup–Web API Recommendation**.

**Authors:** Abhinav Jamwal and Sandeep Kumar  
**Affiliation:** Department of Computer Science and Engineering, Indian Institute of Technology Roorkee, India

## Overview

DEAR is a framework for **cold-start mashup–Web API recommendation**. Given the textual requirements of a previously unseen mashup, DEAR recommends relevant Web APIs by leveraging semantic information and historical service-invocation relationships.

The framework combines:

- **Asymmetric Query–Passage Encoding:** Task-conditioned text representations for mashups, APIs, and tags using a shared pretrained Jina-v3 encoder.
- **Multi-View Hypergraph Learning:** Four entity-specific hypergraph views capturing historical mashup–API interactions and tag associations.
- **Distillation-Calibrated Evidence Routing:** A training-time teacher uses historical API-usage agreement to guide a student router toward informative historical mashups.
- **Cold-Start Inference:** The trained student retrieves and aggregates historical structural evidence without requiring invocation histories for unseen mashups.

DEAR separates offline repository representation learning from query-time recommendation.

## Repository Structure

```text
Dear/
├── dataset/
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   ├── fold_5/
│   ├── fold_6/
│   ├── fold_7/
│   ├── fold_8/
│   ├── fold_9/
│   └── fold_10/
├── utility/
├── Dockerfile
├── Models.py
├── README.md
├── generate_dataset.py
├── main.py
├── metric.py
├── metrics_single.py
├── regenerate_vectors.py
├── requirements.txt
└── run.py
```

### Main Components

| File | Description |
|---|---|
| `main.py` | Main training and evaluation implementation |
| `Models.py` | Model architectures and representation-learning components |
| `run.py` | Experiment execution script |
| `generate_dataset.py` | Dataset preparation utility |
| `regenerate_vectors.py` | Text-representation preparation utility |
| `metric.py` | Evaluation metrics |
| `metrics_single.py` | Additional evaluation utilities |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container environment configuration |
| `dataset/` | Ten fold-specific experimental datasets |
| `utility/` | Supporting implementation utilities |

## Dataset

Experiments use the **ProgrammableWeb mashup–API dataset**, which contains mashup descriptions, Web API metadata, and historical API-invocation relationships.

### Dataset Statistics

| Statistic | Value |
|---|---:|
| Original mashups | 7,947 |
| Original Web APIs | 1,693 |
| Original invocations | 16,200 |
| Processed mashups | 7,163 |
| Processed Web APIs | 1,382 |
| Processed invocations | 13,778 |
| Average APIs per mashup | 1.92 |
| Average mashups per API | 9.97 |
| Interaction density | 0.139% |

### Data Partitioning

The experimental protocol reserves 10% of processed mashups for validation and partitions the remaining 90% into ten disjoint evaluation folds.

For each evaluation round:

1. Nine folds form the historical training repository.
2. One fold is used for cold-start testing.
3. Training-dependent hypergraph structures and teacher supervision are constructed from the historical training repository.
4. Test mashups are evaluated using textual requirements without their invocation histories.
5. Results are aggregated across all ten evaluation folds.

The repository provides fold-specific experimental data in `dataset/fold_1/` through `dataset/fold_10/`.

The provided folds should be retained when reproducing the reported evaluation protocol.

## Installation

Two installation options are available:

1. Docker-based environment
2. Local Python environment

### Option 1: Docker Installation

Docker provides a consistent software environment for executing the experiments.

#### Prerequisites

- Git
- Docker
- An NVIDIA GPU with compatible drivers for CUDA execution
- NVIDIA Container Toolkit for GPU-enabled Docker containers

#### Step 1: Clone the Repository

```bash
git clone https://github.com/ABhiIITRoorkee/Dear.git
cd Dear
```

#### Step 2: Build the Docker Image

Execute the following command from the repository root:

```bash
docker build -t dear:latest .
```

Docker builds the environment according to the provided `Dockerfile`.

#### Step 3: Run the Experiment

For a GPU-enabled Docker environment, the experiment launcher can be invoked with:

```bash
docker run --rm --gpus all \
  --mount type=bind,source="$(pwd)",target=/workspace \
  --workdir /workspace \
  --entrypoint python \
  dear:latest run.py
```

This command mounts the repository into the container and executes `run.py` from the project directory.

Ensure that the entry script is configured for the intended dataset paths and experimental settings.

#### Step 4: Preserve the Experimental Logs

To save terminal output:

```bash
docker run --rm --gpus all \
  --mount type=bind,source="$(pwd)",target=/workspace \
  --workdir /workspace \
  --entrypoint python \
  dear:latest run.py 2>&1 | tee dear_experiment.log
```

The log records the output generated during execution.

### Option 2: Local Python Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/ABhiIITRoorkee/Dear.git
cd Dear
```

#### Step 2: Create a Virtual Environment

```bash
python3 -m venv dear_env
```

Activate the environment on Linux or macOS:

```bash
source dear_env/bin/activate
```

On Windows:

```powershell
dear_env\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a compatible PyTorch and CUDA configuration for GPU execution.

#### Step 4: Check GPU Availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

A result of `True` indicates that PyTorch can access CUDA.

#### Step 5: Run the Experiment

```bash
python run.py
```

The execution uses the configuration defined by the repository scripts.

## Experimental Workflow

The DEAR experimental workflow consists of the following stages.

### 1. Text Representation Preparation

Mashup, API, and tag metadata are represented using a shared pretrained Jina-v3 encoder with task-specific encoding roles.

- Mashup descriptions use `retrieval.query`.
- API descriptions and tags use `retrieval.passage`.
- The generated representations are used for semantic retrieval and initialization of trainable repository embeddings.

The `regenerate_vectors.py` utility supports text-vector preparation.

When using the supplied preprocessed data and embeddings, regeneration is unnecessary unless the experiment specifically requires rebuilding them.

### 2. Hypergraph Construction

Four entity-specific hypergraph views are constructed:

- Mashup interaction view
- API interaction view
- Mashup-tag view
- API-tag view

The views represent historical invocation and tag-association relationships.

Sparse hypergraph propagation and adaptive depth aggregation are used to learn repository representations.

### 3. Evidence Routing

For a mashup requirement, DEAR retrieves semantically related historical mashups.

During training, a teacher combines semantic similarity and historical API-usage agreement to provide evidence-routing supervision.

A student router learns query-specific preferences over historical evidence.

During inference, the student selects and aggregates historical mashup representations to rank candidate APIs.

### 4. Model Training

The framework combines structural representation learning, auxiliary tag supervision, evidence routing, and knowledge distillation.

The training procedure uses historical mashup–API interactions and sampled negative examples.

The training configuration controls hyperparameters, sampling, routing, and optimization settings.

The primary implementation files are:

```text
main.py
Models.py
run.py
```

### 5. Cold-Start Evaluation

Each held-out test mashup is treated as an unseen textual requirement.

The model ranks candidate APIs using the historical repository and learned parameters.

Test-mashup invocation records are used for evaluation, not as inference inputs.

## Evaluation Metrics

DEAR is evaluated using the following Top-N recommendation metrics:

| Metric | Description |
|---|---|
| MP@N | Mean Precision |
| MR@N | Mean Recall |
| MAP@N | Mean Average Precision |
| NDCG@N | Normalized Discounted Cumulative Gain |
| COV@N | API Catalogue Coverage |

The evaluated recommendation cutoffs are:

```text
N = {3, 5, 10, 20}
```

Metrics are computed over test mashups and aggregated across the ten evaluation folds.

## Experimental Results

The manuscript reports comparisons against nine representative recommendation methods:

- MTFM
- PAREI
- SRCLML
- BRW
- GSL-Mash
- GSR
- DeepSeek-V3
- R2API
- CVH-REC

### Ranking Performance

The following values summarize the reported comparison with CVH-REC.

| Metric | CVH-REC | DEAR |
|---|---:|---:|
| MAP@3 | 0.7392 | **0.8114** |
| MAP@5 | 0.7366 | **0.8096** |
| MAP@10 | 0.7238 | **0.7971** |
| MAP@20 | 0.7034 | **0.7753** |
| NDCG@3 | 0.6596 | **0.7140** |
| NDCG@5 | 0.6744 | **0.7254** |
| NDCG@10 | 0.6958 | **0.7459** |
| NDCG@20 | 0.7139 | **0.7635** |

Across the evaluated cutoffs, DEAR reports average relative improvements of:

- **10.01% in MAP**
- **7.49% in NDCG**

These values are taken from the manuscript's experimental evaluation.

## Ablation Studies

The manuscript examines the contributions of individual components through the following variants.

### Representation-Learning Ablations

| Variant | Modification |
|---|---|
| DEAR w/o TextInit | Removes pretrained text initialization |
| DEAR w/o HGBackbone | Removes hypergraph propagation and depth aggregation |
| DEAR w/o TagLoss | Removes auxiliary tag-ranking objectives |
| DEAR UniformDepth | Uses uniform propagation-depth weights |

### Evidence-Routing Ablations

| Variant | Modification |
|---|---|
| DEAR w/o KD | Removes knowledge-distillation supervision |
| DEAR SemanticRouter | Uses semantic-similarity aggregation weights |
| DEAR UniformRouter | Uses uniform evidence-aggregation weights |
| DEAR Route-Off-at-Test | Disables the routed inference pathway |

The experimental analysis also examines training efficiency and recommendation under zero companion co-occurrence support for previously observed APIs.

## Reproducibility

The repository provides the following resources to facilitate experimental reproduction:

- Source code for DEAR
- Ten fold-specific datasets
- Python dependency specification
- Dockerfile for the experimental software environment
- Dataset and text-vector preparation utilities
- Training and evaluation scripts

### Recommended Reproduction Procedure

1. Clone the repository.
2. Build the Docker image or install the Python dependencies.
3. Use the supplied ten-fold datasets.
4. Configure the experiment using the repository scripts.
5. Execute the training and evaluation procedure.
6. Aggregate evaluation results across the ten folds.
7. Compare the resulting metrics with the values reported in the manuscript.

For consistent comparisons, retain the provided data partitions, evaluation cutoffs, and experimental configurations.

### Environment Information

When reporting reproduced experiments, recording the following information is recommended:

- Repository commit
- Docker image or software dependency versions
- GPU model and CUDA environment
- Dataset fold
- Random seed
- Model and optimization configuration
- Evaluation metrics

Docker standardizes the containerized software environment and reduces differences arising from dependency installation across systems.

## Troubleshooting

### Docker Build Issues

Ensure that Docker is running and execute the build command from the repository root containing `Dockerfile`.

### CUDA Not Available

Verify that the NVIDIA drivers, CUDA-compatible PyTorch installation, and NVIDIA Container Toolkit are correctly configured.

### Dataset Path Issues

Confirm that the supplied dataset directories are present:

```bash
ls dataset/
```

The folder should contain `fold_1` through `fold_10`.

### Missing Dependencies

For local Python execution, reinstall the specified dependencies:

```bash
python -m pip install -r requirements.txt
```

### Experiment Configuration

Review the configuration and file paths used by `run.py` and `main.py` when selecting datasets or executing experiments.

## Citation

If you use DEAR or its experimental resources in your research, please cite:

**Abhinav Jamwal and Sandeep Kumar**,  
*DEAR: Distillation-Calibrated Evidence Routing with Asymmetric Query–Passage Encoding for Mashup-Web API Recommendation.*

The final publication citation can be added when bibliographic details become available.

## Contact

For research questions, implementation-related inquiries, or reproducibility discussions, please use the repository's GitHub Issues section.

**Repository:** https://github.com/ABhiIITRoorkee/Dear

**Institution:** Indian Institute of Technology Roorkee, India
