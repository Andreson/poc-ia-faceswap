# FaceSwap AI — GPU Worker

Worker GPU do **FaceSwap AI**: troca o rosto de um vídeo pelo rosto de uma imagem de origem usando InsightFace + CodeFormer, e roda como **RunPod Serverless Endpoint**.

[![GPU Worker CI](https://github.com/Andreson/poc-ia-faceswap/actions/workflows/gpu-worker-ci.yml/badge.svg)](https://github.com/Andreson/poc-ia-faceswap/actions/workflows/gpu-worker-ci.yml)
[![Publish GPU Worker Image](https://github.com/Andreson/poc-ia-faceswap/actions/workflows/gpu-worker-publish.yml/badge.svg)](https://github.com/Andreson/poc-ia-faceswap/actions/workflows/gpu-worker-publish.yml)

## Sumário

- [O que faz](#o-que-faz)
- [Como funciona](#como-funciona)
- [Componentes](#componentes)
- [Principais bibliotecas](#principais-bibliotecas)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Modelos](#modelos)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Executando localmente](#executando-localmente)
- [Contrato do job (RunPod)](#contrato-do-job-runpod)
- [Build e deploy da imagem](#build-e-deploy-da-imagem)
- [CI/CD](#cicd)
- [Especificações](#especificações)

## O que faz

O usuário envia para a plataforma FaceSwap AI uma **imagem com um rosto** e um **vídeo alvo**. O backend (FastAPI) grava os dois arquivos no Cloudflare R2 e dispara um job neste worker. O worker:

1. baixa a imagem e o vídeo do R2;
2. extrai o rosto da imagem de origem;
3. em cada frame do vídeo, troca o maior rosto encontrado pelo rosto de origem e melhora a qualidade do rosto com CodeFormer;
4. remonta o vídeo mantendo o FPS e o áudio originais;
5. envia o resultado de volta ao R2 (`processed/{job_id}/output.mp4`) e informa o progresso ao RunPod durante o processamento.

Este repositório tem **só o worker GPU**. A API, o frontend e o banco de dados ficam no monorepo do FaceSwap AI.

## Como funciona

```
 Backend (FastAPI) ──POST /run──►  RunPod Serverless  ──►  handler.py
                                                            │
                     ┌──────────── download_from_r2 ◄───────┤  storage.py (boto3 → R2)
                     ▼                                      │
         VideoFaceSwapperPipeline (pipeline.py)             │
           ├─ FFmpeg: probe + extração de frames (máx. 1080p) e do áudio
           ├─ InsightFace buffalo_l: detecção + embedding do rosto
           ├─ inswapper_128.onnx: troca do rosto frame a frame
           ├─ CodeFormer (enhancer.py): restauração facial
           └─ FFmpeg: remontagem H.264 + AAC
                     │
                     └──────────── upload_to_r2 ────────────►  processed/{job_id}/output.mp4
```

Comportamentos relevantes do pipeline:

| Situação | Comportamento |
|---|---|
| Vários rostos na imagem ou no frame | Usa o de maior bounding box |
| Nenhum rosto no frame | O frame fica sem alteração |
| Nenhum rosto na imagem de origem | O job falha com uma mensagem de erro |
| Vídeo acima de 1080p | É reduzido para 1080p de altura antes de processar |
| Vídeo sem áudio | Gera a saída sem trilha de áudio |
| A cada 100 frames | Libera memória (`gc` + `torch.cuda.empty_cache`) e envia o progresso |
| Modelos ausentes | Falha na inicialização (*cold start*); os pesos nunca são baixados em tempo de execução |

## Componentes

| Arquivo | Função |
|---|---|
| [gpu_worker/handler.py](gpu_worker/handler.py) | Ponto de entrada do RunPod Serverless. Carrega o pipeline uma vez no cold start, processa o job e devolve a chave do vídeo gerado ou um erro. |
| [gpu_worker/pipeline.py](gpu_worker/pipeline.py) | `VideoFaceSwapperPipeline`: orquestra FFmpeg, InsightFace e CodeFormer. |
| [gpu_worker/enhancer.py](gpu_worker/enhancer.py) | `CodeFormerRestorer`: restauração facial com CodeFormer (arquitetura do `basicsr` + recorte/colagem do `facexlib`). |
| [gpu_worker/storage.py](gpu_worker/storage.py) | Download e upload no Cloudflare R2 via API compatível com S3. |
| [gpu_worker/cli_swap.py](gpu_worker/cli_swap.py) | CLI para rodar o pipeline com arquivos locais, sem RunPod nem R2. |
| [docker/gpu_worker.Dockerfile](docker/gpu_worker.Dockerfile) | Imagem multi-stage sobre `nvidia/cuda:12.1.1-runtime-ubuntu22.04`. |
| [docker/build_and_push.sh](docker/build_and_push.sh) | Build (`linux/amd64`) e push manual para o Docker Hub. |
| [.github/workflows/](.github/workflows/) | CI (lint e build) e publicação da imagem. |

## Principais bibliotecas

| Biblioteca | Uso |
|---|---|
| [InsightFace](https://github.com/deepinsight/insightface) | Detecção e análise facial (`buffalo_l`) e troca de rosto (`inswapper_128`) |
| [ONNX Runtime GPU](https://onnxruntime.ai/) | Inferência dos modelos ONNX do InsightFace em CUDA |
| [PyTorch](https://pytorch.org/) | Execução do CodeFormer |
| [BasicSR](https://github.com/XPixelGroup/BasicSR) | Arquitetura da rede CodeFormer (`ARCH_REGISTRY`) |
| [facexlib](https://github.com/xinntao/facexlib) | Alinhamento, recorte e colagem dos rostos restaurados |
| [OpenCV](https://opencv.org/) (headless) | Leitura e escrita dos frames |
| [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) + FFmpeg | Probe, extração de frames e áudio, remontagem do vídeo |
| [runpod](https://github.com/runpod/runpod-python) | SDK do RunPod Serverless (`serverless.start`, `progress_update`) |
| [boto3](https://boto3.amazonaws.com/) | Cliente S3 para o Cloudflare R2 |
| [ruff](https://docs.astral.sh/ruff/) | Lint (regras `E`, `F`, `I`; Python 3.10; linhas de até 120 caracteres) |

## Estrutura do repositório

```
.
├── .github/workflows/
│   ├── gpu-worker-ci.yml        # lint + validação do build Docker
│   └── gpu-worker-publish.yml   # publicação da imagem no Docker Hub
├── docker/
│   ├── gpu_worker.Dockerfile
│   └── build_and_push.sh
└── gpu_worker/
    ├── handler.py               # entrypoint RunPod
    ├── pipeline.py              # pipeline de face-swap
    ├── enhancer.py              # CodeFormer
    ├── storage.py               # Cloudflare R2
    ├── cli_swap.py              # execução local
    ├── test_input.json          # payload de exemplo
    ├── requirements.txt
    └── pyproject.toml           # configuração do ruff
```

## Pré-requisitos

- Python 3.10+
- FFmpeg instalado no sistema (`ffmpeg` no `PATH`)
- GPU NVIDIA com CUDA 12.x (recomendado); é possível rodar em CPU com `--device cpu`, mas é bem mais lento
- Docker com Buildx (para gerar a imagem)
- Pesos dos modelos baixados localmente (ver [Modelos](#modelos))

> **Apple Silicon / arm64:** o `onnxruntime-gpu` não tem wheels para `linux/aarch64`. Use a imagem Docker `linux/amd64` ou instale o `onnxruntime` para CPU no ambiente local.

## Modelos

O worker **não baixa modelos em tempo de execução**. Todos os pesos precisam estar no diretório apontado por `MODELS_DIR`. Em produção, esse diretório é o Network Volume do RunPod (`/runpod-volume/models`).

```
$MODELS_DIR/
├── models/buffalo_l/        # InsightFace FaceAnalysis (root = MODELS_DIR)
├── inswapper_128.onnx       # modelo de troca de rosto
├── codeformer.pth           # pesos do CodeFormer (chave "params_ema")
└── ...                      # pesos do facexlib (detecção RetinaFace ResNet50 e parsing)
```

## Variáveis de ambiente

| Variável | Obrigatória | Padrão | Descrição |
|---|---|---|---|
| `MODELS_DIR` | Sim | `/runpod-volume/models` (na imagem) | Diretório com os pesos dos modelos |
| `DEVICE` | Não | `cuda` | `cuda` ou `cpu` |
| `R2_ACCOUNT_ID` | Sim (handler) | — | ID da conta Cloudflare (monta o endpoint do R2) |
| `R2_ACCESS_KEY_ID` | Sim (handler) | — | Access key do R2 |
| `R2_SECRET_ACCESS_KEY` | Sim (handler) | — | Secret key do R2 |
| `R2_BUCKET_NAME` | Sim (handler) | — | Bucket de entrada e saída |

As variáveis `R2_*` só são usadas pelo `handler.py`. A CLI não precisa delas.

## Executando localmente

```bash
cd gpu_worker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MODELS_DIR=/caminho/para/models
```

**1. Pipeline com arquivos locais (sem RunPod nem R2):**

```bash
python cli_swap.py \
  --source-image rosto.jpg \
  --target-video video.mp4 \
  --output-video saida.mp4 \
  --device cuda
```

**2. Handler do RunPod com um payload de teste:**

```bash
export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET_NAME=...
python handler.py --test_input test_input.json
```

As chaves informadas em [test_input.json](gpu_worker/test_input.json) precisam existir no bucket R2.

**Lint:**

```bash
pip install ruff
ruff check gpu_worker
```

## Contrato do job (RunPod)

**Entrada**

```json
{
  "input": {
    "job_id": "test-job-001",
    "source_image_key": "sources/test-source.jpg",
    "target_video_key": "targets/test-target.mp4"
  }
}
```

**Saída (sucesso)**

```json
{ "output_video_key": "processed/test-job-001/output.mp4", "job_id": "test-job-001" }
```

**Saída (falha):** o worker não cai; ele devolve a mensagem de erro para o job ser marcado como falho.

```json
{ "error": "No face detected in source image.", "job_id": "test-job-001" }
```

**Progresso:** durante o processamento, o worker envia `{"progress": <0-100>}` via `runpod.serverless.progress_update`.

## Build e deploy da imagem

A imagem é sempre gerada para `linux/amd64`, porque os hosts GPU do RunPod são x86_64. O contexto de build é a raiz deste repositório.

```bash
docker login
./docker/build_and_push.sh 1.0.0      # publica andreson09thiago/faceswap:1.0.0
```

Para mudar o destino, use `DOCKERHUB_USER`, `IMAGE_NAME` e `PLATFORM`. Depois do push, aponte o **RunPod Serverless Endpoint** para a nova tag e monte o Network Volume com os modelos.

## CI/CD

| Workflow | Gatilho | O que faz |
|---|---|---|
| [gpu-worker-ci.yml](.github/workflows/gpu-worker-ci.yml) | Push em `main`/`master` e PRs que alteram `gpu_worker/**` ou o Dockerfile | `ruff check` e build da imagem sem push |
| [gpu-worker-publish.yml](.github/workflows/gpu-worker-publish.yml) | Tag `v*` ou execução manual | Build e push para `docker.io/andreson09thiago/faceswap:<versão>` e `:latest` |

Secrets necessários no repositório: `DOCKERHUB_USERNAME` e `DOCKERHUB_TOKEN`.

Para publicar uma versão:

```bash
git tag v1.0.0 && git push origin v1.0.0
```

## Especificações

As especificações normativas ficam em `.harness/`, no monorepo do FaceSwap AI. As mais relevantes para este worker são:

- `03-ai_pipeline_specification_harness.md`: requisitos `REQ-AI-001` a `REQ-AI-013` (pipeline, erros, progresso)
- `04-implementation_roadmap_harness.md`: requisitos `REQ-OPS-*` (CLI, imagem e deploy no RunPod)

Os comentários no código citam esses IDs. Consulte a especificação antes de mudar o contrato do job ou o pipeline.
