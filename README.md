# Fruitfly-reviewer

[한국어](README.md) | [English](README.en.md)

**MaleCNS 초파리 연결망을 사용해 학습한 실험적 논문 발췌 모델입니다.** 기존 대형 언어모델의 언어 가중치나 API를 사용하지 않습니다. PDF에서 주요 기여와 한계 문장을 골라 짧은 문단을 생성합니다.

> Experimental connectome-based extractive paper reviewer. Language components were trained in this project; no pretrained LLM backbone or LLM API. This is a limited sentence-extraction result, not a general scientific peer reviewer.

## 실제 결과와 범위

39쪽 논문을 PDF부터 다시 읽은 최종 모델의 **수정하지 않은 출력**:

> We present the connectome of the entire Drosophila male central nervous system. Limitations due to lack of hemilineage assignment, ambiguity in lineage/cell type delineation, and technical limitations imposed by the genetic labeling prevented annotation for certain groups of cells.

[원출력](outputs/review-runs/final-v9-fallback/review.txt) · [평가 결과](outputs/sections-v9/evaluation.json) · [학습 기록](outputs/sections-v9/training.jsonl)

대상은 Berg et al., *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, [DOI: 10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015)입니다. 첫 문장은 전체 수컷 CNS 연결망 구축이라는 기여를, 둘째 문장은 실제 연구 한계를 발췌합니다. 둘째 문장의 원래 문맥은 특히 **fruitless/doublesex 발현 주석**입니다. 모든 세포 유형 주석이 불가능했다는 뜻은 아닙니다.

| 검증 | 결과 |
|---|---|
| 독립 시험 논문 10편, 발췌 예제 32개 | 두 문장 완전 일치 15/32 (46.875%) |
| 정상 종료 | 32/32 |
| 연결망 신호 제거 | 일치 0, 빈 출력 |
| 목표 PDF 학습 | 사용하지 않음; 학습 자료는 2022년까지 |
| 숫자 변경 대조 | 변경된 네 숫자 출력에 실패; 다른 기여 문장을 선택 |

**한계:** 새로운 비평을 쓰는 모델이 아닙니다. 작은 시험 집합, 저자 시점의 발췌, PDF 추출기별 출력 변화, 그림 미해석 문제가 있습니다. 연결 제거 검사는 실제 배선이 무작위 배선보다 우월하다는 증거가 아닙니다. 목표 PDF 결과만으로 범용 리뷰 능력을 주장하지 않습니다.

## 구조

- MaleCNS 중앙 뇌 **49,393개 뉴런 / 9,050,172개 연결**을 사용하는 희소 재귀 연결망입니다. 전체 166,700개 뉴런의 생물학적 시뮬레이션은 아닙니다.
- 원본 아티팩트에서 연결 구조, 연결 가중치, 입력 뉴런 인덱스만 읽습니다. 원본의 언어 가중치는 로드하지 않습니다.
- 자체 학습한 임베딩·입력 변환·출력 경로에 원문 복사 메모리, 사용 위치 기록, 초록/본문 표시를 결합했습니다. 이 외부 메모리와 학습된 장치도 결과에 기여합니다.
- 기존 공개 모델의 동화 출력과 소규모 출력층 과적합 실패 이후, 과학 문헌과 문맥 처리 구조를 바꾸어 진행했습니다. 다른 AI가 작성한 목표 리뷰를 결과로 대신하지 않았습니다.

## 실행 — Windows 11 / RTX 5090

검증 환경: Python 3.12.14, NVIDIA 드라이버 616.92, PyTorch 2.7.1+cu128, CUDA 12.8, RTX 5090 32GB. 다른 환경의 동작은 검증하지 않았습니다. 저장소 루트에서 실행합니다.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe check_gpu.py
.venv\Scripts\python.exe download_graph.py
.venv\Scripts\python.exe work/generate_review.py --pdf "C:\path\to\paper.pdf"
```

최종 체크포인트(약 61 MB), 자체 토크나이저와 단어 빈도표는 포함되어 있습니다. 그래프 원본(약 284 MB)은 고정 리비전에서 별도 다운로드하고 SHA-256을 검사합니다. PDF는 직접 준비해야 합니다.

새 결과는 `outputs/review-runs/<시각>/`에 저장됩니다. `review.txt`가 수정하지 않은 모델 출력입니다. 입력 페이지, 정규화 기록, 생성 토큰과 실행 메타데이터도 함께 저장됩니다. **이 폴더에는 입력 논문 전체 텍스트와 로컬 경로가 들어가므로 기본적으로 Git에서 제외됩니다.** pypdf가 해당 PDF를 읽지 못하면 PyMuPDF로 재시도합니다.

## 학습 기록과 재현 범위

eLife 공개 XML에서 학습 자료를 구성했습니다. 논문별 분할을 유지하며 2022년 이후 자료를 제외했습니다. 자체 언어 학습 → 과학 문헌 → 질의·복사 메모리 → 문장 전환 → 사용 위치 기록 → 초록/본문 표시 순으로 모델을 발전시켰습니다. 최종 v9는 학습 251편, 검증 12편, 시험 11편으로 구성된 정제 자료에서 500단계 학습했고, 검증 목적함수로 250단계 체크포인트를 선택했습니다.

이 배포본은 **최종 모델 추론을 재현**하도록 구성했습니다. `work/`의 역사적 학습·평가 코드는 함께 보존하지만, 이전 단계의 체크포인트와 대용량 코퍼스는 포함하지 않습니다. 따라서 학습 코드를 한 번 실행하면 처음부터 최종 모델이 재현되는 패키지는 아닙니다. [실험 경과와 재현 제약](docs/EXPERIMENTS.md)을 확인하세요.

체크포인트 저장은 모델·최적화기·난수 상태를 두 슬롯에 기록하며 손상 시 이전 슬롯으로 복구합니다. 학습 중에는 `work/control_training.py pause --run sections-v9`로 저장 중단을 요청하고 `status`로 프로세스 종료까지 확인합니다. 최종 배포 가중치는 추론용이며 최적화기 상태가 없습니다. 원래 학습 도구의 4시간 제한과 한국 시간 09:25–19:30 중단 조건도 보존되어 있습니다.

## 출처와 권리

[MaleCNS](https://github.com/natverse/malecns), [원본 데모](https://huggingface.co/spaces/VIDraft/fruitfly-brain), [그래프 아티팩트](https://huggingface.co/ngxson/fly-llm-hf), [eLife XML](https://github.com/elifesciences/elife-article-xml)를 참고했습니다. 자세한 출처와 배포 범위는 [NOTICE](NOTICE.md)에 있습니다. 프로젝트의 별도 재사용 라이선스는 아직 지정하지 않았습니다.

논문 PDF, 학습 XML 원본, 전체 입력 텍스트, 가상 환경, 개인 인수인계 문서는 포함하지 않습니다. 실패한 실험과 원본 실행 기록은 로컬 연구 작업공간에 보존되어 있습니다.

## GPT-6 Astra가 수행한 작업 범위

사용자가 목표·제약·실행 환경을 제공하고 진행 방향 및 공개를 결정했으며, GPT-6 Astra는 코딩 에이전트로서 기존 기록과 외부 자료 조사, 실험 설계와 코드 작성·수정, GPU 환경 점검과 실행 검증, 학습 자료 처리, 모델 학습·평가 실행, 중간 저장·재개 구현 및 검사, 출력과 논문 내용의 대조, 실패·한계 분석, 문서 작성·번역, 민감정보 검사와 GitHub 게시 작업을 수행했습니다. 원본 연결망 데이터와 논문은 각 원저자의 작업입니다.

GPT-6 Astra는 프로젝트 개발과 검증을 지원한 에이전트이며 배포 모델의 언어 백본이나 추론 API가 아닙니다. 위에 인용한 리뷰는 별도로 학습한 연결망 기반 모델의 수정하지 않은 실제 출력입니다. Astra가 쓴 리뷰로 대체하거나 목표 논문의 정답 리뷰를 외우게 한 결과가 아닙니다. 이 문서의 결과 해석은 독립적인 외부 검증을 의미하지 않습니다.
