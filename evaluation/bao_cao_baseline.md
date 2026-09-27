# Bao cao baseline RAG

Ngay chay: 2026-09-25

## Retrieval baseline

File: `evaluation/baseline_retrieval.json`

Bo cau hoi: `tests/fixtures/golden_tailieu_v2_100.json`

- Tong case: 100
- Retrieval case: 92
- Passed: 87
- Failed: 13
- Hit@1: 0.6739
- Hit@3: 0.8478
- Hit@5: 0.8587
- MRR: 0.7612
- Source page accuracy: 0.7935
- Article accuracy: 0.9783
- Clause accuracy: 0.9756
- Source type accuracy: 0.9348

Nhan xet: retrieval da sat nguong nghiem thu ban dau, nhung full 100 case con thieu nhe so voi muc Hit@3 >= 0.85 va source page accuracy >= 0.80. Smoke 20 case dau dat Hit@3 = 0.95 va source page accuracy = 0.85.

## Generation baseline

File: `evaluation/baseline_generation.json`

Bo cau hoi: `tests/fixtures/golden_tailieu_v1.json`

- Tong case: 24
- Passed: 5
- Failed: 19
- Answer correctness: 0.3000
- Citation correctness: 0.6000
- Refusal accuracy: 0.0000
- Hallucination proxy: 0.0000
- Answer reference recall: 0.6320

Nhan xet: generation/source verifier con yeu hon retrieval. Can uu tien citation verifier, refusal logic va faithfulness check truoc khi nang LLM.
