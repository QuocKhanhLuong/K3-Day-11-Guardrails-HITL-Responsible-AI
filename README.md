# Day 11 — Controlled Agent Security

**Sinh viên:** Lương Quốc Khánh  
**MSSV:** 2A202601713  
**Bài làm:** cá nhân

## Mục tiêu

Xây dựng pipeline an toàn cho VinBank theo luồng:

```text
User / Email / RAG
    → Rate limiter
    → Direct + indirect input guardrails
    → OpenAI Responses API
    → Output redaction
    → OpenAI multi-criteria LLM-as-Judge
    → Action permission + HITL
    → Deterministic egress policy
    → Audit log + monitoring
```

Email, RAG, web và tool output luôn được coi là **data**, không phải nguồn có quyền thay đổi policy hay tự phê duyệt hành động.

## Cài đặt với conda `lesson11`

```bash
conda activate lesson11
python -m pip install -U pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Điền `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_JUDGE_MODEL=gpt-4.1-mini
```

Không commit `.env` lên GitHub. Có thể đổi model nếu model đó có trong project OpenAI của bạn.

## Chạy bài

Từ thư mục gốc:

```bash
cd src
python main.py --part 2 --student-id 2A202601713
python main.py --part 4 --student-id 2A202601713
python main.py --part 5 --student-id 2A202601713
python main.py --part 1 --student-id 2A202601713
```

Trong lúc debug deterministic layers có thể dùng:

```bash
python main.py --part 5 --student-id 2A202601713 --no-llm-judge
```

Không dùng kết quả `--no-llm-judge` làm evidence cuối nếu báo cáo khẳng định đã chạy LLM-as-Judge.

## Kiểm thử

Từ thư mục gốc:

```bash
python -m compileall -q src
python -m pytest tests/smoke -q
python -m pytest tests/public -q
python scripts/grade.py --submission-dir . --out outputs/grade_report.json
```

## Thành phần chính

- `src/core/openai_runtime.py`: OpenAI Responses API adapter cho unsafe, protected và Guards agents.
- `src/guardrails/input_guardrails.py`: NFKC, loại zero-width, direct/indirect injection, topic và edge validation.
- `src/guardrails/output_guardrails.py`: redact PII/secret và OpenAI judge 4 tiêu chí.
- `src/assignment/rate_limiter.py`: sliding window theo từng user.
- `src/assignment/pipeline.py`: orchestration, provenance, action authorization và `is_egress_allowed()`.
- `src/hitl/hitl.py`: confidence router, approve/reject/timeout, reviewer context và correlation ID.
- `src/assignment/audit_log.py`: request-correlated audit JSON.
- `src/assignment/monitoring.py`: block-rate, rate-limit, judge-fail alerts và snapshot replay.
- `src/attacks/attacks.py`: direct, indirect, obfuscation, authority, action và egress attacks chạy trên target thật.

`google-adk` vẫn có trong dependency để giữ tương thích với starter và public tests, nhưng live runtime không cần `GOOGLE_API_KEY`.

## Evidence phải có trước khi nộp

```text
outputs/results.json
outputs/audit_log.json
outputs/metrics.json
outputs/attack_results.json
report/2A202601713_report.md
```

Các file output phải được sinh từ lần chạy thật với OpenAI API key. Lỗi API/runtime được ghi là `error`, không tính là guardrail chặn thành công.
