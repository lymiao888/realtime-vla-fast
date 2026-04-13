#!/usr/bin/env bash
set -euo pipefail

REPORT_BASENAME="pi05_modules_report"

echo "[1/4] Cleaning old reports..."
rm -f "${REPORT_BASENAME}.nsys-rep" "${REPORT_BASENAME}.sqlite" "${REPORT_BASENAME}.log" "${REPORT_BASENAME}_sum.log"

echo "[2/4] Profiling with nsys..."
nsys profile -t cuda,nvtx,osrt -o "${REPORT_BASENAME}" \
  python benchmark.py --model_version pi05 --num_views 3 --chunk_size 50

echo "[3/4] Exporting nsys stats..."
nsys stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum --format table --output "${REPORT_BASENAME}" \
  "${REPORT_BASENAME}.nsys-rep" > "${REPORT_BASENAME}.log"

echo "[4/4] Aggregating module-level NVTX summary..."
{
  echo "Pi05 modules report:"
  echo "--------------------------------"
  echo "Module|Instances|Total (ms)|Avg (ms)"
  echo "--------------------------------"
  sqlite3 "${REPORT_BASENAME}.sqlite" "
WITH nvtx AS (
  SELECT
    CASE
      WHEN text GLOB '*pi05.decoder.step*.layer*.attn*' THEN 'pi05.decoder.attn'
      WHEN text GLOB '*pi05.decoder.step*.layer*.ffn*' THEN 'pi05.decoder.ffn'
      WHEN text GLOB '*pi05.decoder.step*.input_proj*' THEN 'pi05.decoder.input_proj'
      WHEN text GLOB '*pi05.decoder.step*.output_proj*' THEN 'pi05.decoder.output_proj'
      WHEN text GLOB '*pi05.encoder.layer*.attn*' THEN 'pi05.encoder.attn'
      WHEN text GLOB '*pi05.encoder.layer*.ffn*' THEN 'pi05.encoder.ffn'
      WHEN text GLOB '*pi05.encoder.input_proj*' THEN 'pi05.encoder.input_proj'
      WHEN text GLOB '*pi0.vision.layer*.attn*' THEN 'pi0.vision.attn'
      WHEN text GLOB '*pi0.vision.layer*.ffn*' THEN 'pi0.vision.ffn'
      WHEN text GLOB '*pi0.vision.embed*' THEN 'pi0.vision.embed'
      ELSE NULL
    END AS module,
    (end - start) AS dur_ns
  FROM NVTX_EVENTS
  WHERE eventType = 59
)
SELECT
  module,
  COUNT(*) AS instances,
  ROUND(SUM(dur_ns) / 1e6, 3) AS total_ms,
  ROUND(AVG(dur_ns) / 1e6, 3) AS avg_ms
FROM nvtx
WHERE module IS NOT NULL
GROUP BY module
ORDER BY total_ms DESC;
"
} | tee "${REPORT_BASENAME}_sum.log"

echo "Done. See ${REPORT_BASENAME}.log and ${REPORT_BASENAME}_sum.log"