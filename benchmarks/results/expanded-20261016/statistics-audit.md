# Statistical audit

Status: **PASSED**. 7,410 checks; 0 findings.

Independently recomputed 276 scalar summaries, 69 matched-round comparisons, and all 276 CSV rows from the four raw experiment files. The audit did not import or call the summary implementation.

- Verified session/batch medians, pooled nearest-rank p95/min/max, independent-unit and sample counts, failure/status counts, payload resource labels, and complete profile/metric coverage.
- Verified 90 complete concurrency batches, positive execution windows, worker/call counts, and validated calls divided by shared execution time.
- Reproduced five representative scalar 95% intervals and the native-backend paired interval with 10,000 whole-session/whole-batch or matched-round bootstrap resamples, seed 20261016. All endpoints match within floating-point tolerance.
- Checked every concurrency interval is labeled exploratory (five batches), and paired ratio direction and inclusion of one agree with the saved interval.
- Report generator preserves pointwise, conditional-on-host, correlated-call, descriptive-p95 and payload-only-resource qualifications.

| Profile / backend / metric | Independent units | Estimate | Reproduced 95% CI | Units |
| --- | ---: | ---: | ---: | --- |
| baseline / bubblewrap / warm_run | 30 | 0.0216525042 | [0.0215481528, 0.0218302543] | seconds |
| baseline / docker / warm_run | 30 | 0.114734447 | [0.113817918, 0.116460037] | seconds |
| workload / nsjail / json_run | 10 | 0.069281439 | [0.0685049975, 0.0715530793] | seconds |
| concurrency / docker / throughput / 16 workers / json | 5 | 21.4719438 | [21.2927986, 21.8612602] | calls/second |
| concurrency / bubblewrap / json_run / 4 workers / json | 5 | 0.0851239945 | [0.083529301, 0.0924494935] | seconds |
| baseline / bubblewrap/nsjail / warm_run | 30 | 1.02403968 | [1.0145593, 1.046422] | ratio |

All measured records passed: 150 sequential sessions and 630 concurrent worker sessions across 90 batches, with 6,300 validated concurrent calls. This audit verifies retained numeric evidence; it does not rerun runtime payloads or establish cross-host generalization.

Detailed counts, reproduced intervals and audited input hashes: [statistics-audit.json](statistics-audit.json). Raw structural/correctness evidence: [raw-audit.json](raw-audit.json).
