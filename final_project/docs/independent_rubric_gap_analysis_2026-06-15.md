# Independent Rubric and Gap Analysis

Audit date: June 15, 2026. Status: post-remediation.

The official project is a dual-task benchmark with 36 official runs. Sentiment retains 446,745 rows and emotion retains 446,640 rows after deterministic cleaning, connected-group conflict removal, and normalized-text deduplication. Linear SVM is selected by validation macro-F1 for both tasks, with test macro-F1 0.4590 for sentiment and 0.2854 for emotion. Labels are weak references; no human-gold evaluation is claimed.

The earlier audit identified duplicate cross-split overlap, mixed experiment snapshots, incomplete task packaging, incomplete repeated seeds, test-based ranking risk, missing model weights, and stale documentation. These issues are closed in the official task-isolated 36-run benchmark. The remaining gap is the absence of native-speaker human-gold labels. Rubric evidence coverage reaches all 50 available marks, but no grade is guaranteed.
