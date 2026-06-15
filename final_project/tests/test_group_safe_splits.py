from __future__ import annotations

import pandas as pd

from src.create_splits import build_group_safe_frame, split_group_safe


def test_group_safe_frame_drops_conflicts_and_deduplicates_text() -> None:
    frame = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "6"],
            "clean_text": ["same", "same", "conflict", "conflict", "a", "b"],
            "task_label": ["Joy", "Joy", "Joy", "Sad", "Sad", "Angry"],
            "raw_text": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "raw_label": ["Joy", "Joy", "Joy", "Sad", "Sad", "Angry"],
            "normalized_label": ["Joy", "Joy", "Joy", "Sad", "Sad", "Angry"],
            "text_length": [1, 1, 1, 1, 1, 1],
        }
    )

    filtered, summary = build_group_safe_frame(frame)

    assert filtered["clean_text"].tolist().count("same") == 1
    assert "conflict" not in set(filtered["clean_text"])
    assert summary["rows_removed_as_duplicate"] == 1
    assert summary["rows_removed_in_conflicting_groups"] == 2
    assert filtered["group_id"].notna().all()


def test_split_group_safe_keeps_shared_ids_and_text_in_one_split() -> None:
    rows: list[dict[str, object]] = []
    labels = ["Joy", "Sad", "Angry"]
    for index in range(90):
        rows.append(
            {
                "id": str(index // 2) if index < 12 else str(index),
                "clean_text": f"text-{index // 2}" if index < 12 else f"text-{index}",
                "task_label": labels[index % len(labels)],
                "raw_text": f"raw-{index}",
                "raw_label": labels[index % len(labels)],
                "normalized_label": labels[index % len(labels)],
                "text_length": 2,
                "group_id": index // 2 if index < 12 else index,
            }
        )
    frame = pd.DataFrame(rows)
    # Linked rows must carry one label for stratified group assignment.
    frame.loc[:11, "task_label"] = "Joy"

    train, validation, test = split_group_safe(
        frame,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_seed=42,
        stratify=True,
    )

    split_frames = {"train": train, "validation": validation, "test": test}
    for left_name, right_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left = split_frames[left_name]
        right = split_frames[right_name]
        assert set(left["group_id"]).isdisjoint(set(right["group_id"]))
        assert set(left["clean_text"]).isdisjoint(set(right["clean_text"]))
        assert set(left["id"]).isdisjoint(set(right["id"]))

