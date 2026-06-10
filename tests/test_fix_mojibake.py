from scripts.fix_mojibake import (
    count_mojibake_markers,
    fix_files,
    fix_text,
    should_accept_repair,
)


def test_fix_text_repairs_common_portuguese_mojibake():
    text = (
        "A intelig\u00c3\u00aancia artificial usa "
        "programa\u00c3\u00a7\u00c3\u00a3o e valida\u00c3\u00a7\u00c3\u00a3o."
    )

    fixed = fix_text(text)

    assert fixed == (
        "A intelig\u00eancia artificial usa programa\u00e7\u00e3o e "
        "valida\u00e7\u00e3o."
    )


def test_fix_text_keeps_clean_text_unchanged():
    text = "A intelig\u00eancia artificial usa programa\u00e7\u00e3o e valida\u00e7\u00e3o."

    fixed = fix_text(text)

    assert fixed == text


def test_should_accept_repair_requires_marker_reduction():
    original = "Texto limpo sem marcador."
    repaired = "Texto limpo sem marcador."

    assert should_accept_repair(original, repaired) is False


def test_count_mojibake_markers_counts_known_markers():
    assert count_mojibake_markers("\u00c3\u00a9 \u00c2 teste \ufffd") == 3


def test_fix_files_dry_run_does_not_write(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Programa\u00c3\u00a7\u00c3\u00a3o ajuda.", encoding="utf-8")

    changed = fix_files(tmp_path, apply=False, limit_preview=0)

    assert len(changed) == 1
    assert file_path.read_text(encoding="utf-8") == "Programa\u00c3\u00a7\u00c3\u00a3o ajuda."


def test_fix_files_apply_writes_repaired_text(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Programa\u00c3\u00a7\u00c3\u00a3o ajuda.", encoding="utf-8")

    changed = fix_files(tmp_path, apply=True, limit_preview=0)

    assert len(changed) == 1
    assert file_path.read_text(encoding="utf-8") == "Programa\u00e7\u00e3o ajuda."
