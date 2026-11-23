from app.ocr.merge import Box, Rec, group_lines, merge_recognitions, page_text_and_confidence

MIN = 0.5


def test_pure_arabic_keeps_arabic_when_english_is_empty():
    ar = Rec("الجمهورية العربية", 0.97)
    assert merge_recognitions(ar, Rec("", 0.0), MIN) == ar


def test_latin_only_box_takes_more_confident_result():
    ar, en = Rec("NATI0NAL ID", 0.80), Rec("NATIONAL ID", 0.99)
    assert merge_recognitions(ar, en, MIN) == en


def test_mixed_box_combines_both_scripts():
    merged = merge_recognitions(Rec("الاسم سامي", 0.97), Rec("Name SAMPLE", 0.98), MIN)
    assert merged.text == "الاسم سامي Name SAMPLE"
    assert 0.97 <= merged.score <= 0.98


def test_mixed_box_replaces_partial_latin_from_arabic_model():
    merged = merge_recognitions(Rec("POTجواز سفر", 0.87), Rec("PASSPORT", 0.95), MIN)
    assert merged.text == "جواز سفر PASSPORT"


def test_arabic_model_already_has_latin_part():
    ar = Rec("فاتورة رقم 2024/115", 0.9)
    assert merge_recognitions(ar, Rec("2024/115", 0.99), MIN) == ar


def test_low_confidence_dropped():
    assert merge_recognitions(Rec("xx", 0.2), Rec("", 0.0), MIN) is None


def _box(x0, y0, x1, y1, text, score=0.9):
    return Box([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, score)


def test_lines_ordered_top_to_bottom_and_rtl_right_to_left():
    boxes = [
        _box(600, 10, 900, 50, "بطاقة"),   # right side, line 1
        _box(100, 12, 400, 52, "شخصية"),   # left side, line 1
        _box(100, 100, 600, 140, "NATIONAL ID"),
    ]
    lines = group_lines(boxes)
    assert [[b.text for b in line] for line in lines] == [["بطاقة", "شخصية"], ["NATIONAL ID"]]


def test_ltr_line_ordered_left_to_right():
    lines = group_lines([_box(500, 0, 700, 30, "CARD"), _box(0, 0, 400, 30, "IDENTITY")])
    assert [b.text for b in lines[0]] == ["IDENTITY", "CARD"]


def test_page_confidence_is_length_weighted():
    text, conf = page_text_and_confidence([_box(0, 0, 10, 10, "aaaa", 1.0), _box(0, 50, 10, 60, "b", 0.5)])
    assert text == "aaaa\nb"
    assert conf == 0.9
