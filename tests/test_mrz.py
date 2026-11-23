"""MRZ tests use the public ICAO 9303 specimens (fictional state UTO, fictional holder)."""

import pytest

from app.detection.mrz import check_digit, find_mrz, has_mrz_like_line

TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
TD1 = ["I<UTOD231458907<<<<<<<<<<<<<<<", "7408122F1204159UTO<<<<<<<<<<<6", "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"]
TD2 = ["I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<", "D231458907UTO7408122F1204159<<<<<<<6"]


@pytest.mark.parametrize("value,digit", [("L898902C3", "6"), ("740812", "2"), ("120415", "9"), ("D23145890", "7"),
                                         ("ZE184226B<<<<<", "1")])
def test_check_digit(value, digit):
    assert check_digit(value) == digit


@pytest.mark.parametrize("lines,fmt,doc_type", [(TD3, "TD3", "passport"), (TD1, "TD1", "national_id"),
                                                (TD2, "TD2", "national_id")])
def test_specimens_fully_valid(lines, fmt, doc_type):
    r = find_mrz("Some header text\n" + "\n".join(lines) + "\nfooter")
    assert r.format == fmt and r.all_valid and r.country == "UTO" and r.doc_type == doc_type


def test_ocr_noise_tolerated():
    noisy = [
        "p<uto ERIKSSON«ANNA<MARIA<<<<<<<<<<<<<<<<<<<",  # lower case, space, « for <<
        "L898902C36UTO74O8122F12041S9ZE184226B<<<<10",  # O/S in dates, one filler '<' missing
    ]
    r = find_mrz("\n".join(noisy))
    assert r is not None and r.format == "TD3" and r.all_valid


def test_names_line_with_lost_characters_still_valid():
    # Real OCR output from tests/synth passport: names line lost characters; line 2 has UT0 for UTO.
    ocr = ["P<UTOTEER<<SAMP<<<<<<<<<<<<<<<<<<<<<<<<<", "AB12345671UT09002018M3001019<<<<<<<<<<<<<<08"]
    r = find_mrz("\n".join(ocr))
    assert r is not None and r.format == "TD3" and r.all_valid and r.doc_type == "passport"


def test_tampered_check_digit_detected():
    bad = [TD3[0], TD3[1][:19] + "9" + TD3[1][20:]]  # birth-date check digit wrong
    r = find_mrz("\n".join(bad))
    assert r is not None and not r.all_valid and r.checks["birth_date"] is False


def test_residence_permit_code():
    lines = ["IR" + TD1[0][2:], TD1[1], TD1[2]]
    r = find_mrz("\n".join(lines))
    assert r.doc_type == "residence_permit"


def test_no_mrz_in_normal_text():
    assert find_mrz("فاتورة ضريبية\nINVOICE 2024/115\nTOTAL 1500") is None
    assert not has_mrz_like_line("Dear Customer,\nThanks")


def test_mrz_like_line():
    assert has_mrz_like_line("P<UTOTESTER<<SAMPLE<<<<<<<<<<<")


# --- parse_fields ---------------------------------------------------------------------------------------------

from datetime import date  # noqa: E402

from app.detection.mrz import parse_fields  # noqa: E402

TODAY = date(2026, 11, 19)


def test_parse_td3_specimen():
    f = parse_fields(find_mrz("\n".join(TD3)), TODAY)
    assert f == {
        "format": "TD3", "documentCode": "P", "issuingCountry": "UTO", "surname": "ERIKSSON",
        "givenNames": "ANNA MARIA", "documentNumber": "L898902C3", "nationality": "UTO", "birthDate": "1974-08-12",
        "sex": "F", "expiryDate": "2012-04-15", "optionalData": "ZE184226B", "expired": True, "valid": True,
        "checks": {"documentNumber": True, "birthDate": True, "expiryDate": True, "personalNumber": True,
                   "composite": True},
    }


@pytest.mark.parametrize("lines,fmt", [(TD1, "TD1"), (TD2, "TD2")])
def test_parse_td1_td2_specimens(lines, fmt):
    f = parse_fields(find_mrz("\n".join(lines)), TODAY)
    assert f["format"] == fmt and f["documentCode"] == "I" and f["documentNumber"] == "D23145890"
    assert (f["surname"], f["givenNames"]) == ("ERIKSSON", "ANNA MARIA")
    assert (f["birthDate"], f["expiryDate"], f["sex"], f["nationality"]) == ("1974-08-12", "2012-04-15", "F", "UTO")
    assert f["valid"] and f["expired"]


def test_parse_century_and_not_expired():
    from tests.synth import passport_mrz

    f = parse_fields(find_mrz("\n".join(passport_mrz(birth="100201"))), TODAY)
    assert f["valid"] and f["birthDate"] == "2010-02-01" and f["expiryDate"] == "2030-01-01" and f["expired"] is False


def test_parse_ocr_noise_in_dates_and_failed_check_kept():
    noisy = [TD3[0], TD3[1].replace("7408122", "74O8122")]  # O instead of 0 in the birth date
    f = parse_fields(find_mrz("\n".join(noisy)), TODAY)
    assert f["birthDate"] == "1974-08-12" and f["checks"]["birthDate"]
    tampered = [TD3[0], TD3[1][:13] + "740813" + TD3[1][19:]]  # wrong birth date: check digit fails
    f = parse_fields(find_mrz("\n".join(tampered)), TODAY)
    assert f["birthDate"] == "1974-08-13" and not f["checks"]["birthDate"] and not f["valid"]


def test_letter_fields_undo_digit_lookalikes():
    noisy = [TD3[0].replace("P<UTO", "P<UT0"), TD3[1].replace("UTO", "UT0", 1)]
    f = parse_fields(find_mrz("\n".join(noisy)), TODAY)
    assert f["issuingCountry"] == "UTO" and f["nationality"] == "UTO"


def test_names_reconciled_with_printed_text():
    lines = [TD3[0].replace("ERIKSSON", "ERISON"), TD3[1]]  # OCR lost two letters in the names line
    viz = "Surname: ERIKSSON\nGiven names: ANNA MARIA\n"
    f = parse_fields(find_mrz("\n".join(lines)), TODAY, text=viz + "\n".join(lines))
    assert f["surname"] == "ERIKSSON" and f["givenNames"] == "ANNA MARIA" and f["namesSource"] == "mrz+text"
    f = parse_fields(find_mrz("\n".join(lines)), TODAY, text="\n".join(lines))
    assert f["surname"] == "ERISON" and f["namesSource"] == "mrz"  # nothing to repair from
    ambiguous = "ERIKSSON ERICSSON\n"
    f = parse_fields(find_mrz("\n".join(lines)), TODAY, text=ambiguous)
    assert f["surname"] == "ERISON"  # two candidates: keep the MRZ reading
