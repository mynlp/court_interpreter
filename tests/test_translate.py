import pytest

from court_interpreter.dataset_arrangement import sanitize_line


@pytest.mark.parametrize(
    ["input", "output"],
    [
        ["\t\t\t\t(J) I have a pen.", "I have a pen."],
        ["\t\t\t\t(裁) ほげほげ", "ほげほげ"],
        ["\t\t\t\t(a b c) d e f.", "d e f."],
        ["\t\t\t\t(a b c) ほげほげ", "ほげほげ"],
        [
            "\t\t\t\t(J) • Written statement of C given before the judicial police officer (copy) ",
            "Written statement of C given before the judicial police officer (copy) ",
        ],
        [
            "\t\t\t\t(J) • One plastic bag of a stimulant drug seized by this court (Seized Article No. XX-1 of XXXX) ",
            "One plastic bag of a stimulant drug seized by this court (Seized Article No. XX-1 of XXXX) ",
        ],
        [
            "\t\t\t\t(Thẩm phán) Tịch thu một con dao ngắn mà tòa đang giữ (vật thu giữ số OO -1, năm .....) ",
            "Tịch thu một con dao ngắn mà tòa đang giữ (vật thu giữ số OO -1, năm .....) ",
        ],
        [
            "\t\t\t\t(Thẩm phán) Tịch thu 1 chi phiếu giao ước (vật thu giữ số OO - 1, năm...) mà viện kiểm sát địa phương đang giữ. ",
            "Tịch thu 1 chi phiếu giao ước (vật thu giữ số OO - 1, năm...) mà viện kiểm sát địa phương đang giữ. ",
        ],
        [
            "\t\t\t\t(Thẩm phán) - Những mức án trước đối với các bị cáo đều bị bãi bỏ. ",
            "Những mức án trước đối với các bị cáo đều bị bãi bỏ. ",
        ],
    ],
)
def test_sanitize_line(input: str, output: str):
    pred = sanitize_line(input)

    assert pred == output
