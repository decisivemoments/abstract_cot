from abstract_cot.tokenization.abstract_vocab import AbstractTokenSpec, build_abstract_vocabulary


def test_build_abstract_vocabulary_uses_excel_style_labels():
    tokens = build_abstract_vocabulary(28)
    assert tokens[0] == "<TOKEN_A>"
    assert tokens[25] == "<TOKEN_Z>"
    assert tokens[26] == "<TOKEN_AA>"
    assert tokens[27] == "<TOKEN_AB>"


def test_token_spec_concatenates_all_added_tokens():
    spec = AbstractTokenSpec(abstract_tokens=["<TOKEN_A>", "<TOKEN_B>"])
    assert spec.special_tokens == ["<beginabstract>", "<endabstract>"]
    assert spec.all_added_tokens == [
        "<beginabstract>",
        "<endabstract>",
        "<TOKEN_A>",
        "<TOKEN_B>",
    ]
