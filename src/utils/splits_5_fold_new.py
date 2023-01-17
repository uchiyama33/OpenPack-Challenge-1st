from openpack_toolkit.configs._schema import DataSplitConfig

OPENPACK_CHALLENGE_1_FOLD_SPLIT = DataSplitConfig(
    name="openpack-challenge-2022",
    train=[
        ["U0103", "S0100"],
        ["U0103", "S0200"],
        ["U0103", "S0300"],
        ["U0103", "S0400"],
        ["U0103", "S0500"],
        ["U0105", "S0100"],
        ["U0105", "S0200"],
        ["U0105", "S0300"],
        ["U0105", "S0400"],
        ["U0105", "S0500"],
        ["U0106", "S0100"],
        ["U0106", "S0200"],
        ["U0106", "S0300"],
        ["U0106", "S0400"],
        ["U0106", "S0500"],
        ["U0107", "S0100"],
        ["U0107", "S0200"],
        ["U0107", "S0300"],
        ["U0107", "S0400"],
        ["U0107", "S0500"],
        ["U0109", "S0100"],
        ["U0109", "S0200"],
        ["U0109", "S0300"],
        ["U0109", "S0400"],
        ["U0109", "S0500"],
        ["U0111", "S0100"],
        ["U0111", "S0200"],
        ["U0111", "S0300"],
        ["U0111", "S0400"],
        ["U0111", "S0500"],
        ["U0202", "S0100"],
        ["U0202", "S0200"],
        ["U0202", "S0300"],
        ["U0202", "S0400"],
        ["U0202", "S0500"],
        ["U0205", "S0100"],
        ["U0205", "S0200"],
        ["U0205", "S0300"],
        ["U0205", "S0400"],
        ["U0205", "S0500"],
        ["U0210", "S0100"],
        ["U0210", "S0200"],
        ["U0210", "S0300"],
        ["U0210", "S0400"],
        ["U0210", "S0500"],
    ],
    val=[
        ["U0101", "S0100"],
        ["U0101", "S0200"],
        ["U0101", "S0300"],
        ["U0101", "S0400"],
        ["U0101", "S0500"],
        ["U0102", "S0100"],
        ["U0102", "S0200"],
        ["U0102", "S0300"],
        ["U0102", "S0400"],
        ["U0102", "S0500"],
    ],
    test=[
        ["U0101", "S0100"],
        ["U0101", "S0200"],
        ["U0101", "S0300"],
        ["U0101", "S0400"],
        ["U0101", "S0500"],
        ["U0102", "S0100"],
        ["U0102", "S0200"],
        ["U0102", "S0300"],
        ["U0102", "S0400"],
        ["U0102", "S0500"],
    ],
    submission=[
        ["U0104", "S0100"],
        ["U0104", "S0300"],
        ["U0108", "S0100"],
        ["U0108", "S0200"],
        ["U0108", "S0300"],
        ["U0108", "S0500"],
        ["U0110", "S0100"],
        ["U0110", "S0200"],
        ["U0110", "S0300"],
        ["U0110", "S0400"],
        ["U0110", "S0500"],
        ["U0203", "S0100"],
        ["U0203", "S0200"],
        ["U0203", "S0300"],
        ["U0203", "S0400"],
        ["U0203", "S0500"],
        ["U0204", "S0200"],
        ["U0204", "S0300"],
        ["U0204", "S0400"],
        ["U0207", "S0300"],
        ["U0207", "S0400"],
        ["U0207", "S0500"],
    ],
)


OPENPACK_CHALLENGE_2_FOLD_SPLIT = DataSplitConfig(
    name="openpack-challenge-2022",
    train=[
        ["U0101", "S0100"],
        ["U0101", "S0200"],
        ["U0101", "S0300"],
        ["U0101", "S0400"],
        ["U0101", "S0500"],
        ["U0102", "S0100"],
        ["U0102", "S0200"],
        ["U0102", "S0300"],
        ["U0102", "S0400"],
        ["U0102", "S0500"],
        ["U0103", "S0100"],
        ["U0103", "S0200"],
        ["U0103", "S0300"],
        ["U0103", "S0400"],
        ["U0103", "S0500"],
        ["U0105", "S0100"],
        ["U0105", "S0200"],
        ["U0105", "S0300"],
        ["U0105", "S0400"],
        ["U0105", "S0500"],
        ["U0107", "S0100"],
        ["U0107", "S0200"],
        ["U0107", "S0300"],
        ["U0107", "S0400"],
        ["U0107", "S0500"],
        ["U0111", "S0100"],
        ["U0111", "S0200"],
        ["U0111", "S0300"],
        ["U0111", "S0400"],
        ["U0111", "S0500"],
        ["U0202", "S0100"],
        ["U0202", "S0200"],
        ["U0202", "S0300"],
        ["U0202", "S0400"],
        ["U0202", "S0500"],
        ["U0205", "S0100"],
        ["U0205", "S0200"],
        ["U0205", "S0300"],
        ["U0205", "S0400"],
        ["U0205", "S0500"],
        ["U0210", "S0100"],
        ["U0210", "S0200"],
        ["U0210", "S0300"],
        ["U0210", "S0400"],
        ["U0210", "S0500"],
    ],
    val=[
        ["U0106", "S0100"],
        ["U0106", "S0200"],
        ["U0106", "S0300"],
        ["U0106", "S0400"],
        ["U0106", "S0500"],
        ["U0109", "S0100"],
        ["U0109", "S0200"],
        ["U0109", "S0300"],
        ["U0109", "S0400"],
        ["U0109", "S0500"],
    ],
    test=[
        ["U0106", "S0100"],
        ["U0106", "S0200"],
        ["U0106", "S0300"],
        ["U0106", "S0400"],
        ["U0106", "S0500"],
        ["U0109", "S0100"],
        ["U0109", "S0200"],
        ["U0109", "S0300"],
        ["U0109", "S0400"],
        ["U0109", "S0500"],
    ],
    submission=[
        ["U0104", "S0100"],
        ["U0104", "S0300"],
        ["U0108", "S0100"],
        ["U0108", "S0200"],
        ["U0108", "S0300"],
        ["U0108", "S0500"],
        ["U0110", "S0100"],
        ["U0110", "S0200"],
        ["U0110", "S0300"],
        ["U0110", "S0400"],
        ["U0110", "S0500"],
        ["U0203", "S0100"],
        ["U0203", "S0200"],
        ["U0203", "S0300"],
        ["U0203", "S0400"],
        ["U0203", "S0500"],
        ["U0204", "S0200"],
        ["U0204", "S0300"],
        ["U0204", "S0400"],
        ["U0207", "S0300"],
        ["U0207", "S0400"],
        ["U0207", "S0500"],
    ],
)


OPENPACK_CHALLENGE_3_FOLD_SPLIT = DataSplitConfig(
    name="openpack-challenge-2022",
    train=[
        ["U0101", "S0100"],
        ["U0101", "S0200"],
        ["U0101", "S0300"],
        ["U0101", "S0400"],
        ["U0101", "S0500"],
        ["U0102", "S0100"],
        ["U0102", "S0200"],
        ["U0102", "S0300"],
        ["U0102", "S0400"],
        ["U0102", "S0500"],
        ["U0105", "S0100"],
        ["U0105", "S0200"],
        ["U0105", "S0300"],
        ["U0105", "S0400"],
        ["U0105", "S0500"],
        ["U0106", "S0100"],
        ["U0106", "S0200"],
        ["U0106", "S0300"],
        ["U0106", "S0400"],
        ["U0106", "S0500"],
        ["U0107", "S0100"],
        ["U0107", "S0200"],
        ["U0107", "S0300"],
        ["U0107", "S0400"],
        ["U0107", "S0500"],
        ["U0109", "S0100"],
        ["U0109", "S0200"],
        ["U0109", "S0300"],
        ["U0109", "S0400"],
        ["U0109", "S0500"],
        ["U0111", "S0100"],
        ["U0111", "S0200"],
        ["U0111", "S0300"],
        ["U0111", "S0400"],
        ["U0111", "S0500"],
        ["U0202", "S0100"],
        ["U0202", "S0200"],
        ["U0202", "S0300"],
        ["U0202", "S0400"],
        ["U0202", "S0500"],
        ["U0205", "S0100"],
        ["U0205", "S0200"],
        ["U0205", "S0300"],
        ["U0205", "S0400"],
        ["U0205", "S0500"],
    ],
    val=[
        ["U0103", "S0100"],
        ["U0103", "S0200"],
        ["U0103", "S0300"],
        ["U0103", "S0400"],
        ["U0103", "S0500"],
        ["U0210", "S0100"],
        ["U0210", "S0200"],
        ["U0210", "S0300"],
        ["U0210", "S0400"],
        ["U0210", "S0500"],
    ],
    test=[
        ["U0103", "S0100"],
        ["U0103", "S0200"],
        ["U0103", "S0300"],
        ["U0103", "S0400"],
        ["U0103", "S0500"],
        ["U0210", "S0100"],
        ["U0210", "S0200"],
        ["U0210", "S0300"],
        ["U0210", "S0400"],
        ["U0210", "S0500"],
    ],
    submission=[
        ["U0104", "S0100"],
        ["U0104", "S0300"],
        ["U0108", "S0100"],
        ["U0108", "S0200"],
        ["U0108", "S0300"],
        ["U0108", "S0500"],
        ["U0110", "S0100"],
        ["U0110", "S0200"],
        ["U0110", "S0300"],
        ["U0110", "S0400"],
        ["U0110", "S0500"],
        ["U0203", "S0100"],
        ["U0203", "S0200"],
        ["U0203", "S0300"],
        ["U0203", "S0400"],
        ["U0203", "S0500"],
        ["U0204", "S0200"],
        ["U0204", "S0300"],
        ["U0204", "S0400"],
        ["U0207", "S0300"],
        ["U0207", "S0400"],
        ["U0207", "S0500"],
    ],
)

OPENPACK_CHALLENGE_4_FOLD_SPLIT = DataSplitConfig(
    name="openpack-challenge-2022",
    train=[
        ["U0101", "S0100"],
        ["U0101", "S0200"],
        ["U0101", "S0300"],
        ["U0101", "S0400"],
        ["U0101", "S0500"],
        ["U0102", "S0100"],
        ["U0102", "S0200"],
        ["U0102", "S0300"],
        ["U0102", "S0400"],
        ["U0102", "S0500"],
        ["U0103", "S0100"],
        ["U0103", "S0200"],
        ["U0103", "S0300"],
        ["U0103", "S0400"],
        ["U0103", "S0500"],
        ["U0106", "S0100"],
        ["U0106", "S0200"],
        ["U0106", "S0300"],
        ["U0106", "S0400"],
        ["U0106", "S0500"],
        ["U0107", "S0100"],
        ["U0107", "S0200"],
        ["U0107", "S0300"],
        ["U0107", "S0400"],
        ["U0107", "S0500"],
        ["U0109", "S0100"],
        ["U0109", "S0200"],
        ["U0109", "S0300"],
        ["U0109", "S0400"],
        ["U0109", "S0500"],
        ["U0111", "S0100"],
        ["U0111", "S0200"],
        ["U0111", "S0300"],
        ["U0111", "S0400"],
        ["U0111", "S0500"],
        ["U0205", "S0100"],
        ["U0205", "S0200"],
        ["U0205", "S0300"],
        ["U0205", "S0400"],
        ["U0205", "S0500"],
        ["U0210", "S0100"],
        ["U0210", "S0200"],
        ["U0210", "S0300"],
        ["U0210", "S0400"],
        ["U0210", "S0500"],
    ],
    val=[
        ["U0105", "S0100"],
        ["U0105", "S0200"],
        ["U0105", "S0300"],
        ["U0105", "S0400"],
        ["U0105", "S0500"],
        ["U0202", "S0100"],
        ["U0202", "S0200"],
        ["U0202", "S0300"],
        ["U0202", "S0400"],
        ["U0202", "S0500"],
    ],
    test=[
        ["U0105", "S0100"],
        ["U0105", "S0200"],
        ["U0105", "S0300"],
        ["U0105", "S0400"],
        ["U0105", "S0500"],
        ["U0202", "S0100"],
        ["U0202", "S0200"],
        ["U0202", "S0300"],
        ["U0202", "S0400"],
        ["U0202", "S0500"],
    ],
    submission=[
        ["U0104", "S0100"],
        ["U0104", "S0300"],
        ["U0108", "S0100"],
        ["U0108", "S0200"],
        ["U0108", "S0300"],
        ["U0108", "S0500"],
        ["U0110", "S0100"],
        ["U0110", "S0200"],
        ["U0110", "S0300"],
        ["U0110", "S0400"],
        ["U0110", "S0500"],
        ["U0203", "S0100"],
        ["U0203", "S0200"],
        ["U0203", "S0300"],
        ["U0203", "S0400"],
        ["U0203", "S0500"],
        ["U0204", "S0200"],
        ["U0204", "S0300"],
        ["U0204", "S0400"],
        ["U0207", "S0300"],
        ["U0207", "S0400"],
        ["U0207", "S0500"],
    ],
)


OPENPACK_CHALLENGE_5_FOLD_SPLIT = DataSplitConfig(
    name="openpack-challenge-2022",
    train=[
        ["U0101", "S0100"],
        ["U0101", "S0200"],
        ["U0101", "S0300"],
        ["U0101", "S0400"],
        ["U0101", "S0500"],
        ["U0102", "S0100"],
        ["U0102", "S0200"],
        ["U0102", "S0300"],
        ["U0102", "S0400"],
        ["U0102", "S0500"],
        ["U0103", "S0100"],
        ["U0103", "S0200"],
        ["U0103", "S0300"],
        ["U0103", "S0400"],
        ["U0103", "S0500"],
        ["U0105", "S0100"],
        ["U0105", "S0200"],
        ["U0105", "S0300"],
        ["U0105", "S0400"],
        ["U0105", "S0500"],
        ["U0106", "S0100"],
        ["U0106", "S0200"],
        ["U0106", "S0300"],
        ["U0106", "S0400"],
        ["U0106", "S0500"],
        ["U0109", "S0100"],
        ["U0109", "S0200"],
        ["U0109", "S0300"],
        ["U0109", "S0400"],
        ["U0109", "S0500"],
        ["U0111", "S0100"],
        ["U0111", "S0200"],
        ["U0111", "S0300"],
        ["U0111", "S0400"],
        ["U0111", "S0500"],
        ["U0202", "S0100"],
        ["U0202", "S0200"],
        ["U0202", "S0300"],
        ["U0202", "S0400"],
        ["U0202", "S0500"],
        ["U0210", "S0100"],
        ["U0210", "S0200"],
        ["U0210", "S0300"],
        ["U0210", "S0400"],
        ["U0210", "S0500"],
    ],
    val=[
        ["U0107", "S0100"],
        ["U0107", "S0200"],
        ["U0107", "S0300"],
        ["U0107", "S0400"],
        ["U0107", "S0500"],
        ["U0205", "S0100"],
        ["U0205", "S0200"],
        ["U0205", "S0300"],
        ["U0205", "S0400"],
        ["U0205", "S0500"],
    ],
    test=[
        ["U0107", "S0100"],
        ["U0107", "S0200"],
        ["U0107", "S0300"],
        ["U0107", "S0400"],
        ["U0107", "S0500"],
        ["U0205", "S0100"],
        ["U0205", "S0200"],
        ["U0205", "S0300"],
        ["U0205", "S0400"],
        ["U0205", "S0500"],
    ],
    submission=[
        ["U0104", "S0100"],
        ["U0104", "S0300"],
        ["U0108", "S0100"],
        ["U0108", "S0200"],
        ["U0108", "S0300"],
        ["U0108", "S0500"],
        ["U0110", "S0100"],
        ["U0110", "S0200"],
        ["U0110", "S0300"],
        ["U0110", "S0400"],
        ["U0110", "S0500"],
        ["U0203", "S0100"],
        ["U0203", "S0200"],
        ["U0203", "S0300"],
        ["U0203", "S0400"],
        ["U0203", "S0500"],
        ["U0204", "S0200"],
        ["U0204", "S0300"],
        ["U0204", "S0400"],
        ["U0207", "S0300"],
        ["U0207", "S0400"],
        ["U0207", "S0500"],
    ],
)
