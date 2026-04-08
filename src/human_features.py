from high_level_features import HIGH_LEVEL_FEATURE_COLUMNS, compute_high_level_features

HUMAN_FEATURE_COLUMNS = HIGH_LEVEL_FEATURE_COLUMNS


def compute_human_features(row):
    return compute_high_level_features(row)
