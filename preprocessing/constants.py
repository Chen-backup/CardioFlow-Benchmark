"""Versioned dataset definitions and published split identities."""

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SPLITS = ("train", "val", "test")
CLASSES = {
    "ptbxl": ("CD", "HYP", "MI", "NORM", "STTC"),
    "cpsc2018": ("AF", "IAVB", "LBBB", "NSR", "PAC", "PVC", "RBBB", "STD", "STE"),
    "ludb": ("background", "P", "QRS", "T"),
}
CACHE_NAMES = {"ptbxl": "ptbxl", "cpsc2018": "cpsc2018_heartlang", "ludb": "ludb_chil2025"}
PROTOCOLS = {
    "ptbxl": "ptbxl_diagnostic_superclass_official_folds_100hz_v1",
    "cpsc2018": "cpsc2018_heartlang_center5_clean_v2",
    "ludb": "ludb_chil2025_lead_zscore_native500_to100_v1",
}
HEARTLANG_COMMIT = "0de902772c3620b53032da8f26b4f10e83111159"
LUDB_COMMIT = "7e401f6482e6830eb58a90288b3ddf2ba77a5308"
HEARTLANG_FILES = {
    "icbeb_train.csv": "013fa62513598e7a141d5ea810f73b6b75cdddbab25e11bcbbcdf84a18f000de",
    "icbeb_val.csv": "17fc4f6f95aa818e147ff3b6d5c314c9ae4301989469a0488aaa19638d3e44c8",
    "icbeb_test.csv": "ef85abba57aedba8bf857cf843457b5732f68adfd27b0f45af408870d4210a7f",
}
LUDB_FILES = {
    "train_for_public.csv": "dd77e8ac7047f748696fcef4dd30d4c5e486400cbc7bdc06f8d3a641b0f6b68e",
    "valid_for_public.csv": "18e953a50de3b523351732895123f45c6658ff79ce3cebf44b3fb4df21e61997",
    "test_for_public.csv": "027d47542f2c54c6ec77ee845148da6b320b493ba9750eb5fee1f03cb0fa07fd",
}
CPSC_COLUMNS = ("AFIB", "1AVB", "CLBBB", "NORM", "PAC", "VPC", "CRBBB", "STD", "STE")
# Positions in the pinned CSV concatenation: train, validation, test.
CPSC_REMOVED = (
    8, 124, 150, 180, 266, 292, 425, 426, 463, 749, 792, 793, 892, 928,
    1023, 1068, 1072, 1082, 1112, 1126, 1144, 1291, 1342, 1380, 1416,
    1473, 1515, 1655, 1678, 1720, 1725, 1732, 1741, 1746, 1748, 1790,
    1815, 1855, 1871, 1934, 1938, 1954, 2040, 2070, 2094, 2130, 2131,
    2155, 2250, 2253, 2254, 2255, 2521, 2553, 2663, 2744, 2784, 2788,
    2814, 2918, 2937, 3020, 3037, 3062, 3063, 3226, 3238, 3346, 3398,
    3458, 3476, 3501, 3523, 3674, 3712, 3772, 3926, 4075, 4092, 4109,
    4124, 4135, 4168, 4205, 4207, 4220, 4267, 4285, 4334, 4343, 4381,
    4494, 4567, 4623, 4637, 4643, 4722, 4780, 4797, 4809, 4972, 4980,
    5183, 5367, 5484, 5498,
)
LUDB_VAL = (1, 11, 24, 26, 30, 39, 58, 61, 69, 73, 75, 95, 96, 117, 141, 142, 143, 162, 166, 185)
LUDB_TEST = (12, 15, 25, 27, 29, 36, 41, 45, 57, 79, 80, 85, 93, 97, 136, 139, 152, 161, 173, 190)
