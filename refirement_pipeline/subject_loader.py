from loguru import logger


def load_subject_params(params_path):
    params = {}

    with open(params_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            params[key.strip()] = value.strip()

    height_m = float(params.get("height_m", 1.70))
    mass_kg = float(params.get("mass_kg", 70.0))
    sex = params.get("sex", "male").lower()

    logger.info(
        "Subject params: height_m={}, mass_kg={}, sex={}".format(
            height_m, mass_kg, sex
        )
    )

    return {
        "height_m": height_m,
        "mass_kg": mass_kg,
        "sex": sex,
    }