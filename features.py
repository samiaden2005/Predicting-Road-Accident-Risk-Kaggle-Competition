"""Interaction features grounded in road-safety research.

Sources (see conversation): curvature x speed raises run-off-road risk;
rain/fog raise crash risk further at speed and on curves; curves cause
~32% of fatal nighttime crashes because the hazard itself is hard to see;
wider/more lanes offset curvature risk.
"""

NEW_NUMERIC = [
    "curvature_speed",
    "curvature_per_lane",
    "curvature_x_rainy",
    "curvature_x_foggy",
    "speed_x_rainy",
    "speed_x_foggy",
    "curvature_x_low_light",
    "low_light_x_bad_weather",
    "accident_history_curve",
]


def engineer_features(df):
    df = df.copy()

    is_rainy = (df["weather"] == "rainy").astype(int)
    is_foggy = (df["weather"] == "foggy").astype(int)
    is_low_light = df["lighting"].isin(["dim", "night"]).astype(int)

    df["curvature_speed"] = df["curvature"] * df["speed_limit"]
    df["curvature_per_lane"] = df["curvature"] / df["num_lanes"]
    df["curvature_x_rainy"] = df["curvature"] * is_rainy
    df["curvature_x_foggy"] = df["curvature"] * is_foggy
    df["speed_x_rainy"] = df["speed_limit"] * is_rainy
    df["speed_x_foggy"] = df["speed_limit"] * is_foggy
    df["curvature_x_low_light"] = df["curvature"] * is_low_light
    df["low_light_x_bad_weather"] = is_low_light * ((is_rainy + is_foggy) > 0).astype(int)
    df["accident_history_curve"] = df["num_reported_accidents"] * df["curvature"]

    return df
