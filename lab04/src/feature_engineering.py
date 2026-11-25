
def add_engineered_features(data):
    """
    Create two additional features.
    """

    data = data.copy()

    data["capital_net"] = (
        data["capital-gain"] - data["capital-loss"]
    )

    data["hours_per_age"] = (
        data["hours-per-week"] / (data["age"] - 16)
    )

    return data
