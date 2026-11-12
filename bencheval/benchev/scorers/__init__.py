from benchev.scorers import osworld, pointbox


def score(scorer, samples, preds, split_keys):
    if scorer == "pointbox":
        return pointbox.score(samples, preds, split_keys)
    if scorer == "osworld":
        return osworld.score(samples, preds, split_keys)
    raise ValueError(f"scorer {scorer!r} (medspot is scored by benchev.run via scorers.medspot)")
