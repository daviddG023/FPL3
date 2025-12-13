import pandas as pd

def row_to_text(row: pd.Series) -> str:
    """
    Convert a single CSV row (one player in one fixture) into a natural-language
    description that is friendly for text embeddings.

    - Ignores stats that are 0 or NaN.
    - Only mentions saves if the player is a goalkeeper.
    - Does not assume whether the player is from the home or away team.
    """

    name = str(row.get("name", "")).strip()
    position = str(row.get("position", "")).strip()
    season = row.get("season")
    gw = row.get("GW")
    home_team = row.get("home_team")
    away_team = row.get("away_team")
    kickoff = row.get("kickoff_time")

    # Map short FPL position codes to natural words
    pos_map = {
        "GK": "goalkeeper",
        "GKP": "goalkeeper",
        "DEF": "defender",
        "MID": "midfielder",
        "FWD": "forward",
    }
    pos_upper = position.upper()
    pos_nice = pos_map.get(pos_upper, position.lower() if position else "player")

    subject = name if name else "The player"

    sentences = []

    # ── Intro sentence: season, GW, position, fixture ─────────────────────────────
    intro_parts = []

    if pd.notna(season):
        if pd.notna(gw):
            intro_parts.append(f"In Gameweek {int(gw)} of the {season} season")
        else:
            intro_parts.append(f"In the {season} season")
    elif pd.notna(gw):
        intro_parts.append(f"In Gameweek {int(gw)}")

    if intro_parts:
        intro_sentence = " ".join(intro_parts) + f", {subject} played as a {pos_nice}"
    else:
        intro_sentence = f"{subject} played as a {pos_nice}"

    if pd.notna(home_team) and pd.notna(away_team):
        intro_sentence += f" in the fixture between {home_team} and {away_team}"

    if pd.notna(kickoff):
        intro_sentence += f" on {str(kickoff)}"

    intro_sentence += "."
    sentences.append(intro_sentence)

    # ── Stats phrases (only non-zero) ─────────────────────────────────────────────
    stats_phrases = []

    def add_int_stat(col: str, template: str):
        val = row.get(col)
        if pd.notna(val) and int(val) > 0:
            n = int(val)
            stats_phrases.append(template.format(n=n))

    def add_float_stat(col: str, template: str):
        val = row.get(col)
        if pd.notna(val) and float(val) > 0:
            stats_phrases.append(template.format(v=float(val)))

    # Core stats
    minutes = row.get("minutes")
    if pd.notna(minutes) and int(minutes) > 0:
        stats_phrases.append(f"played {int(minutes)} minutes")

    add_int_stat("goals_scored", "scored {n} goal" + ("s" if row.get("goals_scored", 0) != 1 else ""))
    add_int_stat("assists", "provided {n} assist" + ("s" if row.get("assists", 0) != 1 else ""))
    add_int_stat("clean_sheets", "kept {n} clean sheet" + ("s" if row.get("clean_sheets", 0) != 1 else ""))
    add_int_stat("goals_conceded", "conceded {n} goal" + ("s" if row.get("goals_conceded", 0) != 1 else ""))

    # Saves -> only if goalkeeper
    if pos_upper in ("GK", "GKP", "GOALKEEPER"):
        add_int_stat("saves", "made {n} save" + ("s" if row.get("saves", 0) != 1 else ""))

    add_int_stat("bonus", "earned {n} bonus point" + ("s" if row.get("bonus", 0) != 1 else ""))
    add_int_stat("total_points", "returned {n} FPL point" + ("s" if row.get("total_points", 0) != 1 else ""))

    add_int_stat("yellow_cards", "received {n} yellow card" + ("s" if row.get("yellow_cards", 0) != 1 else ""))
    add_int_stat("red_cards", "received {n} red card" + ("s" if row.get("red_cards", 0) != 1 else ""))

    # Advanced metrics (only if > 0)
    add_float_stat("ict_index", "had an ICT Index of {v:.1f}")
    add_float_stat("influence", "had an Influence score of {v:.1f}")
    add_float_stat("creativity", "had a Creativity score of {v:.1f}")
    add_float_stat("threat", "had a Threat score of {v:.1f}")
    add_float_stat("form", "had a form value of {v:.1f}")

    # Build stats sentence if we have any non-zero stats
    if stats_phrases:
        if len(stats_phrases) == 1:
            stats_sentence = f"{subject} {stats_phrases[0]}."
        else:
            stats_sentence = (
                f"{subject} "
                + ", ".join(stats_phrases[:-1])
                + " and "
                + stats_phrases[-1]
                + "."
            )
        sentences.append(stats_sentence)
    else:
        # No meaningful stats (everything 0) → still give some text so embeddings aren't empty
        sentences.append(f"{subject} did not record any notable FPL statistics in this match.")

    return " ".join(sentences)



if __name__ == "__main__":
    row = pd.read_csv("fpl_two_seasons.csv").iloc[10000]
    print(row_to_text(row))