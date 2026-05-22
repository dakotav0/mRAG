"""
examples/seed_npc_profiles.py — Seed PIDX profiles for a set of NPC characters.

Demonstrates how to generate PIDX bridge packets for multiple entities in bulk,
using the v0.1 bridge format with user origination (decay-exempt).

Run from project root:
    python examples/seed_npc_profiles.py --output-dir data/pidx/mailbox/

Each character gets one .bridge.json file containing:
  - A behavior register seed (class-biased, hand-tuned per archetype)
  - Alignment (moral × order)
  - Primary class and level
  - Identity: archetype, core traits, epistemic stance
  - origination: "user" throughout — foundational, decay-exempt

After generating, ingest with PIDX:
    pidx watch <mailbox_dir> <entity_id>   (one per entity)
  or batch:
    for f in data/pidx/mailbox/*.bridge.json; do
        entity_id=$(basename "$f" | sed 's/npc_//' | sed 's/_origin.bridge.json//')
        pidx ingest "$entity_id" "$f"
        pidx confirm-all "$entity_id"
    done
"""

import json
import os
import argparse
from datetime import datetime, timezone

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
SESSION_REF = "npc_origin_2026-04-17"
ORIGINATION = "user"
ORIENTATION = "user"


def make_packet(npc_id: str, observations: list[dict]) -> dict:
    return {
        "bridge_version": "0.1",
        "orientation": ORIENTATION,
        "session_ref": f"{SESSION_REF}_{npc_id}",
        "timestamp": TIMESTAMP,
        "observations": [
            {**obs, "origination": ORIGINATION}
            for obs in observations
        ]
    }


# ── Character definitions ─────────────────────────────────────────────────────
# Eight archetypes covering the main role slots (warrior, mystic, merchant,
# naturalist, artisan). Each is a standalone example — names are generic
# enough to be adapted to any setting.

NPC_ORIGINS = {

    "kira": make_packet("kira", [
        # Stat block — fighter: str-primary
        {"field": "stats.str", "value": 16},
        {"field": "stats.con", "value": 14},
        {"field": "stats.dex", "value": 13},
        {"field": "stats.wis", "value": 12},
        {"field": "stats.cha", "value": 10},
        {"field": "stats.int", "value":  8},
        # Alignment — pragmatic, no ideological lean
        {"field": "alignment.moral", "value": "neutral"},
        {"field": "alignment.order", "value": "neutral"},
        # Class & archetype
        {"field": "class.primary", "value": "fighter"},
        {"field": "class.level",   "value": 5},
        {"field": "identity.archetype", "value": "warrior"},
        # Core identity
        {"field": "identity.core", "value": "strikes first, asks later"},
        {"field": "identity.core", "value": "protects those who cannot protect themselves"},
        {"field": "identity.core", "value": "scars are just history written on skin"},
        {"field": "identity.stance", "value": "patrol — always moving, always assessing threat"},
        # Behavior register seeds (observed evidence slots, not stat-derived)
        {"field": "behavior.aggression",      "value": 0.78, "raw": "origin: fighter class + STR 16"},
        {"field": "behavior.loyalty",         "value": 0.80, "raw": "origin: protector archetype"},
        {"field": "behavior.sociability",     "value": 0.32, "raw": "origin: fighter, neutral alignment"},
        {"field": "behavior.caution",         "value": 0.38, "raw": "origin: low WIS, neutral order"},
        {"field": "identity.sub_archetype",   "value": None},  # blank — emerge through play
    ]),

    "vex": make_packet("vex", [
        # Stat block — warlock: cha-primary
        {"field": "stats.cha", "value": 17},
        {"field": "stats.con", "value": 14},
        {"field": "stats.wis", "value": 13},
        {"field": "stats.dex", "value": 11},
        {"field": "stats.int", "value": 10},
        {"field": "stats.str", "value":  7},
        {"field": "alignment.moral", "value": "neutral"},
        {"field": "alignment.order", "value": "chaotic"},
        {"field": "class.primary", "value": "warlock"},
        {"field": "class.level",   "value": 6},
        {"field": "identity.archetype", "value": "mystic"},
        {"field": "identity.core", "value": "speaks only what needs to be heard"},
        {"field": "identity.core", "value": "treats knowledge as currency, not gift"},
        {"field": "identity.core", "value": "serves something older than memory"},
        {"field": "identity.stance", "value": "observe — watches more than it acts"},
        {"field": "behavior.curiosity",       "value": 0.75, "raw": "origin: warlock pact, high WIS"},
        {"field": "behavior.caution",         "value": 0.68, "raw": "origin: chaotic order, survival-bent"},
        {"field": "behavior.sociability",     "value": 0.50, "raw": "origin: high CHA, selective use"},
        {"field": "behavior.aggression",      "value": 0.28, "raw": "origin: low STR, flee-first"},
        {"field": "identity.sub_archetype",   "value": None},
    ]),

    "lyra": make_packet("lyra", [
        # Stat block — wizard: int-primary
        {"field": "stats.int", "value": 18},
        {"field": "stats.wis", "value": 15},
        {"field": "stats.con", "value": 12},
        {"field": "stats.dex", "value": 11},
        {"field": "stats.cha", "value": 10},
        {"field": "stats.str", "value":  8},
        {"field": "alignment.moral", "value": "neutral"},
        {"field": "alignment.order", "value": "lawful"},
        {"field": "class.primary", "value": "wizard"},
        {"field": "class.level",   "value": 7},
        {"field": "identity.archetype", "value": "mystic"},
        {"field": "identity.core", "value": "everything is a pattern if you zoom out enough"},
        {"field": "identity.core", "value": "collects starlight observations the way others collect debts"},
        {"field": "identity.core", "value": "gentle in process, absolute in conclusions"},
        {"field": "identity.stance", "value": "study — processes before responding, rarely rushed"},
        {"field": "behavior.curiosity",       "value": 0.88, "raw": "origin: INT 18, wizard class"},
        {"field": "behavior.caution",         "value": 0.65, "raw": "origin: lawful order, high WIS"},
        {"field": "behavior.industriousness", "value": 0.60, "raw": "origin: lawful order, systematic"},
        {"field": "behavior.aggression",      "value": 0.22, "raw": "origin: low STR, lawful neutral"},
        {"field": "identity.sub_archetype",   "value": None},
    ]),

    "rowan": make_packet("rowan", [
        # Stat block — bard: cha-primary
        {"field": "stats.cha", "value": 17},
        {"field": "stats.dex", "value": 15},
        {"field": "stats.wis", "value": 13},
        {"field": "stats.int", "value": 12},
        {"field": "stats.con", "value": 10},
        {"field": "stats.str", "value":  9},
        {"field": "alignment.moral", "value": "good"},
        {"field": "alignment.order", "value": "chaotic"},
        {"field": "class.primary", "value": "bard"},
        {"field": "class.level",   "value": 4},
        {"field": "identity.archetype", "value": "merchant"},
        {"field": "identity.core", "value": "fair trade is the foundation of civilization"},
        {"field": "identity.core", "value": "remembers every face and forgets no debt"},
        {"field": "identity.core", "value": "disarms with warmth before weapons are even considered"},
        {"field": "identity.stance", "value": "engage — always finds the angle, always finds the mutual benefit"},
        {"field": "behavior.sociability",     "value": 0.88, "raw": "origin: CHA 17, good moral, bard class"},
        {"field": "behavior.caution",         "value": 0.72, "raw": "origin: merchant survival instinct"},
        {"field": "behavior.curiosity",       "value": 0.58, "raw": "origin: INT 12, bard information-gathering"},
        {"field": "behavior.aggression",      "value": 0.18, "raw": "origin: low STR, flee-primary"},
        {"field": "identity.sub_archetype",   "value": None},
    ]),

    "marina": make_packet("marina", [
        # Stat block — druid: wis-primary
        {"field": "stats.wis", "value": 16},
        {"field": "stats.con", "value": 15},
        {"field": "stats.int", "value": 13},
        {"field": "stats.dex", "value": 11},
        {"field": "stats.cha", "value": 10},
        {"field": "stats.str", "value":  8},
        {"field": "alignment.moral", "value": "neutral"},
        {"field": "alignment.order", "value": "neutral"},
        {"field": "class.primary", "value": "druid"},
        {"field": "class.level",   "value": 4},
        {"field": "identity.archetype", "value": "naturalist"},
        {"field": "identity.core", "value": "speaks slowly because the land does"},
        {"field": "identity.core", "value": "reads weather, soil, growth like text"},
        {"field": "identity.core", "value": "deep distrust of anything that does not grow"},
        {"field": "identity.stance", "value": "tend — nurtures before anything else"},
        {"field": "behavior.caution",         "value": 0.62, "raw": "origin: WIS 16, neutral alignment"},
        {"field": "behavior.industriousness", "value": 0.70, "raw": "origin: druid, CON 15"},
        {"field": "behavior.curiosity",       "value": 0.60, "raw": "origin: INT 13, naturalist observation"},
        {"field": "behavior.aggression",      "value": 0.28, "raw": "origin: low STR, neutral order"},
        {"field": "identity.sub_archetype",   "value": None},
    ]),

    "sage": make_packet("sage", [
        # Stat block — cleric: wis-primary, elder weights
        {"field": "stats.wis", "value": 18},
        {"field": "stats.cha", "value": 15},
        {"field": "stats.con", "value": 14},
        {"field": "stats.str", "value": 12},
        {"field": "stats.dex", "value": 10},
        {"field": "stats.int", "value":  9},
        {"field": "alignment.moral", "value": "good"},
        {"field": "alignment.order", "value": "neutral"},
        {"field": "class.primary", "value": "cleric"},
        {"field": "class.level",   "value": 12},  # elder — significantly experienced
        {"field": "identity.archetype", "value": "naturalist"},
        {"field": "identity.core", "value": "has seen this before, will not say when"},
        {"field": "identity.core", "value": "grief is just love with nowhere to go"},
        {"field": "identity.core", "value": "memory is the only currency that compounds"},
        {"field": "identity.stance", "value": "counsel — waits to be asked, then gives everything"},
        {"field": "behavior.loyalty",         "value": 0.85, "raw": "origin: good moral, cleric, level 12"},
        {"field": "behavior.caution",         "value": 0.70, "raw": "origin: WIS 18, elder judgment"},
        {"field": "behavior.sociability",     "value": 0.65, "raw": "origin: CHA 15, good moral"},
        {"field": "behavior.curiosity",       "value": 0.50, "raw": "origin: low INT, wisdom not intellect"},
        {"field": "behavior.aggression",      "value": 0.20, "raw": "origin: good/neutral, cleric"},
        {"field": "identity.sub_archetype",   "value": None},
    ]),

    "thane": make_packet("thane", [
        # Stat block — artificer: int-primary
        {"field": "stats.int", "value": 17},
        {"field": "stats.con", "value": 16},
        {"field": "stats.dex", "value": 14},
        {"field": "stats.wis", "value": 12},
        {"field": "stats.cha", "value": 10},
        {"field": "stats.str", "value":  9},
        {"field": "alignment.moral", "value": "neutral"},
        {"field": "alignment.order", "value": "lawful"},
        {"field": "class.primary", "value": "artificer"},
        {"field": "class.level",   "value": 6},
        {"field": "identity.archetype", "value": "artisan"},
        {"field": "identity.core", "value": "measures twice, speaks once"},
        {"field": "identity.core", "value": "a good tool outlives its maker — that is the point"},
        {"field": "identity.core", "value": "frustrated by imprecision, not by people"},
        {"field": "identity.stance", "value": "craft — in motion only when building something worthwhile"},
        {"field": "behavior.industriousness", "value": 0.90, "raw": "origin: artificer, lawful order, CON 16"},
        {"field": "behavior.curiosity",       "value": 0.78, "raw": "origin: INT 17, artificer class"},
        {"field": "behavior.loyalty",         "value": 0.70, "raw": "origin: lawful order, craftsman's bond"},
        {"field": "behavior.sociability",     "value": 0.28, "raw": "origin: low CHA, task-focused"},
        {"field": "behavior.aggression",      "value": 0.32, "raw": "origin: lawful neutral, defensive not offensive"},
        {"field": "identity.sub_archetype",   "value": None},
    ]),

    "grimm": make_packet("grimm", [
        # Stat block — fighter variant, builder-weighted
        {"field": "stats.str", "value": 17},
        {"field": "stats.con", "value": 16},
        {"field": "stats.dex", "value": 13},
        {"field": "stats.wis", "value": 12},
        {"field": "stats.int", "value": 10},
        {"field": "stats.cha", "value":  8},
        {"field": "alignment.moral", "value": "neutral"},
        {"field": "alignment.order", "value": "neutral"},
        {"field": "class.primary", "value": "fighter"},
        {"field": "class.level",   "value": 5},
        {"field": "identity.archetype", "value": "artisan"},
        {"field": "identity.core", "value": "if it is worth saying, it is worth building"},
        {"field": "identity.core", "value": "respects load-bearing walls more than people who do not understand them"},
        {"field": "identity.core", "value": "unassuming until you need something moved or made"},
        {"field": "identity.stance", "value": "labor — finds genuine calm in physical work"},
        {"field": "behavior.industriousness", "value": 0.82, "raw": "origin: fighter/builder, CON 16"},
        {"field": "behavior.loyalty",         "value": 0.72, "raw": "origin: neutral, dependable pattern"},
        {"field": "behavior.aggression",      "value": 0.45, "raw": "origin: STR 17, fighter class — defensive"},
        {"field": "behavior.sociability",     "value": 0.22, "raw": "origin: CHA 8, gruff"},
        {"field": "behavior.caution",         "value": 0.50, "raw": "origin: WIS 12, neutral order"},
        {"field": "identity.sub_archetype",   "value": None},
    ]),
}


def generate(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for npc_id, packet in NPC_ORIGINS.items():
        filename = f"npc_{npc_id}_origin.bridge.json"
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(packet, f, indent=2)
        print(f"  wrote {filename}")
    print(f"\n{len(NPC_ORIGINS)} origin profiles generated in {output_dir}")
    print("\nIngest with:")
    print("  for npc in kira vex lyra rowan marina sage thane grimm; do")
    print(f'    pidx ingest "npc_$npc" "{output_dir}/npc_${{npc}}_origin.bridge.json"')
    print(f'    pidx confirm-all "npc_$npc"')
    print("  done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/pidx/mailbox", help="Where to write .bridge.json files")
    args = parser.parse_args()
    generate(args.output_dir)
