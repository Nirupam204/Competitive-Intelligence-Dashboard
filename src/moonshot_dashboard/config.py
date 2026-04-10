from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_BRANDS = [
    "Safari",
    "Skybags",
    "American Tourister",
    "VIP",
]

PRODUCTS_PER_BRAND = 10
MAX_REVIEWS_PER_PRODUCT = 8

ASPECT_KEYWORDS = {
    "wheels": ["wheel", "spinner", "rolling", "maneuver", "glide"],
    "handle": ["handle", "trolley", "grip", "pull rod", "rod"],
    "durability": ["durable", "durability", "sturdy", "broke", "broken", "crack", "damage"],
    "material": ["material", "shell", "fabric", "polycarbonate", "polypropylene", "zip"],
    "zipper": ["zip", "zipper", "chain"],
    "lock": ["lock", "combination", "tsa"],
    "space": ["space", "capacity", "spacious", "packing", "size"],
    "design": ["design", "look", "stylish", "color", "finish"],
    "value": ["price", "value", "worth", "money", "budget"],
}
