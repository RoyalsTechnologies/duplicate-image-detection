"""Prompts and helpers for deciding whether an image relates to a public concern."""

CONCERN_PROMPTS: tuple[str, ...] = (
    "garbage dump or litter on the street",
    "flooded road or stagnant water",
    "pothole or damaged road surface",
    "air or water pollution",
    "blocked drain or open gutter",
    "broken streetlight or damaged public infrastructure",
    "sanitation problem or dirty public area",
    "environmental hazard in an urban area",
    "community infrastructure damage",
    "public safety issue on a street",
)

NON_CONCERN_PROMPTS: tuple[str, ...] = (
    "selfie portrait of a person",
    "food meal on a table",
    "pet cat or dog photo",
    "indoor home or office scene",
    "screenshot or meme image",
    "product advertisement or shopping",
    "nature landscape without damage",
    "random personal photo",
)
