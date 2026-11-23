"""The 8 personal color types used in Korean personal color analysis, 4 drape colors each.

Colors are hex (#RRGGBB). Change them to anything you like.
Order = fingers: index, middle, ring, pinky.
"""

PALETTES = [
    ("Spring Warm Light", "warm, light, soft & clear", [
        ("Peach", "#FFB59A"), ("Light Coral", "#F99B8B"), ("Butter", "#F7DE95"), ("Light Aqua", "#9FD8C6")]),
    ("Spring Warm Bright", "warm, vivid & clear", [
        ("Coral Red", "#FF5A5F"), ("Tangerine", "#FF8A3D"), ("Sunflower", "#FFCF3A"), ("Turquoise", "#1FBFAE")]),
    ("Summer Cool Light", "cool, light & airy", [
        ("Baby Pink", "#F6C3D6"), ("Lavender", "#C6B5E3"), ("Powder Blue", "#A7C9E6"), ("Cool Mint", "#B3E0D6")]),
    ("Summer Cool Mute", "cool, soft & greyish", [
        ("Dusty Rose", "#C69AA5"), ("Mauve", "#A68AA6"), ("Grey Blue", "#8DA1B7"), ("Slate Teal", "#7FA19F")]),
    ("Autumn Warm Mute", "warm, soft & earthy", [
        ("Camel", "#C49A6C"), ("Terracotta", "#C0745A"), ("Olive", "#8C8A5A"), ("Muted Teal", "#5F8C86")]),
    ("Autumn Warm Deep", "warm, rich & dark", [
        ("Brick", "#9C3D2E"), ("Mustard", "#C9952B"), ("Forest Olive", "#4F5B2E"), ("Chocolate", "#5C3A21")]),
    ("Winter Cool Bright", "cool, vivid & high contrast", [
        ("Fuchsia", "#E0218A"), ("True Red", "#D7002A"), ("Royal Blue", "#1F4FD1"), ("Emerald", "#00936E")]),
    ("Winter Cool Deep", "cool, dark & intense", [
        ("Wine", "#6E1A37"), ("Navy", "#1C2B5A"), ("Plum", "#4B2259"), ("Black", "#141414")]),
]


def bgr(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
