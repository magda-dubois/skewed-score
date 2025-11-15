PLOT_STYLE = {
    "title_size": 12,
    "label_size": 10,
    "tick_size": 10,
    "legend_size": 11,
    "legend_borderpad": 0.1,
    "legend_labelspacing": 0.1,
    "figure_size": (4, 4),
    "dpi": 600,
    "y_ticks": range(0, 11),
    "alpha": 0.6,
    "point_size": 2,
    "box_width": 0.7,
    "jitter": True
}

# Define grader colors
GRADER_COLORS = {
    # Human graders
    "Human X": ('#D0F0C0', '#2E8B57'),      # (Light green, Dark green)
    "Human Y": ('#B0E0E6', '#4682B4'),      # (Light blue, Steel blue)
    "Human Z": ('#E6E6FA', '#9370DB'),      # (Light lavender, Medium purple)
    
    # Autograder variants
    "Autograder A": ('#FFEFD5', '#CD853F'),  # (Light peach, Dark brown)
    "Autograder B": ('#FFE4E1', '#CD5C5C'),  # (Light pink, Indian red)
    "Autograder C": ('#FFF8DC', '#DAA520')   # (Light yellow, Goldenrod)
}

# Define aliases
GRADER_ALIASES = {
    "Human": "Human X",                # Human is Human X
    "Autograder": "Autograder A"       # Autograder is Autograder A
}

# Define names of grader sets for the different questions
GRADER_SETS = {
    "1autograder": ["Human", "Autograder"],
    "2autograders": ["Human", "Autograder A", "Autograder B"],
    "multiple_graders": ["Human X", "Human Y", "Human Z", "Autograder A", "Autograder B", "Autograder C"]
}