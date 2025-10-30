"""Cost values for the costmap."""

# Cost values for the costmap
FREE_SPACE = 0
LETHAL_OBSTACLE = 254
INSCRIBED_INFLATED_OBSTACLE = 253
MAX_NON_OBSTACLE = 252
NO_INFORMATION = 255

# Additional cost levels for different obstacle types
LOW_OBSTACLE = 100
MEDIUM_OBSTACLE = 150
HIGH_OBSTACLE = 200
