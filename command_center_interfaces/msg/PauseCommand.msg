# Pause command message for controlling vehicle stops
# Used to command SMPPI controller to pause at specific locations

std_msgs/Header header

# Duration to pause (in seconds)
float64 pause_duration

# Node ID where pause should occur
string node_id

# Optional: reason for pause (for logging/debugging)
string reason