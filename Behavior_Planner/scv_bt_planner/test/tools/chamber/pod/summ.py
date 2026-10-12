import json, sys
d = json.load(open(sys.argv[1]))
print({k: d.get(k) for k in ("verdict", "distance", "min_goal_dist", "duration", "n_violations", "blocked_seen", "min_groundtruth_z", "behavior_transitions")})
