# Single semantic area definition
string id
string type                          # drivable, crosswalk, sidewalk, no_entry, plaza
geometry_msgs/Polygon polygon        # Area boundary (GPS coordinates as x=lon, y=lat)
float64 speed_limit                  # m/s
int32 priority
bool traversable
float64 cost_weight
string remark
