#ifndef SCV_GLOBAL_PLANNER__ROUTE_PLANNER_HPP_
#define SCV_GLOBAL_PLANNER__ROUTE_PLANNER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "scv_global_planner/route_graph.hpp"

namespace scv_global_planner
{

// One planning strategy. The node owns exactly one of these, chosen by the
// route_mode parameter, and never learns which it got.
//
// plan() returns node indices from start to goal inclusive. On failure it
// returns an empty vector and fills `reason` with something a human can act on.
class RoutePlanner
{
public:
    virtual ~RoutePlanner() = default;
    virtual std::vector<int> plan(const RouteGraph& graph, int start, int goal,
                                  std::string& reason) = 0;
    virtual const char* name() const = 0;
};

}  // namespace scv_global_planner

#endif  // SCV_GLOBAL_PLANNER__ROUTE_PLANNER_HPP_
