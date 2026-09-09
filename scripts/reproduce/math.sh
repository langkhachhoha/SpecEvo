#!/usr/bin/env bash
# Mathematical discovery — the 7 tasks of Table 5.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

dispatch tasks/circle_packing        benchmarks/math/circle_packing               circle_packing
dispatch tasks/circle_packing_rect   benchmarks/math/circle_packing_rect          circle_packing_rect
dispatch tasks/heilbronn_triangle    benchmarks/math/heilbronn_triangle           heilbronn_triangle
dispatch tasks/heilbronn_convex_13   benchmarks/math/heilbronn_convex/13          heilbronn_convex_13
dispatch tasks/minmax_distance_2     benchmarks/math/minimizing_max_min_dist/2    minmax_distance_16_2
dispatch tasks/minmax_distance_3     benchmarks/math/minimizing_max_min_dist/3    minmax_distance_14_3
dispatch tasks/signal_processing     benchmarks/math/signal_processing            signal_processing
