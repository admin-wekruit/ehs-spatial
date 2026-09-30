
| sequence | field | n | median abs err (a) assumed 1.6 m | (b) true height | cov (a) | cov (b) | u median |
|---|---|---|---|---|---|---|---|
| arkit42 | base_above_floor | 57 | 18.6 cm | 1.2 cm | 1.00 | 0.96 | 24.2 cm |
| arkit42 | depth | 34 | 4.9 cm | 1.5 cm | 0.88 | 0.91 | 8.6 cm |
| arkit42 | height | 39 | 6.1 cm | 2.3 cm | 0.82 | 0.92 | 10.2 cm |
| arkit42 | position_xy | 57 | 42.8 cm | 6.2 cm | 1.00 | 1.00 | 61.7 cm |
| arkit42 | principal_axis_tilt_deg | 3 | 0.7 deg | 0.7 deg | 1.00 | 1.00 | 13.0 deg |
| arkit42 | top_above_floor | 54 | 22.1 cm | 1.9 cm | 0.96 | 0.96 | 26.2 cm |
| arkit42 | width | 45 | 9.2 cm | 1.6 cm | 0.91 | 0.89 | 15.6 cm |
| arkit47 | base_above_floor | 149 | 4.2 cm | 3.5 cm | 0.98 | 0.91 | 20.7 cm |
| arkit47 | depth | 24 | 2.8 cm | 3.9 cm | 0.92 | 0.92 | 16.4 cm |
| arkit47 | height | 85 | 3.3 cm | 3.0 cm | 0.91 | 0.88 | 10.1 cm |
| arkit47 | position_xy | 149 | 24.5 cm | 12.5 cm | 0.99 | 0.84 | 81.5 cm |
| arkit47 | principal_axis_tilt_deg | 3 | 2.2 deg | 2.2 deg | 1.00 | 1.00 | 14.5 deg |
| arkit47 | top_above_floor | 142 | 5.5 cm | 3.3 cm | 0.98 | 0.85 | 26.8 cm |
| arkit47 | width | 100 | 4.8 cm | 3.3 cm | 0.93 | 0.89 | 11.9 cm |
| tum | base_above_floor | 269 | 7.5 cm | 2.7 cm | 0.97 | 0.91 | 21.5 cm |
| tum | depth | 64 | 3.9 cm | 2.9 cm | 0.91 | 0.86 | 15.3 cm |
| tum | height | 129 | 2.2 cm | 2.0 cm | 0.91 | 0.88 | 8.5 cm |
| tum | planar_slope_deg | 13 | 0.9 deg | 0.9 deg | 1.00 | 1.00 | 3.1 deg |
| tum | position_xy | 269 | 18.3 cm | 7.5 cm | 0.99 | 0.98 | 77.9 cm |
| tum | principal_axis_tilt_deg | 10 | 2.2 deg | 2.2 deg | 1.00 | 1.00 | 11.5 deg |
| tum | top_above_floor | 256 | 7.5 cm | 2.8 cm | 0.98 | 0.92 | 23.3 cm |
| tum | width | 176 | 3.3 cm | 2.4 cm | 0.95 | 0.92 | 13.0 cm |

True camera height over the floor (the delivered scale assumes 1.6 m): arkit42 1.23-1.23 m; arkit47 1.48-1.50 m; tum 1.42-1.52 m

| bounds (warm calls, GT on >= 50 % of the mask) | n | hold (a) | hold (b) |
|---|---|---|---|
| arkit42 depth at most | 5 | 4 | 4 |
| arkit42 height at least | 3 | 3 | 3 |
| arkit42 height at most | 13 | 13 | 13 |
| arkit42 top_above_floor at least | 3 | 3 | 3 |
| arkit42 width at least | 4 | 4 | 4 |
| arkit42 width at most | 7 | 6 | 6 |
| arkit47 depth at most | 10 | 9 | 9 |
| arkit47 height at least | 14 | 13 | 13 |
| arkit47 height at most | 47 | 44 | 44 |
| arkit47 top_above_floor at least | 7 | 7 | 7 |
| arkit47 width at least | 8 | 8 | 8 |
| arkit47 width at most | 31 | 28 | 28 |
| tum depth at least | 1 | 1 | 1 |
| tum depth at most | 25 | 25 | 25 |
| tum height at least | 19 | 18 | 18 |
| tum height at most | 114 | 108 | 108 |
| tum top_above_floor at least | 13 | 12 | 12 |
| tum width at least | 15 | 10 | 10 |
| tum width at most | 73 | 66 | 66 |
