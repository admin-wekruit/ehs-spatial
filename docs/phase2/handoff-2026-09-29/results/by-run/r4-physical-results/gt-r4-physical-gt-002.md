
| sequence | field | n | median abs err (a) assumed 1.6 m | (b) true height | cov (a) | cov (b) | u median |
|---|---|---|---|---|---|---|---|
| arkit42 | base_above_floor | 58 | 18.7 cm | 1.2 cm | 1.00 | 0.97 | 24.3 cm |
| arkit42 | depth | 35 | 4.6 cm | 1.6 cm | 0.91 | 0.89 | 8.6 cm |
| arkit42 | height | 39 | 6.1 cm | 2.3 cm | 0.82 | 0.92 | 9.3 cm |
| arkit42 | position_xy | 58 | 43.0 cm | 6.2 cm | 1.00 | 1.00 | 63.9 cm |
| arkit42 | principal_axis_tilt_deg | 3 | 0.8 deg | 0.8 deg | 1.00 | 1.00 | 13.0 deg |
| arkit42 | top_above_floor | 55 | 21.8 cm | 1.9 cm | 0.96 | 0.96 | 26.4 cm |
| arkit42 | width | 44 | 9.3 cm | 1.7 cm | 0.91 | 0.89 | 14.9 cm |
| arkit47 | base_above_floor | 150 | 4.0 cm | 3.5 cm | 0.98 | 0.91 | 20.7 cm |
| arkit47 | depth | 25 | 2.8 cm | 3.9 cm | 0.88 | 0.88 | 15.4 cm |
| arkit47 | height | 86 | 3.3 cm | 3.0 cm | 0.91 | 0.88 | 10.1 cm |
| arkit47 | position_xy | 150 | 24.4 cm | 12.9 cm | 0.99 | 0.84 | 82.1 cm |
| arkit47 | principal_axis_tilt_deg | 3 | 2.2 deg | 2.2 deg | 1.00 | 1.00 | 14.5 deg |
| arkit47 | top_above_floor | 143 | 5.5 cm | 3.3 cm | 0.98 | 0.85 | 26.8 cm |
| arkit47 | width | 102 | 4.8 cm | 3.5 cm | 0.94 | 0.90 | 11.9 cm |
| tum | base_above_floor | 270 | 7.7 cm | 2.7 cm | 0.98 | 0.91 | 21.6 cm |
| tum | depth | 70 | 3.7 cm | 2.6 cm | 0.90 | 0.86 | 16.1 cm |
| tum | height | 124 | 2.4 cm | 2.2 cm | 0.90 | 0.88 | 8.9 cm |
| tum | planar_slope_deg | 15 | 1.0 deg | 1.0 deg | 1.00 | 1.00 | 3.5 deg |
| tum | position_xy | 270 | 18.3 cm | 7.9 cm | 0.99 | 0.97 | 80.0 cm |
| tum | principal_axis_tilt_deg | 9 | 2.1 deg | 2.1 deg | 1.00 | 1.00 | 11.0 deg |
| tum | top_above_floor | 259 | 7.5 cm | 2.8 cm | 0.98 | 0.91 | 23.5 cm |
| tum | width | 169 | 3.1 cm | 2.3 cm | 0.92 | 0.89 | 12.5 cm |

True camera height over the floor (the delivered scale assumes 1.6 m): arkit42 1.23-1.23 m; arkit47 1.48-1.50 m; tum 1.42-1.52 m

| bounds (warm calls, GT on >= 50 % of the mask) | n | hold (a) | hold (b) |
|---|---|---|---|
| arkit42 depth at least | 15 | 13 | 14 |
| arkit42 depth at most | 5 | 4 | 4 |
| arkit42 height at least | 3 | 3 | 3 |
| arkit42 height at most | 14 | 14 | 14 |
| arkit42 top_above_floor at least | 3 | 3 | 3 |
| arkit42 visible_length at least | 3 | 3 | 3 |
| arkit42 width at least | 5 | 5 | 5 |
| arkit42 width at most | 8 | 7 | 6 |
| arkit47 depth at least | 77 | 72 | 70 |
| arkit47 depth at most | 10 | 9 | 9 |
| arkit47 height at least | 14 | 13 | 12 |
| arkit47 height at most | 47 | 44 | 43 |
| arkit47 top_above_floor at least | 7 | 7 | 6 |
| arkit47 visible_length at least | 1 | 1 | 1 |
| arkit47 width at least | 8 | 8 | 8 |
| arkit47 width at most | 30 | 27 | 26 |
| tum depth at least | 117 | 103 | 102 |
| tum depth at most | 17 | 16 | 16 |
| tum height at least | 16 | 15 | 15 |
| tum height at most | 120 | 115 | 114 |
| tum top_above_floor at least | 11 | 10 | 9 |
| tum visible_length at least | 7 | 3 | 3 |
| tum width at least | 17 | 11 | 10 |
| tum width at most | 79 | 69 | 68 |
