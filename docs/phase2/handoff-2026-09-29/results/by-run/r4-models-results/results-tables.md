## Models per call

| video | call | object cards | with a model | box / cylinder / plane / open frame | by type (overruled) | residual median / p90 cm | seen share (median) | one side | SAM 3D eligible / tried / accepted | first model / final models s (display start) | analysis s | GPU peaks GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| me340 | first | 261 | 261 (100%) | 167 / 38 / 45 / 11 | 65 (8) | 1.93 / 5.7 | 0.33 | 207 | 60 / 30 / 2 | 81.2 / 140.028 (59.009) | 230.057 | [61.2, 53.49] |
| me340 | warm | 254 | 254 (100%) | 175 / 33 / 35 / 11 | 45 (10) | 1.94 / 6.67 | 0.33 | 202 | 62 / 30 / 0 | - / 140.729 (55.943) | 230.255 | [66.8, 53.4] |
| samsclub-a2 | first | 623 | 623 (100%) | 553 / 34 / 32 / 4 | 33 (11) | 1.56 / 3.37 | 0.33 | 618 | 116 / 30 / 5 | 56.278 / 125.284 (39.507) | 227.785 | [69.08, 60.38] |
| samsclub-a2 | warm | 624 | 624 (100%) | 557 / 34 / 29 / 4 | 33 (11) | 1.56 / 3.34 | 0.33 | 619 | 117 / 30 / 5 | 58.887 / 129.914 (40.525) | 228.307 | [67.75, 60.08] |
| walmart | first | 719 | 719 (100%) | 577 / 46 / 58 / 38 | 58 (37) | 1.69 / 2.91 | 0.33 | 682 | 169 / 30 / 6 | 90.549 / 113.278 (34.876) | 226.716 | [61.92, 57.22] |
| walmart | warm | 721 | 721 (100%) | 582 / 47 / 57 / 35 | 55 (39) | 1.69 / 2.9 | 0.33 | 685 | 175 / 30 / 6 | 91.457 / 114.858 (32.922) | 226.352 | [63.04, 57.36] |

## Time added to the cards (this run vs the baseline run, same videos and calls; s)

| video | call | cards.v1 | cards.v3 | cards write | model fits CPU s (all processes) | cards v1 put | cards v3 put | cards bytes written (all versions) |
|---|---|---|---|---|---|---|---|---|
| me340 | first | 1.631 (base 1.837) | 2.534 (base 1.968) | 5.846 (base 8.032) | 3.843 | 35.944 (base 34.649) | 58.873 (base 57.092) | 4.2 MB (base 3.1) |
| me340 | warm | 1.689 (base 1.684) | 2.159 (base 1.594) | 7.401 (base 11.96) | 3.78 | 33.382 (base 30.857) | 55.815 (base 52.627) | 4.2 MB (base 3.1) |
| samsclub-a2 | first | 4.357 (base 4.239) | 3.631 (base 2.679) | 4.339 (base 8.66) | 9.11 | 26.186 (base 25.592) | 39.23 (base 38.03) | 9.9 MB (base 7.4) |
| samsclub-a2 | warm | 5.364 (base 4.769) | 3.303 (base 2.61) | 4.721 (base 10.922) | 9.156 | 26.363 (base 23.855) | 39.597 (base 35.363) | 9.9 MB (base 7.4) |
| walmart | first | 4.021 (base 3.67) | 3.27 (base 2.844) | 5.09 (base 21.547) | 9.516 | 20.864 (base 30.275) | 34.555 (base 42.192) | 10.2 MB (base 7.5) |
| walmart | warm | 3.602 (base 3.607) | 3.063 (base 2.685) | 6.222 (base 30.788) | 9.194 | 19.388 (base 22.827) | 32.609 (base 34.584) | 10.2 MB (base 7.6) |

## Audit by eye: 30 random models per video (agent-labelled; crop | model from the same camera)

| video | looked at | plausible | implausible | unclear |
|---|---|---|---|---|
| me340 | 30 | 19 | 7 | 4 |
| samsclub-a2 | 30 | 27 | 3 | 0 |
| walmart | 30 | 26 | 3 | 1 |
