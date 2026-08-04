# 3v3 Match Log Analysis

Source: `acceptance/remote/match_20260804_115349.json`  
SHA-256: `9a494e2f00cc751df2090339c2f7b2e3e99c75333fd6a3f2f4e8d6ef598ac2be`

Identity comes from the log's explicit `identities` map; connection order is not used. Falls use explicit false→true event edges. Match/team goal counts consume scored edges only from the unique ball authority `client_0`; other clients' scored flags are observational. Orientation thresholds are descriptive and do not create events. Distances are base-centre proxies, not foot contacts; reward components are unavailable.

| client | identity | fall events | fallen frames/rate | observed score edges | distance mean/min/final (m) |
|---|---|---:|---:|---:|---:|
| client_0 | A / attacker / ONNX | 1 | 1 / 0.050 | 0 | 1.643 / 1.401 / 2.182 |
| client_1 | A / keeper / ONNX | 0 | 0 / 0.000 | 0 | 4.944 / 3.891 / 3.891 |
| client_2 | B / defender / Rule | 0 | 0 / 0.000 | 0 | 3.265 / 3.200 / 3.280 |
| client_3 | B / keeper / Rule | 0 | 0 / 0.000 | 0 | 5.400 / 5.280 / 5.280 |
| client_4 | A / defender / ONNX | 0 | 0 / 0.000 | 0 | 2.483 / 1.556 / 1.556 |
| client_5 | B / attacker / Rule | 0 | 0 / 0.000 | 0 | 1.197 / 0.961 / 0.961 |

## Ball and events

Ball x moved from 0.0000 m to 0.3597 m (net 0.3597 m; observed range 0.0000–0.3597 m). Explicit score-event edges: 0; goal scored: **no**. No goal-line inference is used.

## Initial-condition caveat

exact identical initial robot states are coordinator spawn sentinels; distance windows start per client at first non-sentinel frame.
