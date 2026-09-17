# quality report

int8 row = the engine's load-time quantization (`nanollama -q`) reproduced
weight-for-weight in PyTorch.


| weights | size | val loss/acc | val ppl |
|---|---|---|---|
| fp32 (shipped .bin) | 0.67 MB | 4.7470 | 115.24 |
| int8 (engine `-q`) | 0.23 MB* | 4.7473 | 115.27 |

\* estimated in-memory size — the engine quantizes at load time; the shipped .bin stays fp32.

## samples — prompt: `Once upon a time`

**fp32**
```
asoyecqMe�wo�eb s�vwTpen8sadkRvnvdifiS
T�asaaSpdgRl�.mwvd.
e/efeffahtBw8wMtshft�iv�ys�bcSRSg Sai.clMmbew�lf8Mtil�c
yt��h ff�intO8rui�lT�sBw.h k�livTkkuwv cwe
```
**int8**
```
thnbeefsfBleu�. nw.Stncmdkwdhf
�sttstlbBuflRbeuttSMy Rl blSm�Op
fwfBThq�fomoooa/ud8uTlkgugtyh  khneR f cqRc��.Of �Bfulys tcpMmqhweufOOy8oROi�BaTcegTOgct�oqup
```