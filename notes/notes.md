# Notes

## Text representation of a decision tree

Following [this](https://www.researchgate.net/publication/271213587_A_Survey_of_Merging_Decision_Trees_Data_Mining_Approaches) we can use this representation:


Example of a node with no children:
```
x <= x1: 0.20
x  > x1: 0.80
```

Example of a node with one child:
```
x <= x1: 0.20
x  > x1:
    x <= x2:
        x <= x3: 0.24
        x  > x3: 0.10
    x  > x2:
        x <= x4: 0.25
        x  > x4: 0.97
```

Example of a node with two child nodes:
```
x <= x1: 
    x <= x2:
        x <= x3: 0.24
        x  > x3: 0.10
    x  > x2:
        x <= x4: 0.25
        x  > x4: 0.97
x  > x1:
    x <= x5:
        x <= x6: 0.24
        x  > x6: 0.10
    x  > x5:
        x <= x7: 0.25
        x  > x7: 0.97
```

$x_1$, $x_2$, etc will correspond to the node's decision boundaries.