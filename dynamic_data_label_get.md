# For baseline methods:

``` bash
# update data
for ratio in 0.2,0.4,0.6,0.8,1.0;do just append-data-cor 123 census13 original "$ratio";done

# update labels
for ratio in 0.2,0.4,0.6,0.8,1.0;do just wkld-label census13 original+original_cor_"$ratio" base;done
```
