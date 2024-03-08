# Get updated data and labels for baseline methods:

``` bash
# update data
for ratio in {0.2,0.4,0.6,0.8,1.0};do just append-data-cor 123 census13 original "$ratio";done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just append-data-cor 123 forest10 original "$ratio";done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just append-data-cor 123 power7 original "$ratio";done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just append-data-cor 123 dmv11 original "$ratio";done
```

```bash
# update labels
for ratio in {0.2,0.4,0.6,0.8,1.0};do just wkld-label census13 original+original_cor_"$ratio" base;done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just wkld-label forest10 original+original_cor_"$ratio" base;done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just wkld-label power7 original+original_cor_"$ratio" base;done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just wkld-label dmv11 original+original_cor_"$ratio" base;done
```

# Get updated data and labels for HORDE:
``` bash
# encode data, and store the vectors in ./lacarb/estimator/mine/vec_data/[dataset]_[version].pkl
for ratio in {0.2,0.4,0.6,0.8,1.0};do just mine-data_get census13 original+original_cor_"$ratio";done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just mine-data_get forest10 original+original_cor_"$ratio";done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just mine-data_get power7 original+original_cor_"$ratio";done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just mine-data_get dmv11 original+original_cor_"$ratio";done
```

```bash
# encode queries and labels
# store the train [query,label] vectors in ./lacarb/estimator/mine/vec_data/[dataset]_[version]_workload.pkl
# store the valid [query,label] vectors in ./lacarb/estimator/mine/vec_data/valid_[dataset]_[version]_workload.pkl
# store the test [query,label] vectors in ./lacarb/estimator/mine/vec_data/test_[dataset]_[version]_workload.pkl
for ratio in {0.2,0.4,0.6,0.8,1.0};do just train-query_get census13 original+original_cor_"$ratio";done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just train-query_get forest10 original+original_cor_"$ratio";done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just train-query_get power7 original+original_cor_"$ratio";done

for ratio in {0.2,0.4,0.6,0.8,1.0};do just train-query_get dmv11 original+original_cor_"$ratio";done
```
