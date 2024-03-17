## The following codes are used for baseline model performance in static environments.

Naru: 

just train-naru [dataset] [version] [layers] [hc_hiddens] [embed_size]
[input_encoding] [output_encoding] [residual] [warmups] [sizelimit] [epochs] [workload] [seed]

just test-naru [model] [psample] [dataset] [version] [workload] [seed]

```bash
# census
just train-naru census13 original 4 16 8 embed embed True 0 0 100 base 123
just test-naru original-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 2000 census13 original base 123

# forest
just train-naru forest10 original 4 64 8 embed embed True 4000 0 100 base 123
just test-naru original-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 2000 forest10 original base 123

# power
just train-naru power7 original 5 128 16 embed embed True 4000 0 100 base 123
just test-naru original-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 2000 power7 original base 123

# dmv
just train-naru dmv11 original 4 512 128 embed embed True 4000 0 100 base 123
just test-naru original-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 2000 dmv11 original base 123
```

-------------------------------------------------------------------------------------------------------

MSCN:

just train-mscn [dataset] [version] [workload] [num_samples] [hid_units] [epochs] [bs] [train_num] [sizelimit] [seed]

just test-mscn [model] [dataset] [version] [workload] [seed]

```bash
# census
just train-mscn census13 original base 500 8 100 256 100000 0 123
just test-mscn original_base-mscn_hid8_sample500_ep100_bs256_100k-123 census13 original base 123

# forest
just train-mscn forest10 original base 3000 32 100 256 100000 0 123
just test-mscn original_base-mscn_hid32_sample3000_ep100_bs256_100k-123 forest10 original base 123

# power
just train-mscn power7 original base 5000 64 100 256 100000 0 123
just test-mscn original_base-mscn_hid64_sample5000_ep100_bs256_100k-123 power7 original base 123

# dmv
just train-mscn dmv11 original base 10000 256 100 256 100000 0 123
just test-mscn original_base-mscn_hid256_sample10000_ep100_bs256_100k-123 dmv11 original base 123
```

-----------------------------------------------------------------------------------

DeepDB：

just train-deepdb [dataset] [version] [hdf_sample_size] [rdc_threshold] [ratio_min_instance_slice] [sizelimit] [workload] [seed]

just test-deepdb [model] [dataset] [version] [workload] [seed]

```bash
# census
just train-deepdb census13 original 1000000 0.4 0.01 0 base 123
just test-deepdb original-spn_rdc0.4_ms0.01-123 census13 original base 123

# forest
just train-deepdb forest10 original 1000000 0.4 0.005 0 base 123
just test-deepdb original-spn_rdc0.4_ms0.005-123 forest10 original base 123

# power
just train-deepdb power7 original 10000000 0.3 0.001 0 base 123
just test-deepdb original-spn_rdc0.3_ms0.001-123 power7 original base 123

# dmv
just train-deepdb dmv11 original 1000000 0.2 0.001 0 base 123
just test-deepdb original-spn_rdc0.2_ms0.001-123 dmv11 original base 123
```
