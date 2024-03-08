## Naru
```bash
# census13 (0.2, 0.4, 0.6, 0.8, 1.0)
just update-naru original-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-33 census13 original+original_cor_0.2 base 123 40;
just test-naru original+original_cor_0.2-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 2000 census13 original+original_cor_0.2 base 123

just update-naru original-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 census13 original+original_cor_0.4 base 123 40;
just test-naru original+original_cor_0.4-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 2000 census13 original+original_cor_0.4 base 123

just update-naru original-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 census13 original+original_cor_0.6 base 123 40;
just test-naru original+original_cor_0.6-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 2000 census13 original+original_cor_0.6 base 123

just update-naru original-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 census13 original+original_cor_0.8 base 123 40;
just test-naru original+original_cor_0.8-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 2000 census13 original+original_cor_0.8 base 123

just update-naru original-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 census13 original+original_cor_1.0 base 123 40;
just test-naru original+original_cor_1.0-resmade_hid16,16,16,16_emb8_ep100_embedInembedOut_warm0-123 2000 census13 original+original_cor_1.0 base 123

# forest10 (0.2, 0.4, 0.6, 0.8, 1.0)
just update-naru original-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 forest10 original+original_cor_0.2 base 123 40
just test-naru original+original_cor_0.2-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 2000 forest10 original+original_cor_0.2 base 123

just update-naru original-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 forest10 original+original_cor_0.4 base 123 40
just test-naru original+original_cor_0.4-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 2000 forest10 original+original_cor_0.4 base 123

just update-naru original-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 forest10 original+original_cor_0.6 base 123 40
just test-naru original+original_cor_0.6-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 2000 forest10 original+original_cor_0.6 base 123

just update-naru original-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 forest10 original+original_cor_0.8 base 123 40
just test-naru original+original_cor_0.8-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 2000 forest10 original+original_cor_0.8 base 123

just update-naru original-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 forest10 original+original_cor_1.0 base 123 40
just test-naru original+original_cor_1.0-resmade_hid64,64,64,64_emb8_ep100_embedInembedOut_warm4000-123 2000 forest10 original+original_cor_1.0 base 123

# power7 (0.2, 0.4, 0.6, 0.8, 1.0)
just update-naru original-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 power7 original+original_cor_0.2 base 123 40
just test-naru original+original_cor_0.2-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 2000 power7 original+original_cor_0.2 base 123

just update-naru original-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 power7 original+original_cor_0.4 base 123 40
just test-naru original+original_cor_0.4-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 2000 power7 original+original_cor_0.4 base 123

just update-naru original-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 power7 original+original_cor_0.6 base 123 40
just test-naru original+original_cor_0.6-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 2000 power7 original+original_cor_0.6 base 123

just update-naru original-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 power7 original+original_cor_0.8 base 123 40
just test-naru original+original_cor_0.8-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 2000 power7 original+original_cor_0.8 base 123

just update-naru original-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 power7 original+original_cor_1.0 base 123 40
just test-naru original+original_cor_1.0-resmade_hid128,128,128,128,128_emb16_ep100_embedInembedOut_warm4000-123 2000 power7 original+original_cor_1.0 base 123

# dmv11 (0.2, 0.4, 0.6, 0.8, 1.0)
just update-naru original-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 dmv11 original+original_cor_0.2 base 123 40;
just test-naru original+original_cor_0.2-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 2000 dmv11 original+original_cor_0.2 base 123

just update-naru original-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 dmv11 original+original_cor_0.4 base 123 40;
just test-naru original+original_cor_0.4-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 2000 dmv11 original+original_cor_0.4 base 123

just update-naru original-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 dmv11 original+original_cor_0.6 base 123 40;
just test-naru original+original_cor_0.6-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 2000 dmv11 original+original_cor_0.6 base 123

just update-naru original-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 dmv11 original+original_cor_0.8 base 123 40;
just test-naru original+original_cor_0.8-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 2000 dmv11 original+original_cor_0.8 base 123

just update-naru original-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 dmv11 original+original_cor_1.0 base 123 40;
just test-naru original+original_cor_1.0-resmade_hid512,512,512,512_emb128_ep100_embedInembedOut_warm4000-123 2000 dmv11 original+original_cor_1.0 base 123
```

## MSCN
```bash
# census13 (0.2, 0.4, 0.6, 0.8, 1.0)
just train-mscn census13 original+original_cor_0.2 base 500 8 100 256 100000 0 123;
just test-mscn original+original_cor_0.2_base-mscn_hid8_sample500_ep100_bs256_100k-123 census13 original+original_cor_0.2 base 123

just train-mscn census13 original+original_cor_0.4 base 500 8 100 256 100000 0 123;
just test-mscn original+original_cor_0.4_base-mscn_hid8_sample500_ep100_bs256_100k-123 census13 original+original_cor_0.4 base 123

just train-mscn census13 original+original_cor_0.6 base 500 8 100 256 100000 0 123;
just test-mscn original+original_cor_0.6_base-mscn_hid8_sample500_ep100_bs256_100k-123 census13 original+original_cor_0.6 base 123

just train-mscn census13 original+original_cor_0.8 base 500 8 100 256 100000 0 123;
just test-mscn original+original_cor_0.8_base-mscn_hid8_sample500_ep100_bs256_100k-123 census13 original+original_cor_0.8 base 123

just train-mscn census13 original+original_cor_1.0 base 500 8 100 256 100000 0 123;
just test-mscn original+original_cor_1.0_base-mscn_hid8_sample500_ep100_bs256_100k-123 census13 original+original_cor_1.0 base 123

# forest10 (0.2, 0.4, 0.6, 0.8, 1.0)
just train-mscn forest10 original+original_cor_0.2 base 3000 32 100 256 100000 0 123;
just test-mscn original+original_cor_0.2_base-mscn_hid32_sample3000_ep100_bs256_100k-123 forest10 original+original_cor_0.2 base 123

just train-mscn forest10 original+original_cor_0.4 base 3000 32 100 256 100000 0 123;
just test-mscn original+original_cor_0.4_base-mscn_hid32_sample3000_ep100_bs256_100k-123 forest10 original+original_cor_0.4 base 123

just train-mscn forest10 original+original_cor_0.6 base 3000 32 100 256 100000 0 123;
just test-mscn original+original_cor_0.6_base-mscn_hid32_sample3000_ep100_bs256_100k-123 forest10 original+original_cor_0.6 base 123

just train-mscn forest10 original+original_cor_0.8 base 3000 32 100 256 100000 0 123;
just test-mscn original+original_cor_0.8_base-mscn_hid32_sample3000_ep100_bs256_100k-123 forest10 original+original_cor_0.8 base 123

just train-mscn forest10 original+original_cor_1.0 base 3000 32 100 256 100000 0 123;
just test-mscn original+original_cor_1.0_base-mscn_hid32_sample3000_ep100_bs256_100k-123 forest10 original+original_cor_1.0 base 123

# power7 (0.2, 0.4, 0.6, 0.8, 1.0)
just train-mscn power7 original+original_cor_0.2 base 5000 64 100 256 100000 0 123;
just test-mscn original+original_cor_0.2_base-mscn_hid64_sample5000_ep100_bs256_100k-123 power7 original+original_cor_0.2 base 123

just train-mscn power7 original+original_cor_0.4 base 5000 64 100 256 100000 0 123;
just test-mscn original+original_cor_0.4_base-mscn_hid64_sample5000_ep100_bs256_100k-123 power7 original+original_cor_0.4 base 123

just train-mscn power7 original+original_cor_0.6 base 5000 64 100 256 100000 0 123;
just test-mscn original+original_cor_0.6_base-mscn_hid64_sample5000_ep100_bs256_100k-123 power7 original+original_cor_0.6 base 123

just train-mscn power7 original+original_cor_0.8 base 5000 64 100 256 100000 0 123;
just test-mscn original+original_cor_0.8_base-mscn_hid64_sample5000_ep100_bs256_100k-123 power7 original+original_cor_0.8 base 123

just train-mscn power7 original+original_cor_1.0 base 5000 64 100 256 100000 0 123;
just test-mscn original+original_cor_1.0_base-mscn_hid64_sample5000_ep100_bs256_100k-123 power7 original+original_cor_1.0 base 123

# dmv11 (0.2, 0.4, 0.6, 0.8, 1.0)
just train-mscn dmv11 original+original_cor_0.2 base 10000 256 100 256 100000 0 123;
just test-mscn original+original_cor_0.2_base-mscn_hid256_sample10000_ep100_bs256_100k-123 dmv11 original+original_cor_0.2 base 123

just train-mscn dmv11 original+original_cor_0.4 base 10000 256 100 256 100000 0 123;
just test-mscn original+original_cor_0.4_base-mscn_hid256_sample10000_ep100_bs256_100k-123 dmv11 original+original_cor_0.4 base 123

just train-mscn dmv11 original+original_cor_0.6 base 10000 256 100 256 100000 0 123;
just test-mscn original+original_cor_0.6_base-mscn_hid256_sample10000_ep100_bs256_100k-123 dmv11 original+original_cor_0.6 base 123

just train-mscn dmv11 original+original_cor_0.8 base 10000 256 100 256 100000 0 123;
just test-mscn original+original_cor_0.8_base-mscn_hid256_sample10000_ep100_bs256_100k-123 dmv11 original+original_cor_0.8 base 123

just train-mscn dmv11 original+original_cor_1.0 base 10000 256 100 256 100000 0 123;
just test-mscn original+original_cor_1.0_base-mscn_hid256_sample10000_ep100_bs256_100k-123 dmv11 original+original_cor_1.0 base 123
```

## DeepDB
```bash
# census13 (0.2, 0.4, 0.6, 0.8, 1.0)
just train-deepdb census13 original+original_cor_0.2 1000000 0.4 0.01 0 base 123;
just test-deepdb census13-original+original_cor_0.2-spn_rdc0.4_ms0.01-123 census13 original+original_cor_0.2 base 123;

just train-deepdb census13 original+original_cor_0.4 1000000 0.4 0.01 0 base 123;
just test-deepdb census13-original+original_cor_0.4-spn_rdc0.4_ms0.01-123 census13 original+original_cor_0.4 base 123;

just train-deepdb census13 original+original_cor_0.6 1000000 0.4 0.01 0 base 123;
just test-deepdb census13-original+original_cor_0.6-spn_rdc0.4_ms0.01-123 census13 original+original_cor_0.6 base 123;

just train-deepdb census13 original+original_cor_0.8 1000000 0.4 0.01 0 base 123;
just test-deepdb census13-original+original_cor_0.8-spn_rdc0.4_ms0.01-123 census13 original+original_cor_0.8 base 123;

just train-deepdb census13 original+original_cor_1.0 1000000 0.4 0.01 0 base 123;
just test-deepdb census13-original+original_cor_1.0-spn_rdc0.4_ms0.01-123 census13 original+original_cor_1.0 base 123;

# forest10 (0.2, 0.4, 0.6, 0.8, 1.0)
just train-deepdb forest10 original+original_cor_0.2 1000000 0.4 0.005 0 base 123;
just test-deepdb forest10-original+original_cor_0.2-spn_rdc0.4_ms0.005-123 forest10 original+original_cor_0.2 base 123;

just train-deepdb forest10 original+original_cor_0.4 1000000 0.4 0.005 0 base 123;
just test-deepdb forest10-original+original_cor_0.4-spn_rdc0.4_ms0.005-123 forest10 original+original_cor_0.4 base 123;

just train-deepdb forest10 original+original_cor_0.6 1000000 0.4 0.005 0 base 123;
just test-deepdb forest10-original+original_cor_0.6-spn_rdc0.4_ms0.005-123 forest10 original+original_cor_0.6 base 123;

just train-deepdb forest10 original+original_cor_0.8 1000000 0.4 0.005 0 base 123;
just test-deepdb forest10-original+original_cor_0.8-spn_rdc0.4_ms0.005-123 forest10 original+original_cor_0.8 base 123;

just train-deepdb forest10 original+original_cor_1.0 1000000 0.4 0.005 0 base 123;
just test-deepdb forest10-original+original_cor_1.0-spn_rdc0.4_ms0.005-123 forest10 original+original_cor_1.0 base 123;

# power7 (0.2, 0.4, 0.6, 0.8, 1.0)
just train-deepdb power7 original+original_cor_0.2 10000000 0.3 0.001 0 base 123;
just test-deepdb power7-original+original_cor_0.2-spn_rdc0.3_ms0.001-123 power7 original+original_cor_0.2 base 123;

just train-deepdb power7 original+original_cor_0.4 10000000 0.3 0.001 0 base 123;
just test-deepdb power7-original+original_cor_0.4-spn_rdc0.3_ms0.001-123 power7 original+original_cor_0.4 base 123;

just train-deepdb power7 original+original_cor_0.6 10000000 0.3 0.001 0 base 123;
just test-deepdb power7-original+original_cor_0.6-spn_rdc0.3_ms0.001-123 power7 original+original_cor_0.6 base 123;

just train-deepdb power7 original+original_cor_0.8 10000000 0.3 0.001 0 base 123;
just test-deepdb power7-original+original_cor_0.8-spn_rdc0.3_ms0.001-123 power7 original+original_cor_0.8 base 123;

just train-deepdb power7 original+original_cor_1.0 10000000 0.3 0.001 0 base 123;
just test-deepdb power7-original+original_cor_1.0-spn_rdc0.3_ms0.001-123 power7 original+original_cor_1.0 base 123;

# dmv11 (0.2, 0.4, 0.6, 0.8, 1.0)
just train-deepdb dmv11 original+original_cor_0.2 1000000 0.2 0.001 0 base 123;
just test-deepdb dmv11-original+original_cor_0.2-spn_rdc0.2_ms0.001-123 dmv11 original+original_cor_0.2 base 123;

just train-deepdb dmv11 original+original_cor_0.4 1000000 0.2 0.001 0 base 123;
just test-deepdb dmv11-original+original_cor_0.4-spn_rdc0.2_ms0.001-123 dmv11 original+original_cor_0.4 base 123;

just train-deepdb dmv11 original+original_cor_0.6 1000000 0.2 0.001 0 base 123;
just test-deepdb dmv11-original+original_cor_0.6-spn_rdc0.2_ms0.001-123 dmv11 original+original_cor_0.6 base 123;

just train-deepdb dmv11 original+original_cor_0.8 1000000 0.2 0.001 0 base 123;
just test-deepdb dmv11-original+original_cor_0.8-spn_rdc0.2_ms0.001-123 dmv11 original+original_cor_0.8 base 123;

just train-deepdb dmv11 original+original_cor_1.0 1000000 0.2 0.001 0 base 123;
just test-deepdb dmv11-original+original_cor_1.0-spn_rdc0.2_ms0.001-123 dmv11 original+original_cor_1.0 base 123;
```



