#!/bin/bash
# Extract test queries from chrM

SEQ=$(grep -v "^>" chrM.fna | tr -d '\n')
LEN=${#SEQ}

echo "Creating test queries from chrM (length: $LEN)"

# Query 1: 100bp from position 1000 (exact match)
echo "@read_exact_100bp" > queries_chrM.fq
echo "${SEQ:1000:100}" >> queries_chrM.fq
echo "+" >> queries_chrM.fq
echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII" >> queries_chrM.fq

# Query 2: 150bp from position 5000 (exact match)
echo "@read_exact_150bp" >> queries_chrM.fq
echo "${SEQ:5000:150}" >> queries_chrM.fq
echo "+" >> queries_chrM.fq
printf 'I%.0s' {1..150} >> queries_chrM.fq
echo "" >> queries_chrM.fq

# Query 3: 100bp from position 10000 with 2 substitutions
READ3="${SEQ:10000:100}"
READ3="${READ3:0:40}N${READ3:41:40}N${READ3:81}"
echo "@read_with_2N" >> queries_chrM.fq
echo "$READ3" >> queries_chrM.fq
echo "+" >> queries_chrM.fq
echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII" >> queries_chrM.fq

# Query 4: 80bp from position 15000 (exact match, testing different length)
echo "@read_exact_80bp" >> queries_chrM.fq
echo "${SEQ:15000:80}" >> queries_chrM.fq
echo "+" >> queries_chrM.fq
printf 'I%.0s' {1..80} >> queries_chrM.fq
echo "" >> queries_chrM.fq

echo "Created queries_chrM.fq with 4 test reads"
wc -l queries_chrM.fq
