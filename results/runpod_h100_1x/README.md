# RunPod 1xH100 実験結果 (2026-04-07)

## 環境
- GPU: NVIDIA H100 80GB HBM3 SXM x1
- Platform: RunPod
- Dataset: fineweb10B_sp1024 (1 shard)

## 結果

| Run | val_bpb | val_loss | Steps | Time | Model Size |
|-----|---------|----------|-------|------|------------|
| Baseline (9L, MLP2x) | **1.3599** | 2.2961 | 1,333/20,000 | 600s | 13.56MB (int8+zlib) |
| Improved (9L, MLP3x, SmearGate, BigramHash, SWA) | 1.5070 | 2.5446 | 958/20,000 | 600s | 16.24MB (int6+zlib) |

## 分析

改良版のスコアがベースラインより悪い原因:
- **train shard 1つ**のみ使用（データ不足）
- 改良版はバッチサイズが大きい (786K vs 524K tokens) ため、同じ10分でステップ数が少ない (958 vs 1,333)
- 改良版は本来 **8xH100 + 80 shards** で設計されている
- SWAが早期に開始 (step 50) し、収束前のチェックポイントが混入

## 次のステップ
- `--train-shards 10` 以上でデータ増量して再実行
- バッチサイズを1xH100向けに調整 (524K)
- 8xH100でフルスペック実行
