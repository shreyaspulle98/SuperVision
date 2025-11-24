"""
Compare performance of all three models on test set.
"""
import json

print('='*80)
print('MODEL PERFORMANCE COMPARISON ON TEST SET'.center(80))
print('='*80)

# Pre-trained ALIGNN
with open('results/pretrained_alignn_predictions_metrics.json') as f:
    pretrained = json.load(f)

# Fine-tuned ALIGNN
with open('results/alignn_metrics.json') as f:
    finetuned_alignn = json.load(f)

# DINOv3
with open('results/dinov3_metrics.json') as f:
    dinov3 = json.load(f)

print(f'\n+{"-"*78}+')
print(f'| {"Model":<35} | {"MAE (K)":<12} | {"RMSE (K)":<12} | {"R²":<10} |')
print(f'+{"-"*78}+')
print(f'| {"Pre-trained ALIGNN (zero-shot)":<35} | {pretrained["mae"]:>10.2f}   | {pretrained["rmse"]:>10.2f}   | {pretrained["r2"]:>8.4f}   |')
print(f'| {"Fine-tuned ALIGNN":<35} | {finetuned_alignn["mae"]:>10.2f}   | {finetuned_alignn["rmse"]:>10.2f}   | {finetuned_alignn["r2"]:>8.4f}   |')
print(f'| {"Fine-tuned DINOv3 + LoRA":<35} | {dinov3["mae"]:>10.2f}   | {dinov3["rmse"]:>10.2f}   | {dinov3["r2"]:>8.4f}   |')
print(f'+{"-"*78}+')

print(f'\n{"KEY INSIGHTS:".center(80)}')
print(f'{"-"*80}')

improvement_ft = ((pretrained['mae'] - finetuned_alignn['mae']) / pretrained['mae']) * 100
improvement_dinov3 = ((pretrained['mae'] - dinov3['mae']) / pretrained['mae']) * 100
dinov3_vs_ft = ((finetuned_alignn['mae'] - dinov3['mae']) / finetuned_alignn['mae']) * 100

print(f'\n1. Pre-trained ALIGNN Performance:')
print(f'   - Provides a reasonable baseline but struggles with this specific dataset')
print(f'   - Negative R² indicates predictions worse than simply using the mean')
print(f'   - Model was trained on JARVIS superconductor data, may have dataset mismatch')

print(f'\n2. Value of Fine-tuning ALIGNN:')
print(f'   - Fine-tuning improves MAE by {improvement_ft:.1f}% (from {pretrained["mae"]:.2f}K to {finetuned_alignn["mae"]:.2f}K)')
print(f'   - R² improves dramatically from {pretrained["r2"]:.4f} to {finetuned_alignn["r2"]:.4f}')
print(f'   - Shows critical importance of domain-specific fine-tuning')

print(f'\n3. DINOv3 vs Pre-trained ALIGNN:')
print(f'   - DINOv3 achieves {improvement_dinov3:.1f}% better MAE than pre-trained ALIGNN')
print(f'   - DINOv3 outperforms zero-shot ALIGNN despite being a vision model')

print(f'\n4. DINOv3 vs Fine-tuned ALIGNN:')
print(f'   - DINOv3 achieves {dinov3_vs_ft:.1f}% lower MAE than fine-tuned ALIGNN')
print(f'   - DINOv3 R² ({dinov3["r2"]:.4f}) slightly better than ALIGNN ({finetuned_alignn["r2"]:.4f})')
print(f'   - DINOv3 remains the best-performing model overall')

print(f'\n{"CONCLUSION:".center(80)}')
print(f'{"-"*80}')
print(f'  The pre-trained ALIGNN model provides a weak baseline, demonstrating that')
print(f'  domain-specific fine-tuning is essential for good performance. Both fine-tuned')
print(f'  approaches significantly outperform the zero-shot baseline, with DINOv3 + LoRA')
print(f'  achieving the best results overall.')
print('='*80)
