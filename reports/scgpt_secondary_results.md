# scGPT secondary K562 evaluations

These two schemes reuse the original V0.1 QC, training-only HVG selection,
split assignments, output genes, and condition-macro evaluation. The baseline
rows come from the saved V0.1 results; scGPT rows are newly trained on the same
split. The original test sets were already inspected, so these comparisons are
exploratory rather than independent confirmation.

`seen` holds out a gemgroup while retaining the perturbation identities.
`unseen_gene` holds out every condition containing specified genes; unlike the
V0.1 identity-only models, scGPT has pretrained tokens for most held-out genes.

| scheme | model | n_conditions | pearson_delta | mse_all | top_20_recovery | top_100_recovery |
| --- | --- | --- | --- | --- | --- | --- |
| seen | ridge | 236 | 0.764078 | 0.003558 | 0.627754 | 0.565381 |
| seen | mlp | 236 | 0.768077 | 0.003348 | 0.635593 | 0.573051 |
| seen | mini_vc | 236 | 0.745418 | 0.003523 | 0.595551 | 0.559619 |
| seen | scgpt_frozen | 236 | 0.763006 | 0.003413 | 0.628178 | 0.570297 |
| seen | scgpt_finetuned | 236 | 0.763444 | 0.003417 | 0.630297 | 0.569449 |
| unseen_gene | ridge | 27 | 0.522390 | 0.009518 | 0.396296 | 0.492593 |
| unseen_gene | mlp | 27 | 0.458386 | 0.010876 | 0.393333 | 0.500667 |
| unseen_gene | mini_vc | 27 | 0.422115 | 0.011269 | 0.340741 | 0.455926 |
| unseen_gene | scgpt_frozen | 27 | 0.445031 | 0.009644 | 0.338889 | 0.467778 |
| unseen_gene | scgpt_finetuned | 27 | 0.464058 | 0.009290 | 0.357407 | 0.462222 |

The three missing scGPT perturbation tokens (C19orf26, C3orf72, KIAA1804)
receive separate trainable fallback vectors; an unseen fallback vector cannot
provide biological zero-shot semantics. The input audit in each model directory
records this mapping and the official checkpoint hashes.


On `seen`, the two scGPT arms are close to each other and to Ridge, while MLP
has the highest response Pearson. On `unseen_gene`, partial fine-tuning changes
response Pearson from 0.4450 to 0.4641
and all-gene MSE from 0.009644 to 0.009290.
Ridge has response Pearson 0.5224. The scGPT fine-tuned
arm has the lowest all-gene MSE in this 27-condition
split, but this does not establish broad unseen-gene transfer. The split is small,
was already examined in V0.1, and the target is a single K562 CRISPRa dataset.
