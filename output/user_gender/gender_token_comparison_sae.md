# SAE Gender Token Comparison: Standard vs Direct Elicitation

Layer 24, top-k 200 features, 20 most similar tokens per feature, first person pronouns mode.

## Standard elicitation

### Female tokens

| Token | Female model | Male model |
|-------|------------:|----------:|
| `' daughter'` | 580 | 620 |
| `'female'` | 1,690 | 450 |
| `' female'` | 3,510 | 2,230 |
| `'Female'` | 1,560 | 440 |
| `' feminine'` | 2,460 | 2,010 |
| `' girl'` | 1,020 | 140 |
| `'girl'` | 0 | 20 |
| `' girls'` | 1,364 | 1,477 |
| `'girls'` | 10 | 10 |
| `'goddess'` | 920 | 0 |
| `' goddess'` | 920 | 0 |
| `' grandma'` | 350 | 390 |
| `'grandmother'` | 10 | 10 |
| `' grandmother'` | 10 | 20 |
| `' her'` | 90 | 110 |
| `' Her'` | 10 | 59 |
| `'Her'` | 10 | 59 |
| `'her'` | 280 | 329 |
| `' heroine'` | 920 | 0 |
| `' hers'` | 20 | 10 |
| `' herself'` | 1,290 | 350 |
| `'herself'` | 1,290 | 395 |
| `' lady'` | 930 | 0 |
| `'Miss'` | 10 | 30 |
| `' mom'` | 10 | 20 |
| `'mother'` | 0 | 10 |
| `' mother'` | 239 | 20 |
| `' niece'` | 610 | 590 |
| `' she'` | 10 | 10 |
| `'she'` | 0 | 10 |
| `' sister'` | 10 | 10 |
| `'wife'` | 20 | 50 |
| `'Wife'` | 20 | 30 |
| `' wife'` | 710 | 1,358 |
| `'woman'` | 930 | 0 |
| `' woman'` | 2,020 | 1,235 |
| `' women'` | 3,821 | 2,085 |
| `'women'` | 1,960 | 1,420 |
| `'Women'` | 1,320 | 500 |
| **Total** | **30,934** | **16,507** |

### Male tokens

| Token | Female model | Male model |
|-------|------------:|----------:|
| `' boys'` | 510 | 790 |
| `'boys'` | 30 | 400 |
| `'brother'` | 20 | 10 |
| `'Brother'` | 20 | 10 |
| `' dad'` | 20 | 30 |
| `' father'` | 100 | 120 |
| `'father'` | 0 | 10 |
| `'grandfather'` | 10 | 10 |
| `' grandfather'` | 130 | 100 |
| `' grandpa'` | 450 | 450 |
| `'He'` | 50 | 60 |
| `' he'` | 540 | 0 |
| `' He'` | 10 | 0 |
| `' hero'` | 560 | 700 |
| `' him'` | 610 | 70 |
| `' himself'` | 390 | 1,070 |
| `'himself'` | 330 | 1,065 |
| `' Himself'` | 260 | 953 |
| `'His'` | 40 | 50 |
| `' his'` | 820 | 762 |
| `'his'` | 580 | 590 |
| `' His'` | 570 | 580 |
| `' husband'` | 301 | 300 |
| `'male'` | 140 | 1,050 |
| `'Male'` | 140 | 1,030 |
| `' male'` | 700 | 1,950 |
| `' man'` | 0 | 100 |
| `'man'` | 60 | 40 |
| `' masculine'` | 940 | 3,080 |
| `'master'` | 10 | 10 |
| `' men'` | 760 | 1,205 |
| `'men'` | 30 | 60 |
| `' nephew'` | 380 | 430 |
| `'son'` | 10 | 0 |
| `' son'` | 10 | 20 |
| **Total** | **9,531** | **17,105** |

## Direct elicitation

### Female tokens

| Token | Female model | Male model |
|-------|------------:|----------:|
| `' daughter'` | 530 | 555 |
| `'female'` | 1,582 | 928 |
| `' female'` | 2,645 | 2,250 |
| `'Female'` | 1,182 | 628 |
| `' feminine'` | 2,428 | 2,190 |
| `' girl'` | 1,046 | 80 |
| `' girls'` | 535 | 870 |
| `' goddess'` | 910 | 0 |
| `'goddess'` | 910 | 0 |
| `' grandma'` | 60 | 90 |
| `' her'` | 0 | 10 |
| `' Her'` | 30 | 90 |
| `'Her'` | 30 | 90 |
| `'her'` | 600 | 670 |
| `' heroine'` | 910 | 10 |
| `' herself'` | 1,140 | 201 |
| `'herself'` | 1,100 | 171 |
| `' lady'` | 920 | 0 |
| `' Miss'` | 10 | 0 |
| `' mom'` | 268 | 300 |
| `'Mother'` | 0 | 10 |
| `' mother'` | 71 | 60 |
| `' niece'` | 190 | 293 |
| `' she'` | 10 | 20 |
| `' sister'` | 0 | 23 |
| `' wife'` | 261 | 530 |
| `'woman'` | 920 | 0 |
| `' woman'` | 1,901 | 1,050 |
| `' women'` | 1,934 | 1,568 |
| `'women'` | 1,090 | 768 |
| `'Women'` | 890 | 178 |
| **Total** | **24,103** | **13,633** |

### Male tokens

| Token | Female model | Male model |
|-------|------------:|----------:|
| `' boys'` | 171 | 82 |
| `'boys'` | 40 | 72 |
| `' brother'` | 20 | 50 |
| `' dad'` | 268 | 300 |
| `' father'` | 120 | 143 |
| `' god'` | 10 | 138 |
| `'grandfather'` | 10 | 10 |
| `' grandfather'` | 10 | 20 |
| `' grandpa'` | 60 | 100 |
| `'He'` | 110 | 90 |
| `' he'` | 330 | 80 |
| `' hero'` | 100 | 181 |
| `'hero'` | 10 | 10 |
| `' him'` | 400 | 80 |
| `' himself'` | 193 | 440 |
| `'himself'` | 213 | 460 |
| `' Himself'` | 190 | 261 |
| `'His'` | 170 | 193 |
| `' his'` | 813 | 840 |
| `'his'` | 450 | 460 |
| `' His'` | 490 | 490 |
| `' husband'` | 510 | 582 |
| `'male'` | 622 | 1,542 |
| `'Male'` | 240 | 1,042 |
| `' male'` | 1,405 | 2,642 |
| `' man'` | 20 | 80 |
| `' masculine'` | 760 | 2,582 |
| `' master'` | 10 | 0 |
| `'master'` | 10 | 0 |
| `' men'` | 283 | 770 |
| `'men'` | 10 | 0 |
| `' nephew'` | 70 | 163 |
| `' sir'` | 10 | 20 |
| `' son'` | 268 | 300 |
| `' uncle'` | 10 | 10 |
| **Total** | **8,406** | **14,233** |

## Summary

| Metric | Standard (F model) | Standard (M model) | Direct (F model) | Direct (M model) |
|--------|-------------------:|-------------------:|-----------------:|-----------------:|
| Female tokens | 30,934 | 16,507 | 24,103 | 13,633 |
| Male tokens | 9,531 | 17,105 | 8,406 | 14,233 |
| F/M ratio | 3.25 | 0.97 | 2.87 | 0.96 |
