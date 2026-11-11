from fractions import Fraction
from itertools import product
from collections import Counter
from math import factorial
import datetime,json
from pathlib import Path
p=Path('/tmp/genai-bootstrap-draft')
d=tuple(Fraction(x) for x in (-1,0,1));n=len(d);copies=4;rows=tuple(x for x in d for _ in range(copies))
mean=lambda xs:sum(xs,Fraction())/len(xs)
mu=mean(d)
means=tuple(mean(draw) for draw in product(d,repeat=n))
cluster_counts=Counter(means);cluster_pmf={v:Fraction(c,len(means)) for v,c in cluster_counts.items()}
cluster_expectation=sum(v*prob for v,prob in cluster_pmf.items())
cluster_variance=sum((v-cluster_expectation)**2*prob for v,prob in cluster_pmf.items())
empirical_variance=mean(tuple((v-mu)**2 for v in d))
# Exactly sum all multinomial count classes for the erroneous 12-row IID resample.
# Each of the three observed values occurs in 4/12 rows, so its probability is 1/3.
N=len(rows);row_pmf=Counter();classes=[]
for negative in range(N+1):
 for zero in range(N-negative+1):
  positive=N-negative-zero
  ways=factorial(N)//(factorial(negative)*factorial(zero)*factorial(positive))
  probability=Fraction(ways,3**N);value=Fraction(positive-negative,N)
  row_pmf[value]+=probability
  classes.append({'negative':negative,'zero':zero,'positive':positive,'ordered_value_draws':ways,'probability':str(probability),'bootstrap_mean':str(value)})
row_expectation=sum(v*prob for v,prob in row_pmf.items())
row_variance=sum((v-row_expectation)**2*prob for v,prob in row_pmf.items())
# Independent convolution of the one-draw empirical PMF reproduces the same row distribution.
integer_sum_pmf={0:Fraction(1)}
for _ in range(N):
 next_pmf=Counter()
 for total,prob in integer_sum_pmf.items():
  for value in (-1,0,1):next_pmf[total+value]+=prob/Fraction(3)
 integer_sum_pmf=next_pmf
assert row_pmf=={Fraction(total,N):prob for total,prob in integer_sum_pmf.items()}
assert sum(cluster_pmf.values())==sum(row_pmf.values())==1
assert mu==cluster_expectation==row_expectation==0
assert empirical_variance==Fraction(2,3)
assert cluster_variance==empirical_variance/3==Fraction(2,9)
assert row_variance==empirical_variance/12==Fraction(1,18)
assert cluster_variance/row_variance==4
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arithmetic':'fractions.Fraction; exact rational arithmetic','prompt_means':[str(v) for v in d],'copies_per_prompt':copies,'rows':N,'observed_mean':str(mu),'empirical_prompt_variance_denominator_n':str(empirical_variance),'cluster_ordered_draws_enumerated':len(means),'cluster_bootstrap_mean':str(cluster_expectation),'cluster_bootstrap_variance':str(cluster_variance),'iid_row_multinomial_classes_enumerated':len(classes),'iid_row_distinct_value_sequences_represented':3**N,'iid_row_original_index_sequences_represented':N**N,'iid_row_bootstrap_mean':str(row_expectation),'iid_row_bootstrap_variance':str(row_variance),'variance_ratio_cluster_over_row':str(cluster_variance/row_variance),'standard_deviation_ratio_cluster_over_row':'2','independent_convolution_matches_multinomial_pmf':True,'cluster_pmf':[{'mean':str(v),'probability':str(prob),'ordered_draw_count':cluster_counts[v]} for v,prob in sorted(cluster_pmf.items())],'iid_row_pmf':[{'mean':str(v),'probability':str(prob)} for v,prob in sorted(row_pmf.items())]}
(p/'exact-fraction-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(p/'iid-row-count-classes.json').write_text(json.dumps(classes,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['cluster_pmf','iid_row_pmf']},ensure_ascii=False,indent=2))
