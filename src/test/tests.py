reactive_python_dag_builder_utils__.ask_for_ranges_to_compute(code= """










query = \"\"\"
select * from delta_old_job
full outer join delta_new_job
on delta_old_job.ad_id = delta_new_job.ad_id
;
\"\"\"

query = 4

import pandas as pd

credentials_redshift = get_credentials_redshift_marketingdb()
engine_redshift = connect_to_redshift(credentials_redshift)


df_delta_ven10 = pd.read_sql(query, engine_redshift)

df_delta_ven10 = pd.read_sql(query, engine_redshift)

some_more_query = query+1

# % [
    # Save locally as parquet:
df_delta_ven10.to_csv(\"df_delta_ven10.parquet\")
len(df_delta_ven10)

# % ]

a, b = 5, 6

print(a, b)



def set_diff_stats(**kwargs):
    assert len(kwargs) == 2, \'set_diff_stats() takes exactly 2 arguments\'
    (name_set1, name_set2), (set1, set2) = kwargs.keys(), kwargs.values()
    set1, set2 = set(set1), set(set2)
    print(f\'len({name_set1})={len(set1)}\', f\'len({name_set2})={len(set2)}\')
    print(f\'len({name_set1}.intersection({name_set2}))={len(set1.intersection(set2))}\')
    print(f\'len({name_set1}.difference({name_set2}))={len(set1.difference(set2))}\')
    print(f\'len({name_set2}.difference({name_set1}))={len(set2.difference(set1))}\')

    print(f\'Fraction of {name_set1} that is in {name_set2}:\', len(set1.intersection(set2)) / len(set1))
    print(f\'Fraction of {name_set2} that is in {name_set1}:\', len(set2.intersection(set1)) / len(set2))

    # print(f\'Elements that are in {name_set1} but not in {name_set2}:\', set1.difference(set2))
    # print(f\'Elements that are in {name_set2} but not in {name_set1}:\', set2.difference(set1))



set_diff_stats(df_delta_ven10=df_delta_ven10, user_ads_data_only_PUBLIC=user_ads_data_only_PUBLIC)


cdc = pd.DataFrame(\"uhhhhhhhhh\")

for c in cdc:
    print(c)

if 3<5:
    print(\'yes\')
else:
    a=75
    print(\'no\')



d = some_more_query

d +=2

d +=3

d +=4

d +=5

d +=6

""", current_line=88, get_upstream=True, get_downstream=False, stale_only=True)




# Get the 'lineno' property from every node in the NetworkX graph reactive_python_dag_builder_utils__.current_dag:
# [n['lineno'] for n in reactive_python_dag_builder_utils__.current_dag.nodes]  #  This says TypeError: 'int' object is not subscriptable
[n[1].get('lineno') for n in reactive_python_dag_builder_utils__.current_dag.nodes(data=True)]




code = """
a = 1                                                         
b = 2                                                         
a + b
"""
reactive_python_dag_builder_utils__.update_dag_and_get_ranges(
    code=code, current_line=2, get_upstream=False, get_downstream=True, include_code=True, stale_only=True)



code = """
a = 1                                                         
b = 2                                                         
a + b
"""
reactive_python_dag_builder_utils__.update_dag_and_get_ranges(
    code=code, current_line=3, get_upstream=True, get_downstream=False, include_code=True, stale_only=True)

