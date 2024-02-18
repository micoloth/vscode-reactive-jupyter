sample_python_code = """
import pandas as pd

credentials_redshift = get_credentials_redshift_marketingdb()
engine_redshift = connect_to_redshift(credentials_redshift)

query = \"""
select * from delta_old_job full outer join delta_new_job on delta_old_job.ad_id = delta_new_job.ad_id ; 
\"""

df_delta_ven10 = pd.read_sql(query, engine_redshift)
some_more_query = query+1

# Save locally as parquet:
# % [
df_delta_ven10.to_csv('df_delta_ven10.parquet')
len(df_delta_ven10)
a, b = 5, 6
print(a, b)
# % ]

def set_diff_stats(**kwargs):
    assert len(kwargs) == 2, 'set_diff_stats() takes exactly 2 arguments'
    (name_set1, name_set2), (set1, set2) = kwargs.keys(), kwargs.values()
    set1, set2 = set(set1), set(set2)
    print(f'len({name_set1})={len(set1)}', f'len({name_set2})={len(set2)}')
    print(f'len({name_set1}.intersection({name_set2}))={len(set1.intersection(set2))}')
    print(f'len({name_set1}.difference({name_set2}))={len(set1.difference(set2))}')
    print(f'len({name_set2}.difference({name_set1}))={len(set2.difference(set1))}')

    print(f'Fraction of {name_set1} that is in {name_set2}:', len(set1.intersection(set2)) / len(set1))
    print(f'Fraction of {name_set2} that is in {name_set1}:', len(set2.intersection(set1)) / len(set2))

    # print(f'Elements that are in {name_set1} but not in {name_set2}:', set1.difference(set2))
    # print(f'Elements that are in {name_set2} but not in {name_set1}:', set2.difference(set1))


set_diff_stats(df_delta_ven10=df_delta_ven10, user_ads_data_only_PUBLIC=user_ads_data_only_PUBLIC)

cdc = pd.DataFrame('uhhhhnnhhhhh')
for c in cdc:
    print(c)

if 3<5:
    print('yes')
else:
    a=75
    print('no')

d = some_more_query
d +=2; d +=3
d +=4
d +=5
d +=6
"""

# DONE (omg): a bug in nested classes
# TODO: solve the fact that in "for a in v", a is Store (which is Actually ok), BUT v is Load, so it will NOT count as output: SEE NEXT POINT
# TODO: introduce a whole Mutual dependent (equivalence) concept, where in (a=b; a+=1, a IS MUTED, - and in a=b.field; a+=1, TOO !!!)
# TODO: Make "if" dependent, i guess
# TODO: Merge Nodes functionality (for nodes on the same line, and also for Cells, especially with the # % [] functionality)


# THE PARSE FUNCTION:
# parse(source, filename='<unknown>', mode='exec', *, type_comments=False, feature_version=None)
# Same as:           compile(source, filename, mode, flags=PyCF_ONLY_AST) # and the other, defaults
# type_comments = True    feature_version=(3, 4) (mino, major)
# The filename argument should give the file from which the code was read; pass some recognizable value if it wasn’t read from a file ('<string>' is commonly used).
# The mode argument specifies what kind of code must be compiled; it can be 
#     'exec' if source consists of a sequence of statements, 
#     'eval' if it consists of a single expression, or 
#     'single' if it consists of a single interactive statement (in the latter case, expression statements that evaluate 
#           to something other than None will be printed).

# %load_ext autoreload
# %autoreload 2





[[6, 6, "outdated", "", "import boto3", "6-6: import boto3"], [55, 55, "outdated", "", "import pyarrow", "55-55: import pyarrow"], [59, 70, "outdated", "", "def get_filesystem(profile_name):\n    session = boto3.Session(profile_name=profile_name)  # )\n    credentials = session.get_credentials()  # Get credentials out of session:\n    # Read with credentials:\n    filesystem = pyarrow.fs.S3FileSystem(\n        access_key=credentials.access_key,\n        secret_key=credentials.secret_key,\n     …redentials:\n    filesystem = pyarrow.fs.S3FileSystem(\n        access_key=credentials.access_key,\n        secret_key=credentials.secret_key,\n        session_token=credentials.token,\n        region=\'eu-west-1\',\n        # role_arn=\'arn:aws:iam::939571286166:role/aws_iam_role-unicron_readonly_dev_access\'\n    )\n    return filesystem"], [73, 73, "outdated", "current", "filesystem_pro = get_filesystem(\"sbt-it-pro:power\")", "73-73: filesystem_pro = get_filesystem(\"sbt-it-pro:power\")"]]


        
def test_dag_builder():

    from reactive_python_engine import reactive_python_dag_builder_utils__, ReactivePythonDagBuilderUtils__
    import ast
    from dataclasses import dataclass
    from typing import List, Optional, Set

    dagnodes_to_dag = reactive_python_dag_builder_utils__.dagnodes_to_dag
    ast_to_dagnodes = reactive_python_dag_builder_utils__.ast_to_dagnodes
    draw_dag = reactive_python_dag_builder_utils__.draw_dag
    update_staleness_info_in_new_dag = reactive_python_dag_builder_utils__.update_staleness_info_in_new_dag


    # filename = "sample.py"  #"ast_sample.py"
    # filename = "src/reactive_python_engine.py"  #"ast_sample.py"
    filename = "/Users/michele.tasca/SCRIPTS/python_scripts.py"
    with open(filename, 'r') as file:
        code = file.read()
    # code = sample_python_code
    old_code = code
    new_code = code.replace("df_delta_ven10 = pd.read_sql(query, engine_redshift)", "df_delta_ven10 = pd.read_sql(query, engine_redshift)\n\ndf_delta_ven10 = pd.read_sql(df_delta_ven10, engine_redshift)")

    # old_dag = dagnodes_to_dag(ast_to_dagnodes(ast.parse(old_code, "", mode='exec', type_comments=True), old_code))
    # new_dag = dagnodes_to_dag(ast_to_dagnodes(ast.parse(new_code, "", mode='exec', type_comments=True), new_code))



    reactive_python_dag_builder_utils__ = ReactivePythonDagBuilderUtils__()
    reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code= """import pandas as p\nimport pandas as pd""", current_line=None, get_upstream=False, get_downstream=True, stale_only=False)
    
    reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=old_code)
    reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=old_code, current_line=10, stale_only=True)
    reactive_python_dag_builder_utils__.ask_for_ranges_to_compute(code=old_code, current_line=10, stale_only=True)
    reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=old_code, current_line=648)
    reactive_python_dag_builder_utils__.current_dag.nodes
    reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=new_code)

    reactive_python_dag_builder_utils__.current_dag.nodes[3]
    reactive_python_dag_builder_utils__.update_dag_and_get_ranges(current_line=11)


    # draw_dag(old_dag)
    # update_staleness_info_in_new_dag(old_dag, new_dag)
    # draw_dag(new_dag)

    from networkx import topological_sort, find_cycle

    # Get the text of each node in the new_dag that is stale:
    stale_nodes = [new_dag.nodes[node]['text'] for node in topological_sort(new_dag) if not new_dag.nodes[node]['stale']]
    # Write to a file:
    with open("stale_nodes.txt", 'w') as file:
        file.write("\n\n".join(stale_nodes))



# test_dag_builder()




# %load_ext autoreload
# %autoreload 2


from reactive_python_engine import reactive_python_dag_builder_utils__

# import "time":
import time
import ast


draw_dag = reactive_python_dag_builder_utils__.draw_dag
update_staleness_info_in_new_dag = reactive_python_dag_builder_utils__.update_staleness_info_in_new_dag
get_input_variables_for = reactive_python_dag_builder_utils__.get_input_variables_for
get_output_variables_for = reactive_python_dag_builder_utils__.get_output_variables_for
annotate = reactive_python_dag_builder_utils__.annotate


code = """
# % [
a = c+1; b = (c+1+
    1+3); d=a+b
z=55
# % ]
k=7**2
"""
# code = """
# # % [
# a = c+1
# b = (c+1+
#     1+3)
# d=a+b
# z=55
# # % ]
# k=7**2
# """

reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=code, include_code=True)



current_time = time.time()


# test
code = """
x = True, False, None
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == set()
assert outputs == {'x'}

# test
code = """
x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'x'}
assert outputs == set()

# test
code = """
x, y = v
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'v'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'y'}

# test
code = """
x = y = v
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'v'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'y'}


# test
code = """
x = True, False, None
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == set()
assert outputs == {'x'}



# test_no_nonlocal
code = """
def f(x):
    g = x+a
    return g
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}


# test_no_nonlocal
code = """
def f():
    x += a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
# Turns out it's CORRECT IN PYTHON that x doesnt work like this.. Neat!


# test_no_nonlocal
code = """
def f(x):
    def g():
        x += a
    return g
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}


# test_recursive_function:
code = """
def f(x):
    if x == 0:
        return a
    return x * f(x-1)
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}


# test_no_nonlocal
code = """
def f(x):
    def g():
        x += 2
    return g
"""
tree = ast.parse(code).body[0]

# test_no_nonlocal
code = """
def f(x):
    def g():
        x += 2
    return g

def m(y):
    k = f(y-1)
    return k+2
"""
tree = ast.parse(code) # NOTE no .body[0] !!

# test_basic_nonlocal
code = """
def f(x):
    def g():
        nonlocal x
        x += 2
    return g
"""
tree = ast.parse(code).body[0]
# assert annotate(tree).error_scope.variables == Variables()
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}

# test_nonlocal_not_found
code = """
x=1
def f():
    def g():
        nonlocal x
        x += 2
    return g
"""
tree = ast.parse(code)


# test_global_escapes_scope
code = """
def f():
    global v
    v[100] =2
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'v'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f', 'v'}

# test_global_escapes_scope
code = """
v[n] = c
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'v', 'n', 'c'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'v'}


# test_global_escapes_scope
code = """
v[n][m] = c
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'v', 'n', 'c', 'm'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'v'}



# test_global_escapes_scope
code = """
v[n][m[g]] = c[g][k]
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'g', 'k', 'm', 'n', 'v'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'v'}

# test_global_escapes_scope
code = """
v.heck[n].ops[m[g]].wow = c[g].yelp[k].gulp
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'g', 'k', 'm', 'n', 'v'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'v'}






# test_global_escapes_scope
code = """
def f(x):
    def g(y):
        def h():
            global x
            x += 2
    return g
x = 2
"""
tree = ast.parse(code)

# test_global_escapes_scope
code = """
def f(x):
    def g(y):
        def h():
            global x
            x += 2
    return g
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'f'}



# test_global_escapes_scope_even_without_declaration
code = """
def f(x):
    def g(y):
        def h():
            global x
            x += 2
    return g
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'f'}

# test_symbol_in_different_frame_from_parent
code = """
def f(x, y):
    def g(y):
        nonlocal x
        def x():
            y
"""
tree = ast.parse(code).body[0]


# test_basic_lambda
code = """
lambda x: x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == set()

# test_noarg_lambda
code = """
f = lambda: x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}

# test_multiarg_lambda
code = """
lambda x, y, *args: x if y else args
"""

# test_nested_lambdas
code = """
lambda x, y: lambda y, z: t + x + y + z
t = x = y = z = 0
"""

# test_default_params_in_parent  # Smart, this guy!
code = """
def f(x):
    lambda x=x: x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}

# test_default_params_in_parent  # Smart, this guy!
code = """
def f(x):
    lambda x=x: y
"""
tree = ast.parse(code).body[0]



# test
code = """
def f(n: int): 
    return n + 1
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'int'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}


# test
code = """
def g(n: int): 
    a = b+3
    return a + 1
"""
tree = ast.parse(code).body[0]

# test
code = """
def g(): 
    a = b+3
    return n
"""
tree = ast.parse(code).body[0]

# test
code = """
def h():
    x = True, False, None
    return 3
"""


# test
code = """
async def f(n: int): 
    return n + 1
"""

# test
code = """
async def g(n: int): 
    a = b+3
    return a + 1
"""

# test
code = """
async def g(): 
    a = b+3
    return n
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'b', 'n'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'g'}


# test
code = """
l = lambda x: x + 1
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'l'}

# test
code = """
m = lambda x: x + a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'m'}

# test
code = """
n = lambda x: a

"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'n'}



# test
code = """
yy = 4+5*( (k:=(a+b)) + (lambda x: 7+a+x)(c) )
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'b', 'c'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'k', 'yy'}

# test
code = """
yy = 4+5*( (k:=(a+b)) + (lambda k: 7+a+x)(c) )
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'b', 'c', 'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'k', 'yy'}


# test
code = """
yy += 4+5* (a,b[c])
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'b', 'yy', 'c'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'yy'}


# test
code = """
yy[c] += 4+5* (a,b)
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'b', 'yy', 'c'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'yy'}






######################### classes ############################


# Only the functions, deindented and with no annotations:
code = """
class X: pass
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'X'}


code = """
class X:
    x = 2
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'X'}


code = """
class X:
    x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'X'}


code = """
class X:
    x = 3
    x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}  # APPARENTLY this is right...
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'X'}


code = """
def f():
    x = 2
    class X:
        x, y
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'y'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}


code = """
class X:
    def f(): pass
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'X'}


code = """
class X:
    x = 2
    def f():
        return x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'X'}


code = """
def f(x):
    class X:
        x = 2
        def f():
            return x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}


code = """
class Y:
    class X:
        x = 2
        def f():
            return x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'Y'}


code = """
def f(x):
    class X:
        x = 2
        def f():
            global x
            x = 2
            return x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f', 'x'}


code = """
class X:
    def __init__(x):    
        x = 2
        def f(x=x):
            pass
    def g():
        return x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'X'}

code = """
class X:
    x = 2
    [x for t in x]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'X'}

# The only actually useful test:
code = """
class X:
    def __init__(self, x):    
        self.x = 2
    def g(self):
        return self.x + a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'X'}





# Only the functions, deindented and with no annotations:
code = """
def f():
    x = 2
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}


code = """
def f():
    _ = 2
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}

code = """
def f():
    x[0] = 2
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f', 'x'}

code = """
def f():
    x += 2
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f', 'x'}

code = """
def f():
    x, y = 1, 2
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}

code = """
def f():
    (x := 2)
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}


code = """
def f():
    for x in range(2):
        pass
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'range'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}

code = """
def f():
    for x in range(v):
        pass
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'range', 'v'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}


code = """
import os
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'os'}


code = """
import os, uhuh
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'os', 'uhuh'}

code = """
from os import system
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'system'}

code = """
from os import *
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == set('*')

code = """
os.filesss
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'os'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == set()




code = """
if g(b[k].value):
    y = x+4
    w.asset = f(x)
    z = w.asset.value
else:
    c
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'f', 'k', 'w', 'x', 'g', 'b'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'w', 'y', 'z'}


code = """
(a, b) = (1, c[d])
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'd'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'a', 'b'}

code = """
(a, b) = (a, b)
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'b'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'a', 'b'}




######################## TODO/ Wrong stuff:

# test
code = """
yy = 4+5*( (k:=(a+b)) + (lambda x: 7+a+k)(c) )
"""
# TODO: This is wrong.. :(  
# To be clear, the ONLY way to fix this is to collect the (out->in) "internal edges" AFTER the fact (collection), 
# determining if they are (out->in) or (in->out) >BASED ON THE POS IN THE EXPRESSION<, this is the whole point !!!)
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'b', 'c', 'k'} 
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'k', 'yy'}


# test
code = """
yy += 4+5*( kk:=a + (lambda k: 7+a+k)(c) )
"""
# TODO: This is wrong.. :(  
# To be clear, the ONLY way to fix this is to collect the (out->in) "internal edges" AFTER the fact (collection), 
# determining if they are (out->in) or (in->out) >BASED ON THE POS IN THE EXPRESSION<, this is the whole point !!!)
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'c', 'yy'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'yy', 'kk'}


# code = """
# for k, d[k] in range(v):
#     pass
# """
# print(ast.dump(tree, indent=4))
# tree = ast.parse(code).body[0]
# inputs, errors = get_input_variables_for(annotate(tree)); inputs
# assert inputs == {'range', 'v', 'd'}
# outputs, errors = get_output_variables_for(annotate(tree)); outputs
# assert outputs == {'f'}
# TODO this is wrong !!!

# code = """
# for x in range(1,3):
#     y = x+4
#     w.asset = f(x)
#     z = w.asset.value
# """
# tree = ast.parse(code).body[0]
# inputs, errors = get_input_variables_for(annotate(tree)); inputs
# assert inputs == {'f', 'range', 'w'}
# outputs, errors = get_output_variables_for(annotate(tree)); outputs
# assert outputs == {'w', 'x', 'y', 'z'}




# Check how long it took:
print(time.time() - current_time)


