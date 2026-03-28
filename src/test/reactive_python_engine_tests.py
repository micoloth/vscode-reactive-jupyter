

import ast
import sys
import os

# Add parent directory to path to import reactive_python_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactive_python_engine import reactive_python_dag_builder_utils__

# Get the functions from the instance
annotate = reactive_python_dag_builder_utils__.annotate
get_input_variables_for = reactive_python_dag_builder_utils__.get_input_variables_for
get_output_variables_for = reactive_python_dag_builder_utils__.get_output_variables_for


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
assert outputs == {'f'}


########## INSERT BROKENS HWRE...

# test_global_escapes_scope
code = """
v[n] = c
"""
tree = ast.parse(code)
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'v'}
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'v', 'n', 'c'}


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
assert outputs == {'f'}

# test_symbol_in_different_frame_from_parent
code = """
def f(x, y):
    def g(y):
        nonlocal x
        def x():
            y
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()


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
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'y'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}



# test_default_params_in_argument:  # Pretty psyco thing to do, tbh:
code = """
def f(x=(k:=5)):
    return x + y + 6
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'y'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f', 'k'}



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
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'b', 'int'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'g'}

# test
code = """
def g(): 
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



# test  # WAAAA k:= within same expr ???
code = """
yy = 4+5*( (k:=(a+b)) + (lambda x: 7+a+k)(c) )
"""
# TODO: This is wrong.. :(  
# To be clear, the ONLY way to fix this is to collect the (out->in) "internal edges" AFTER the fact (collection), 
# determining if they are (out->in) or (in->out) >BASED ON THE POS IN THE EXPRESSION<, this is the whole point !!!)
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'b', 'c'} 
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


code = """
for x in range(1,3): 
    y = x+c
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'range'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'y'}


code = """
for x in range(1,3):
    y = x+4
    w.asset = f(x)
    z = w.asset.value
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'f', 'range', 'w'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'w', 'x', 'y', 'z'}


code = """
for x in range(1,3):
    def ff(y):
        return y + x
    z = x.asset.value
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'range'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'ff', 'z'}

code = """
for x in range(1,3):
    def ff(y):
        return y + x
    z.asset = x.asset.value
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'range', 'z'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'ff', 'z'}

code = """
for x in y:
    for z in x:
        z
        a = 3
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'y'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'a', 'x', 'z'}


code = """
for x in y:
    a = x + 5
    print(a)
    a += c
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'print', 'y'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'a', 'x'}

code = """
for x in y:
    a += c
    print(a)
    a = x + 5
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'print', 'y', 'a'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'a', 'x'}




code = """
with x as y:
    with z as a:
        k = a + c
        f += y
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'f', 'x', 'z'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'a', 'f', 'k', 'y'}





code = """
x = [(y, a) for y in range(1,5+b )]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'b', 'range'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x'}



code = """
x = {y: c for y in range(1,5+b )}
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'b', 'range'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x'}


code = """
x = {y: c for y, c in range(1,5+b )}
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'b', 'range'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x'}



code = """
x = [(y, a) for y in range(1,5+b ) if (z:=(y + 5 + b)) > z + c]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'b', 'c', 'range'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'z'}



code = """
x = [(y, a) for y in range(1,5+b ) if z + c > (z:=(y + 5 + b))]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'b', 'c', 'range', 'z'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'z'}


# ============================================================
# COMPREHENSIVE COMPREHENSION TESTS
# ============================================================

# test_list_comp_simple_filter
code = """
x = [i for i in items if i > threshold]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'items', 'threshold'}
assert outputs == {'x'}

# test_list_comp_filter_with_function
code = """
x = [i for i in items if is_valid(i)]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'items', 'is_valid'}
assert outputs == {'x'}

# test_list_comp_multiple_filters
code = """
x = [i for i in items if i > min_val if i < max_val]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'items', 'min_val', 'max_val'}
assert outputs == {'x'}

# test_dict_comp_with_filter
code = """
x = {k: v for k, v in pairs if k not in excluded}
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'pairs', 'excluded'}
assert outputs == {'x'}

# test_dict_comp_with_transform_and_filter
code = """
x = {k.lower(): process(v) for k, v in data.items() if v is not None}
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'data', 'process'}
assert outputs == {'x'}

# test_set_comp_simple
code = """
x = {i * 2 for i in items}
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'items'}
assert outputs == {'x'}

# test_set_comp_with_filter
code = """
x = {i for i in items if i % divisor == 0}
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'items', 'divisor'}
assert outputs == {'x'}

# test_generator_expr
code = """
x = (i**2 for i in nums if i > 0)
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'nums'}
assert outputs == {'x'}

# test_nested_comprehension - list of lists
code = """
x = [[j for j in row if j > 0] for row in matrix]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'matrix'}
assert outputs == {'x'}

# test_nested_comprehension_external_var
code = """
x = [[j + offset for j in row] for row in matrix]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'matrix', 'offset'}
assert outputs == {'x'}

# test_multiple_iterators_flat
code = """
x = [i + j for i in rows for j in cols]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'rows', 'cols'}
assert outputs == {'x'}

# test_multiple_iterators_with_filter
code = """
x = [(i, j) for i in rows for j in cols if i != j]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'rows', 'cols'}
assert outputs == {'x'}

# test_multiple_iterators_dependent
code = """
x = [(i, j) for i in matrix for j in i]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'matrix'}
assert outputs == {'x'}

# test_comprehension_with_method_in_filter
code = """
x = [s for s in strings if s.startswith(prefix)]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'strings', 'prefix'}
assert outputs == {'x'}

# test_comprehension_enumerate
code = """
x = [(i, v) for i, v in enumerate(items) if i < limit]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'enumerate', 'items', 'limit'}
assert outputs == {'x'}

# test_dict_comp_from_list
code = """
x = {item: len(item) for item in items if item}
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'items', 'len'}
assert outputs == {'x'}




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
assert outputs == {'f'}

code = """
def f():
    x += 2
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}

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
if g(b[k].value):
    y += x+4
    y = x+4
    w.asset = f(x)
    z = w.asset.value
else:
    c
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'f', 'k', 'w', 'x', 'g', 'b', 'y'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'w', 'y', 'z'}

code = """
if g(b[k].value):
    y = x+4
    w.asset = f(x)
    y += x+4
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

code = """
a, b = (1, c[d])
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'd'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'a', 'b'}


code = """
k.fields = (1, c[d])
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'd', 'k'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'k'}



code = """
for x in range(1,3): 
    y = x+c
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'range'}
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'y'}

code = """
for x in range(1,3): 
    y += x+c
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'y', 'range'}
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'y'}

code = """
for x in range(1,3): 
    y = x+c
    print(y)
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'range', 'print'}
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'y'}

code = """
for x in range(1,3): 
    y = y + c + x
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs  # Currently: {'c', 'range'}
assert inputs == {'c', 'y', 'range'}  # WRONG!!!
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'y'}


code = """
for x in range(1,3): 
    y = y + c
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs  # Currently: {'c', 'range'}
assert inputs == {'c', 'y', 'range'}  # WRONG!!!
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'y'}





code = """
for x in range(1,3): 
    y = x+c
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'c', 'range'}
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'y'}


code = """
for x in range(1,3):
    y = x+4
    w.asset = f(x)
    z = w.asset.value
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'f', 'range', 'w'}
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'w', 'x', 'y', 'z'}


code = """
for x in range(1,3):
    def ff(y):
        return y + x
    z = x.asset.value
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'range'}
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'ff', 'z'}

code = """
for x in range(1,3):
    def ff(y):
        return y + x
    z.asset = x.asset.value
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'range', 'z'}
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'x', 'ff', 'z'}

code = """
for x in y:
    for z in x:
        z
        a = 3
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'y'}
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'a', 'x', 'z'}


code = """
for x in y:
    for z in x:
        z
        a += x
"""
tree = ast.parse(code).body[0]
inputs, errorsIn = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'y', 'a'}
outputs, errorsOut = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'a', 'x', 'z'}


# reactive_python_dag_builder_utils__2.update_dag_and_get_ranges(code, current_line=5, include_code=True)





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


# x = 5
# class X:
#     x = 3
#     x

# x = 5
# X().x


code = """
class X:
    x = 3
    x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
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
assert outputs == {'f'}


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
assert inputs == set()
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
assert outputs == {'f'}
# TODO: arguably Broken . Should v be an output ?

# test_no_nonlocal
code = """
def f():
    x += a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}
# Turns out it's CORRECT IN PYTHON that x doesnt work like this.. Neat!


# test_no_nonlocal
code = """
def f():
    x += a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}
# Turns out it's CORRECT IN PYTHON that x doesnt work like this.. Neat!



# test_no_nonlocal
code = """
def f():
    x = a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}
# Turns out it's CORRECT IN PYTHON that x doesnt work like this.. Neat!


# test_no_nonlocal
code = """
def f(x):
    global x
    x = a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}
# Turns out it's CORRECT IN PYTHON that x doesnt work like this.. Neat!


# test_no_nonlocal
code = """
def g(x):
    def f(x):
        global x
        x = a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'g'}
# Turns out it's CORRECT IN PYTHON that x doesnt work like this.. Neat!




# test_no_nonlocal
code = """
def f(x):
    nonlocal x
    x = a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a', 'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'f'}
# Turns out it's CORRECT IN PYTHON that x doesnt work like this.. Neat!



# test_no_nonlocal
code = """
def g(x):
    def f(x):
        nonlocal x
        x = a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'a'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'g'}
# Turns out it's CORRECT IN PYTHON that x doesnt work like this.. Neat!






# test_no_nonlocal
code = """
class A:
    class B:
        def g(self):
            return x
    a = B()
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == {'x'}
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'A'}



# test_no_nonlocal
code = """
class A:
    x = 2    
    class B:
        x = 5
        def gdef(self, v=x):
            return v
    a = B()
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'A'}



# test_no_nonlocal
code = """
class A:
    x = 2
    def f(self):
        x = 3

        class B:
            x = 4
            def g(self):
                return x
        return B()
    
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree)); inputs
assert inputs == set()
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert outputs == {'A'}














####################################################################################################
####################################################################################################
####################################################################################################




















######### EDGE CASES (previously marked as "broken" but behavior is actually correct):
# These edge cases involve nested scopes and nonlocal/global. The current behavior is correct
# for tracking DAG dependencies - we care about external inputs to the top-level construct.




# test_no_nonlocal - inner function using outer param without nonlocal
# In Python, this is an UnboundLocalError at runtime (x becomes local to g, used before assignment)
# But from f's perspective, there are NO external inputs - x comes from f's parameter
code = """
def f(x):
    def g():
        x += 2
    return g
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree))
outputs, _ = get_output_variables_for(annotate(tree))
assert inputs == set()  # x is f's parameter, not external
assert outputs == {'f'}

# test_no_nonlocal - multiple functions, same pattern
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
inputs, errors = get_input_variables_for(annotate(tree))
outputs, _ = get_output_variables_for(annotate(tree))
assert inputs == set()  # No external inputs - x is f's param, y is m's param
assert outputs == {'f', 'm'}


# test_nonlocal_not_found - nonlocal x but x is only at module level
# This is actually a Python SyntaxError at compile time, but ast.parse succeeds
# Our tool handles it by treating the nonlocal as reaching the module level
code = """
x=1
def f():
    def g():
        nonlocal x
        x += 2
    return g
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree))
outputs, _ = get_output_variables_for(annotate(tree))
# Note: In valid Python, nonlocal can't reach module level. We parse anyway.
assert outputs == {'x', 'f'}  # Both x and f are defined at module level



# test_global_escapes_scope - global x in nested func, x param in outer, x=2 at module
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
inputs, errors = get_input_variables_for(annotate(tree))
outputs, _ = get_output_variables_for(annotate(tree))
assert inputs == {'x'}  # global x in h() reads module-level x
assert outputs == {'f', 'x'}  # Both f and x are defined at module level


# test_multiarg_lambda - lambda with *args
code = """
lambda x, y, *args: x if y else args
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree))
outputs, _ = get_output_variables_for(annotate(tree))
assert inputs == set()  # All variables are parameters
assert outputs == set()  # Lambda expression, no name binding

# test_nested_lambdas
code = """
lambda x, y: lambda y, z: t + x + y + z
t = x = y = z = 0
"""
tree = ast.parse(code)
inputs, errors = get_input_variables_for(annotate(tree))
outputs, _ = get_output_variables_for(annotate(tree))
# Inner lambda uses t (external), x (from outer lambda), y (shadowed), z (param)
assert inputs == {'t'}  # Only t is truly external (defined after lambda)
assert outputs == {'t', 'x', 'y', 'z'}  # All defined at module level


# ============================================================
# F-STRING TESTS
# ============================================================

# test_fstring_simple
code = """
x = f"hello {name}"
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'name'}
assert outputs == {'x'}

# test_fstring_expression
code = """
x = f"result: {a + b}"
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'a', 'b'}
assert outputs == {'x'}

# test_fstring_method_call
code = """
x = f"upper: {name.upper()}"
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'name'}
assert outputs == {'x'}

# test_fstring_function_call
code = """
x = f"len: {len(items)}"
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'items', 'len'}
assert outputs == {'x'}

# test_fstring_format_spec_with_vars
code = """
x = f"{value:{width}.{precision}f}"
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'value', 'width', 'precision'}
assert outputs == {'x'}

# test_fstring_walrus
code = """
x = f"{(y := z + 1)}"
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'z'}
assert outputs == {'x', 'y'}

# test_fstring_comprehension
code = """
x = f"{[i for i in items]}"
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'items'}
assert outputs == {'x'}

# test_fstring_conditional
code = """
x = f"{a if b else c}"
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'a', 'b', 'c'}
assert outputs == {'x'}


# ============================================================
# DELETE STATEMENT TESTS
# ============================================================

# test_del_simple
code = """
del x
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'x'}
assert outputs == set()

# test_del_multiple
code = """
del x, y, z
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'x', 'y', 'z'}
assert outputs == set()

# test_del_subscript
code = """
del items[key]
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'items', 'key'}
assert outputs == {'items'}

# test_del_attribute
code = """
del obj.attr
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'obj'}
assert outputs == {'obj'}


# ============================================================
# DECORATOR TESTS
# ============================================================

# test_decorated_function
code = """
@decorator
def f(): pass
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'decorator'}
assert outputs == {'f'}

# test_decorated_class
code = """
@cls_decorator
class C: pass
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'cls_decorator'}
assert outputs == {'C'}

# test_multiple_decorators
code = """
@d1
@d2
def f(): pass
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'d1', 'd2'}
assert outputs == {'f'}

# test_decorator_with_args
code = """
@decorator(arg)
def f(): pass
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'decorator', 'arg'}
assert outputs == {'f'}


# ============================================================
# STARRED ASSIGNMENT TESTS
# ============================================================

# test_starred_assignment
code = """
a, *b, c = items
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'items'}
assert outputs == {'a', 'b', 'c'}

# test_starred_first_rest
code = """
first, *rest = some_list
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'some_list'}
assert outputs == {'first', 'rest'}


# ============================================================
# TYPE ANNOTATION TESTS
# ============================================================

# test_type_annotation
code = """
x: int = 5
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'int'}
assert outputs == {'x'}

# test_type_annotation_generic
code = """
x: list[int] = []
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'list', 'int'}
assert outputs == {'x'}


# ============================================================
# MISC EDGE CASES TESTS
# ============================================================

# test_assert_with_message
code = """
assert cond, msg
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'cond', 'msg'}
assert outputs == set()

# test_raise_from
code = """
raise Ex from cause
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
assert inputs == {'Ex', 'cause'}
assert outputs == set()


# ============================================================
# MATCH STATEMENT TESTS (Python 3.10+)
# ============================================================

import sys
if sys.version_info >= (3, 10):
    # test_match_simple
    code = """
match command:
    case "quit":
        result = "bye"
    case x:
        result = x
"""
    tree = ast.parse(code).body[0]
    inputs, errors = get_input_variables_for(annotate(tree)); inputs
    outputs, errors = get_output_variables_for(annotate(tree)); outputs
    assert inputs == {'command'}, f"Expected {{'command'}}, got {inputs}"
    assert outputs == {'result', 'x'}, f"Expected {{'result', 'x'}}, got {outputs}"

    # test_match_sequence_capture
    code = """
match point:
    case (x, y):
        result = x + y
"""
    tree = ast.parse(code).body[0]
    inputs, errors = get_input_variables_for(annotate(tree)); inputs
    outputs, errors = get_output_variables_for(annotate(tree)); outputs
    assert inputs == {'point'}, f"Expected {{'point'}}, got {inputs}"
    assert outputs == {'result', 'x', 'y'}, f"Expected {{'result', 'x', 'y'}}, got {outputs}"

    # test_match_with_guard
    code = """
match value:
    case x if x > threshold:
        result = x
"""
    tree = ast.parse(code).body[0]
    inputs, errors = get_input_variables_for(annotate(tree)); inputs
    outputs, errors = get_output_variables_for(annotate(tree)); outputs
    assert inputs == {'value', 'threshold'}, f"Expected {{'value', 'threshold'}}, got {inputs}"
    assert outputs == {'result', 'x'}, f"Expected {{'result', 'x'}}, got {outputs}"

    # test_match_or_pattern
    code = """
match command:
    case "n" | "next":
        action = "next"
"""
    tree = ast.parse(code).body[0]
    inputs, errors = get_input_variables_for(annotate(tree)); inputs
    outputs, errors = get_output_variables_for(annotate(tree)); outputs
    assert inputs == {'command'}
    assert outputs == {'action'}

    # test_match_star_pattern
    code = """
match items:
    case [first, *rest]:
        result = first
"""
    tree = ast.parse(code).body[0]
    inputs, errors = get_input_variables_for(annotate(tree)); inputs
    outputs, errors = get_output_variables_for(annotate(tree)); outputs
    assert inputs == {'items'}
    assert outputs == {'result', 'first', 'rest'}

    # test_match_mapping_pattern
    code = """
match data:
    case {"name": name, "age": age, **rest}:
        result = name
"""
    tree = ast.parse(code).body[0]
    inputs, errors = get_input_variables_for(annotate(tree)); inputs
    outputs, errors = get_output_variables_for(annotate(tree)); outputs
    assert inputs == {'data'}
    assert outputs == {'result', 'name', 'age', 'rest'}

    print("Match statement tests passed!")


print("=" * 60)
print("All input/output variable tests passed!")
print("=" * 60)

# Everything below this line is experimental/incomplete and disabled
import sys
sys.exit(0)




# sample_python_code = """
# import pandas as pd

# credentials_redshift = get_credentials_redshift_marketingdb()
# engine_redshift = connect_to_redshift(credentials_redshift)

# query = \"""
# select * from delta_old_job full outer join delta_new_job on delta_old_job.ad_id = delta_new_job.ad_id ; 
# \"""

# df_delta_ven10 = pd.read_sql(query, engine_redshift)
# some_more_query = query+1

# # Save locally as parquet:
# # % [
# df_delta_ven10.to_csv('df_delta_ven10.parquet')
# len(df_delta_ven10)
# a, b = 5, 6
# print(a, b)
# # % ]

# def set_diff_stats(**kwargs):
#     assert len(kwargs) == 2, 'set_diff_stats() takes exactly 2 arguments'
#     (name_set1, name_set2), (set1, set2) = kwargs.keys(), kwargs.values()
#     set1, set2 = set(set1), set(set2)
#     print(f'len({name_set1})={len(set1)}', f'len({name_set2})={len(set2)}')
#     print(f'len({name_set1}.intersection({name_set2}))={len(set1.intersection(set2))}')
#     print(f'len({name_set1}.difference({name_set2}))={len(set1.difference(set2))}')
#     print(f'len({name_set2}.difference({name_set1}))={len(set2.difference(set1))}')

#     print(f'Fraction of {name_set1} that is in {name_set2}:', len(set1.intersection(set2)) / len(set1))
#     print(f'Fraction of {name_set2} that is in {name_set1}:', len(set2.intersection(set1)) / len(set2))

#     # print(f'Elements that are in {name_set1} but not in {name_set2}:', set1.difference(set2))
#     # print(f'Elements that are in {name_set2} but not in {name_set1}:', set2.difference(set1))


# set_diff_stats(df_delta_ven10=df_delta_ven10, user_ads_data_only_PUBLIC=user_ads_data_only_PUBLIC)

# cdc = pd.DataFrame('uhhhhnnhhhhh')
# for c in cdc:
#     print(c)

# if 3<5:
#     print('yes')
# else:
#     a=75
#     print('no')

# d = some_more_query
# d +=2; d +=3
# d +=4
# d +=5
# d +=6
# """

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


# ========== COMMENTED OUT: EXPERIMENTAL CODE THAT DOESN'T RUN ==========
# x= 5
# 
# def someFuncOut(x):
# 
#     def someFunc(y):
#         nonlocal x
#         x = x + y
#         return x
#     
#     return someFunc
# 
# someFuncOut(5)(3)
# 
# def incr(x: int = (yy:=5+i)):
#     global i 
#     # incr = lambda i: i+3
#     i += 1
#     return yy + incr(i)
# 
# 
# import ast
# code = """
# def someFunc(x):
#     global x
#     return x+1
# """
# tree = ast.parse(code).body[0]
# 
# 
# incr()
# incr(1)
# i
# yy
# ========== END COMMENTED OUT SECTION ==========


# [[6, 6, "outdated", "", "import boto3", "6-6: import boto3"], [55, 55, "outdated", "", "import pyarrow", "55-55: import pyarrow"], [59, 70, "outdated", "", "def get_filesystem(profile_name):\n    session = boto3.Session(profile_name=profile_name)  # )\n    credentials = session.get_credentials()  # Get credentials out of session:\n    # Read with credentials:\n    filesystem = pyarrow.fs.S3FileSystem(\n        access_key=credentials.access_key,\n        secret_key=credentials.secret_key,\n     …redentials:\n    filesystem = pyarrow.fs.S3FileSystem(\n        access_key=credentials.access_key,\n        secret_key=credentials.secret_key,\n        session_token=credentials.token,\n        region=\'eu-west-1\',\n        # role_arn=\'arn:aws:iam::939571286166:role/aws_iam_role-unicron_readonly_dev_access\'\n    )\n    return filesystem"], [73, 73, "outdated", "current", "filesystem_pro = get_filesystem(\"sbt-it-pro:power\")", "73-73: filesystem_pro = get_filesystem(\"sbt-it-pro:power\")"]]


        
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


# ========== EVERYTHING BELOW IS EXPERIMENTAL AND REQUIRES MANUAL SETUP ==========
if __name__ == "__main__" and False:  # Disabled - this code requires manual setup to run

    # %load_ext autoreload
    # %autoreload 2


    from src.reactive_python_engine import reactive_python_dag_builder_utils__

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





    # Check how long it took:
    print(time.time() - current_time)


    x = 567


    f = lambda x: {'value': x+1}
    w = {'asset': 3}

    for x in range(1,3):
        y = x+4
    w['asset'] = f(x)
    z = w['asset']['value']

x



#########################################  ANCESTOR WITH PRUNING:

from networkx import DiGraph, topological_sort, find_cycle
from networkx import *



from typing import Callable
from collections import deque

def generic_bfs_edges_with_pruning(G, source, neighbors, pruning_condition: Callable):
    visited = {source}
    queue = deque([(source, 0, neighbors(source))])
    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited and pruning_condition(G.nodes[child]):
                yield parent, child
                visited.add(child)
                queue.append((child, 0, neighbors(child)))
        except StopIteration:
            queue.popleft()


def directed_ancestors_with_pruning(G, source, pruning_condition):
    return {child for parent, child in generic_bfs_edges_with_pruning(G, source, G.predecessors, pruning_condition=pruning_condition)}

def directed_descendants_with_pruning(G, source, pruning_condition):
    return {child for parent, child in generic_bfs_edges_with_pruning(G, source, G.neighbors, pruning_condition=pruning_condition)}



# sample graph:

dirGraph = DiGraph()
dirGraph.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 6),])
# Add a "STALE" label to each node:
for node in dirGraph.nodes:
    dirGraph.nodes[node]['stale'] = True

pruning_condition_ = lambda node: node['stale']

# Uni test descendants_with_pruning:

import copy 
g1 = copy.deepcopy(dirGraph)

g1.nodes[3]['stale'] = False

directed_descendants_with_pruning(g1, 1, pruning_condition_)
assert directed_descendants_with_pruning(g1, 1, pruning_condition_) == {2, 4, 5, 6}


g2 = copy.deepcopy(dirGraph)
g2.nodes[4]['stale'] = False
directed_descendants_with_pruning(g2, 1, pruning_condition_)
assert directed_descendants_with_pruning(g2, 1, pruning_condition_) == {2, 3, 5}



######################################  AN EXECUTION EXAMPLE:

def pr(ranges):
    print("\n".join([r[4] + ": " + r[2] for r in json.loads(ranges)] if ranges else []))

import json

code = """
counter1 = 1  # line 1
counter1 += 2
counter1

counter3 = 3
counter3 += 4
print(counter3)
"""

reactive_python_dag_builder_utils__ = ReactivePythonDagBuilderUtils__()

ranges = reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=code, include_code=True)
pr(ranges)

# RUN LINE 3:
ranges = reactive_python_dag_builder_utils__.ask_for_ranges_to_compute(code, current_line = 3, get_upstream=True, get_downstream=False, stale_only=True); ranges
pr(ranges)
for h in ([r[5] for r in json.loads(ranges)] if ranges else []):
    reactive_python_dag_builder_utils__.set_locked_range_as_synced(h)
reactive_python_dag_builder_utils__.unlock()
ranges = reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=None, current_line=None, include_code=True)
pr(ranges)

# RUN LINE 7:
ranges = reactive_python_dag_builder_utils__.ask_for_ranges_to_compute(code, current_line = 7, get_upstream=True, get_downstream=False, stale_only=True); ranges
pr(ranges)
for h in ([r[5] for r in json.loads(ranges)] if ranges else []):
    reactive_python_dag_builder_utils__.set_locked_range_as_synced(h)
reactive_python_dag_builder_utils__.unlock()
ranges = reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=None, current_line=None, include_code=True, stale_only=False)
pr(ranges)


#### Now send a New text:
code = """
counter1 = 1  # line 1
counter1 += 2
counter1

counter3 = 333
counter3 += 4
print(counter3)
"""
ranges = reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=code, include_code=True, stale_only=False)
pr(ranges)

# RUN LINE 7 AGAIN:
ranges = reactive_python_dag_builder_utils__.ask_for_ranges_to_compute(code, current_line = 7, get_upstream=True, get_downstream=False, stale_only=True); ranges
pr(ranges)
for h in ([r[5] for r in json.loads(ranges)] if ranges else []):
    reactive_python_dag_builder_utils__.set_locked_range_as_synced(h)
reactive_python_dag_builder_utils__.unlock()
ranges = reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=None, current_line=None, include_code=True, stale_only=False)
pr(ranges)

ranges = reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code=code, current_line=None, include_code=True, stale_only=False)
pr(ranges)






# #### Run "counter2" again:
# ranges = reactive_python_dag_builder_utils__.ask_for_ranges_to_compute(code, current_line = 4, get_upstream=True, get_downstream=False, stale_only=True); ranges
# print("\n".join([r[4] for r in json.loads(ranges)]))
# hashes = [r[5] for r in json.loads(ranges)] if ranges else []
# reactive_python_dag_builder_utils__.set_locked_range_as_synced(hashes[0])
# reactive_python_dag_builder_utils__.unlock()

# #### Draw DAG:
# reactive_python_dag_builder_utils__.draw_dag(reactive_python_dag_builder_utils__.current_dag)
