

import ast
import json
import hashlib
import itertools
import functools
import dataclasses
from collections import abc, deque
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Callable


from networkx import DiGraph, topological_sort, has_path

# Furthermore we need these modified methods, which are not in the library:
def generic_bfs_edges_with_pruning(G, source, neighbors, pruning_condition: Callable):
    visited = {source}
    queue = deque([(source, neighbors(source))])
    while queue:
        parent, children = queue[0]
        try:
            child = next(children)
            if child not in visited and pruning_condition(G.nodes[child]):
                yield parent, child
                visited.add(child)
                queue.append((child, neighbors(child)))
        except StopIteration:
            queue.popleft()

def directed_ancestors_with_pruning(G, source, pruning_condition):
    return {child for parent, child in generic_bfs_edges_with_pruning(G, source, G.predecessors, pruning_condition=pruning_condition)}

def directed_descendants_with_pruning(G, source, pruning_condition):
    return {child for parent, child in generic_bfs_edges_with_pruning(G, source, G.neighbors, pruning_condition=pruning_condition)}



@dataclasses.dataclass
class AstNodeData:
    node_name: str
    temp_scope: "TempScope"
    is_input: Optional[bool]
    is_output: Optional[bool]
    is_assignment: Optional[bool]


# | FunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list, expr? returns, string? type_comment, type_param* type_params)
# | AsyncFunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list, expr? returns, string? type_comment, type_param* type_params)
# | ClassDef(identifier name, expr* bases, keyword* keywords, stmt* body, expr* decorator_list, type_param* type_params)
# | Assign(expr* targets, expr value, string? type_comment)
# | AugAssign(expr target, operator op, expr value)
# | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
# | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
# | While(expr test, stmt* body, stmt* orelse)
# | If(expr test, stmt* body, stmt* orelse)
# | With(withitem* items, stmt* body, string? type_comment)
# | AsyncWith(withitem* items, stmt* body, string? type_comment)
# | Match(expr subject, match_case* cases)
# | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
# | TryStar(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
# | Global(identifier* names)
# | Nonlocal(identifier* names)

# !! TypeAlias(expr name, type_param* type_params, expr value)
# !! AnnAssign(expr target, expr annotation, expr? value, int simple)
# !! NamedExpr(expr target, expr value)
# ^^ Alla fine, controlla se serve aggiungerli !!
    
# k Assert(expr test, expr? msg)
# k Return(expr? value)
# k Delete(expr* targets)
# k Raise(expr? exc, expr? cause)
# k Expr(expr value)
# k Pass | Break | Continue
# k Import(alias* names)
# k ImportFrom(identifier? module, alias* names, int? level)

# | Lambda(arguments args, expr body)
# | ListComp(expr elt, comprehension* generators)
# | SetComp(expr elt, comprehension* generators)
# | DictComp(expr key, expr value, comprehension* generators)
# | GeneratorExp(expr elt, comprehension* generators)
# | Attribute(expr value, identifier attr, expr_context ctx)
# | Subscript(expr value, expr slice, expr_context ctx)
# | Name(identifier id, expr_context ctx)

# k BoolOp(boolop op, expr* values)
# k BinOp(expr left, operator op, expr right)
# k UnaryOp(unaryop op, expr operand)
# k IfExp(expr test, expr body, expr orelse)
# k Dict(expr* keys, expr* values)
# k Set(expr* elts)
# k Await(expr value)
# k Yield(expr? value)
# k YieldFrom(expr value)
# k Compare(expr left, cmpop* ops, expr* comparators)
# k Call(expr func, expr* args, keyword* keywords)
# k FormattedValue(expr value, int conversion, expr? format_spec)
# k JoinedStr(expr* values)
# k Constant(constant value, string? kind)
# k Starred(expr value, expr_context ctx)
# k List(expr* elts, expr_context ctx)
# k Tuple(expr* elts, expr_context ctx)
# k Slice(expr? lower, expr? upper, expr? step)

####################################################################################################
######################### INTERMEDIATE DATACLASSES: Variables, TempScope, and Scope ##########################################
####################################################################################################

@dataclasses.dataclass
class ExposedVariables():
    input_variables: Set[str] = dataclasses.field(default_factory=set)
    output_variables: Set[str] = dataclasses.field(default_factory=set)
    nonlocal_variables: Set[str] = dataclasses.field(default_factory=set)
    global_variables: Set[str] = dataclasses.field(default_factory=set)
    introduced_variables: Set[str] = dataclasses.field(default_factory=set) # These are the params in a function, or the target n a For or a With or exception, or even the variables in a class
    inputs_variables_in_function_in_class: Set[str] = dataclasses.field(default_factory=set)  # These are the variables that are used by the body of a function that is inside a class
    # it's not empty only when you are visiting one of these things.

    # # Make this hashable, using all the fields:
    # def __hash__(self):
    #     return hash((self.node, self.is_input, self.is_output))


def h_merge(*exposed_variables: ExposedVariables):
    """Merge horizontally, or in Parallel"""
    result = ExposedVariables()
    for ev in exposed_variables:
        result.input_variables |= ev.input_variables
        result.output_variables |= ev.output_variables
        result.nonlocal_variables |= ev.nonlocal_variables
        result.global_variables |= ev.global_variables
        result.introduced_variables |= ev.introduced_variables
        result.inputs_variables_in_function_in_class |= ev.inputs_variables_in_function_in_class
    return result

def v_merge(*exposed_variables: ExposedVariables, _class=False):
    """Merge vertically, or in Sequence: The order matters, here!"""
    result = ExposedVariables()
    for ev in exposed_variables:
        result.input_variables |= (ev.input_variables - result.output_variables)
        result.output_variables |= ev.output_variables
        result.nonlocal_variables |= ev.nonlocal_variables
        result.global_variables |= ev.global_variables
        result.introduced_variables |= ev.introduced_variables
        if not _class:
            result.inputs_variables_in_function_in_class |= (ev.inputs_variables_in_function_in_class - result.output_variables)
        else:
            result.inputs_variables_in_function_in_class |= ev.inputs_variables_in_function_in_class
    return result


# Unit test:
# ev1 = ExposedVariables(input_variables={'a', 'b'}, output_variables={'c', 'd'}, assigned_variables={'c', 'd'}, )
# ev2 = ExposedVariables(input_variables={'e', 'c'}, output_variables={'f', 'd'}, assigned_variables={'f', 'd'}, )
# v_merge(ev1, ev2)




############################################################################################################################
########### THE AST VISITORS: the first produces a TempScope, the second uses it to produce a Scope #######################
############################################################################################################################

class TempScopeVisitor(ast.NodeVisitor):
    def __init__(self, variables: ExposedVariables, is_lhs_target=False, is_also_input_of_aug_assign=False, _class=False):
        """ Not how it receives 'variables' by REFERENCE !! """
        # self.node: ast.AST = node
        # self.parent: "TempScope" = parent
        self.variables: ExposedVariables = variables  # dataclasses.field(default_factory=ExposedVariables)
        self.is_lhs_target = is_lhs_target
        self.is_also_input_of_aug_assign = is_also_input_of_aug_assign
        self._class = _class

    # NOTE NamedExpr(target, value): HERE we go. WAIT tho.. This is ALREADY FINE cuz the target is a Store? isnt this right?
    # NOTE Attribute(value, attr, ctx) where attr is a BARE STR and ctx like in name: SOULD i act on this?? K, probably not
    # NOTE: About the := operator - it SHOULD be already handled by the visitName's !!!

    def visit_Name(self, name_node, ):
        is_input = type(name_node.ctx) is ast.Load or self.is_also_input_of_aug_assign
        is_output = self.is_lhs_target or type(name_node.ctx) is ast.Store or type(name_node.ctx) is ast.Del
        # is_introduction = # TODO

        if is_input and not name_node.id in self.variables.output_variables:  
            # TODO: This is an attempt to fix the problem of "k:=x" within an expression.
            # it's NOT OBVIOUS that it doesn't break anything !!
            self.variables.input_variables.add(name_node.id)
        if is_output:
            self.variables.output_variables.add(name_node.id)

    def visit_Subscript(self, subscr_node):  # HERE we go!
        if type(subscr_node.ctx) in [ast.Store, ast.Del] or self.is_lhs_target:
            lhs_visitor = TempScopeVisitor(self.variables, is_lhs_target=True, is_also_input_of_aug_assign=self.is_also_input_of_aug_assign, _class=self._class)
            lhs_visitor.visit(subscr_node.value)
        elif type(subscr_node.ctx) is ast.Load:
            self.visit(subscr_node.value)
        else:
            raise RuntimeError("Unsupported node type: {name_node}".format(name_node=subscr_node))
        lhs_load_visitor = TempScopeVisitor(self.variables, is_lhs_target=False, is_also_input_of_aug_assign=False, _class=self._class)
        lhs_load_visitor.visit(subscr_node.slice)

    def visit_Attribute(self, attribute):  # HERE we go!
        if type(attribute.ctx) in [ast.Store, ast.Del] or self.is_lhs_target:
            lhs_visitor = TempScopeVisitor(self.variables, is_lhs_target=True, is_also_input_of_aug_assign=self.is_also_input_of_aug_assign, _class=self._class)
            lhs_visitor.visit(attribute.value)
        elif type(attribute.ctx) is ast.Load:
            self.visit(attribute.value)
        else:
            raise RuntimeError("Unsupported node type: {name_node}".format(name_node=attribute))
        
    def visit_Assign(self, assign_node):  # HERE we go!
        value = get_vars_for_nodes(self, assign_node.value)
        targets = get_vars_for_nodes(self, *assign_node.targets)
        self.variables = v_merge(self.variables, *value, h_merge(*targets), _class=self._class)

    def visit_AugAssign(self, augassign_node):  # HERE we go!
        if type(augassign_node.target.ctx) in [ast.Store, ast.Del] or self.is_lhs_target:
            lhs_visitor = TempScopeVisitor(self.variables, is_lhs_target=True, is_also_input_of_aug_assign=True, _class=self._class)
            lhs_visitor.visit(augassign_node.target)
        elif type(augassign_node.target.ctx) is ast.Load:
            self.visit(augassign_node.target)
        else:
            raise RuntimeError("Unsupported node type: {name_node}".format(name_node=augassign_node))
        value_visitor = TempScopeVisitor(self.variables, is_lhs_target=False, is_also_input_of_aug_assign=False, _class=self._class)
        value_visitor.visit(augassign_node.value)

    def visit_List(self, list_node: ast.List):
        if type(list_node.ctx) in [ast.Store, ast.Del] or self.is_lhs_target:
            lhs_visitor = TempScopeVisitor(self.variables, is_lhs_target=True, is_also_input_of_aug_assign=self.is_also_input_of_aug_assign, _class=self._class)
            visit_all(lhs_visitor, list_node.elts)
        elif type(list_node.ctx) is ast.Load:
            super().generic_visit(list_node)
        else:
            raise RuntimeError("Unsupported node type: {name_node}".format(name_node=list_node))

    def visit_Tuple(self, tuple_node: ast.Tuple):
        if type(tuple_node.ctx) in [ast.Store, ast.Del] or self.is_lhs_target:
            lhs_visitor = TempScopeVisitor(self.variables, is_lhs_target=True, is_also_input_of_aug_assign=self.is_also_input_of_aug_assign, _class=self._class)
            visit_all(lhs_visitor, tuple_node.elts)
        elif type(tuple_node.ctx) is ast.Load:
            super().generic_visit(tuple_node)
        else:
            raise RuntimeError("Unsupported node type: {name_node}".format(name_node=tuple_node))

    def visit_alias(self, alias_node):
        variable = alias_node.asname if alias_node.asname is not None else alias_node.name
        self.variables.output_variables.add(variable)

    def visit_arg(self, arg):
        self.variables.introduced_variables.add(arg.arg)

    def _visit_function(self, func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda], func_name=None):
        
        # 1. Visit the type comment and the decorators
        if type(func_node) != ast.Lambda:
            visit_all(self, getattr(func_node, 'type_comment', None), func_node.decorator_list, func_node.returns)

        # 2. Visit the arguments
        argument_visitor = TempScopeVisitor(ExposedVariables(), is_lhs_target=False, is_also_input_of_aug_assign=False, _class=self._class)
        ArgumentsVisitor(self, argument_visitor).visit(func_node.args)
        self.variables.input_variables |= argument_visitor.variables.input_variables
        self.variables.output_variables |= argument_visitor.variables.output_variables

        # 3. Visit the body
        if type(func_node.body) == list:
            vars_stmts_body = get_vars_for_nodes(self, *func_node.body, _class=False)  
        else: # This is the Lambda case
            vars_stmts_body = get_vars_for_nodes(self, func_node.body, _class=False)
        vars_body = join_body_stmts_into_vars(*vars_stmts_body, _class=False)
        
        # 4 get func name:
        func_name_set: Set[str] = set([func_name]) if func_name else set()

        # 4. Keep the body's Input but REMOVE the arguments' introduced variables. The only output is the function name, PLUS eventual outputs from Self
        # And also the name itself, for Recursive calls

        input_vars = (vars_body.input_variables - argument_visitor.variables.introduced_variables - func_name_set) | vars_body.global_variables
        inputs_variables_in_function_in_class = (vars_body.inputs_variables_in_function_in_class - argument_visitor.variables.introduced_variables - func_name_set) | vars_body.global_variables

        if self._class:
            self.variables.inputs_variables_in_function_in_class |= input_vars
        else:
            self.variables.input_variables |= (input_vars - self.variables.output_variables)
            self.variables.inputs_variables_in_function_in_class |= (inputs_variables_in_function_in_class - self.variables.output_variables)

        # self.variables.inputs_variables_in_function_in_class |= (vars_body.inputs_variables_in_function_in_class - argument_visitor.variables.introduced_variables - func_name_set) | vars_body.global_variables
        # The following whould Never happen if not self._class, but u never know:
        self.variables.output_variables |= func_name_set
        # self.variables.nonlocal_variables |= vars_body.nonlocal_variables
        self.variables.global_variables |= vars_body.global_variables
        # TODO: globals??  IN THEORY, a global is 

    def visit_FunctionDef(self, func_node: ast.FunctionDef):
        self._visit_function(func_node, func_name=func_node.name)

    def visit_AsyncFunctionDef(self, func_node: ast.AsyncFunctionDef):
        self._visit_function(func_node, func_name=func_node.name)

    def visit_Lambda(self, func_node: ast.Lambda):
        self._visit_function(func_node)

    def _visit_comprehension(self, targets: List[ast.AST], comprehensions: List[ast.comprehension]):

        comprehension_scopes = []
        all_vars_target = []
        for comprehension in comprehensions:
            vars_target, vars_iter = get_vars_for_nodes(self, comprehension.target, comprehension.iter)
            vars_ifs = get_vars_for_nodes(self, *comprehension.ifs)
            comprehension_scopes.append(v_merge(vars_target, vars_iter, *vars_ifs))
            all_vars_target.extend(vars_target.output_variables)

        targets_scopes = get_vars_for_nodes(self, *targets)

        vars_scope = v_merge(*comprehension_scopes, h_merge(*targets_scopes))
        # Remove vars target from the output vars by hand:
        vars_scope.output_variables -= set(all_vars_target)
        self.variables = v_merge(self.variables, vars_scope)

    def visit_DictComp(self, comp_node: ast.DictComp):
        return self._visit_comprehension([comp_node.key, comp_node.value], comp_node.generators)

    def visit_ListComp(self, comp_node: ast.ListComp):
        return self._visit_comprehension([comp_node.elt], comp_node.generators)

    def visit_SetComp(self, comp_node: ast.SetComp):
        return self._visit_comprehension([comp_node.elt], comp_node.generators)

    def visit_GeneratorExp(self, comp_node: ast.GeneratorExp):
        return self._visit_comprehension([comp_node.elt], comp_node.generators)

    def _visit_for(self, node: Union[ast.For, ast.AsyncFor]):
        visit_all(self, getattr(node, 'type_comment', None))
        vars_iter, vars_target = get_vars_for_nodes(self, node.iter, node.target)
        vars_stmts_body = get_vars_for_nodes(self, node.target, node.iter, *node.body)
        vars_body = join_body_stmts_into_vars(*vars_stmts_body, _class=self._class)
        vars_stmts_orelse = get_vars_for_nodes(self, node.target, node.iter, *node.orelse)
        vars_orelse = join_body_stmts_into_vars(*vars_stmts_orelse, _class=self._class)

        self.variables = v_merge(self.variables, v_merge(vars_iter, vars_target, h_merge(vars_body, vars_orelse), _class=self._class), _class=self._class)

    def visit_For(self, node: ast.For):
        self._visit_for(node)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self._visit_for(node)


    def _visit_with(self, node: Union[ast.With, ast.AsyncWith]):
        visit_all(self, getattr(node, 'type_comment', None))
        vars_stmts_body = get_vars_for_nodes(self, *node.body)
        vars_body = join_body_stmts_into_vars(*vars_stmts_body) # TO CHECK: Is this right, _class=self._class?

        items_vars = []
        for item in node.items:
            vars = get_vars_for_nodes(self, item.context_expr, *([item.optional_vars] if item.optional_vars else []))
            items_vars.append(v_merge(*vars))  # TO CHECK: Is this right?

        items_vars = v_merge(*items_vars, _class=self._class) # TO CHECK: Is this right?
        self.variables = v_merge(self.variables, v_merge(items_vars, vars_body, _class=self._class), _class=self._class)  # TO CHECK: Is this right?

    def visit_With(self, node: ast.With):
        self._visit_with(node)
    
    def visit_AsyncWith(self, node: ast.AsyncWith):
        self._visit_with(node)

    def _visit_if_while(self, node: Union[ast.While, ast.If]):
        vars_test = get_vars_for_nodes(self, node.test, _class=self._class)
        vars_stmts_body = get_vars_for_nodes(self, *node.body, _class=self._class)
        vars_body = join_body_stmts_into_vars(*vars_stmts_body, _class=self._class)
        vars_stmts_orelse = get_vars_for_nodes(self, *node.orelse, _class=self._class)
        vars_orelse = join_body_stmts_into_vars(*vars_stmts_orelse, _class=self._class)

        self.variables = v_merge(self.variables, v_merge(*vars_test, h_merge(vars_body, vars_orelse), _class=self._class), _class=self._class)
        
    def visit_If(self, node: ast.If):
        self._visit_if_while(node)

    def visit_While(self, node: ast.While):
        self._visit_if_while(node)  

    def _visit_try(self, node: ast.Try):  # ast.TryStar
        vars_stmts_body = get_vars_for_nodes(self, *node.body, _class=self._class)
        vars_body = join_body_stmts_into_vars(*vars_stmts_body, _class=self._class)
        vars_stmts_orelse = get_vars_for_nodes(self, *node.orelse, _class=self._class)
        vars_orelse = join_body_stmts_into_vars(*vars_stmts_orelse, _class=self._class)
        vars_stmts_finalbody = get_vars_for_nodes(self, *node.finalbody, _class=self._class)
        vars_finalbody = join_body_stmts_into_vars(*vars_stmts_finalbody, _class=self._class)

        all_vars_handlers = []
        for handler in node.handlers:
            scope = TempScopeVisitor(ExposedVariables(), is_lhs_target=self.is_lhs_target, is_also_input_of_aug_assign=self.is_also_input_of_aug_assign, _class=self._class)
            visit_all(scope, handler.type, handler.name)
        
            vars_stmts_handler = get_vars_for_nodes(scope, *handler.body, _class=self._class)
            vars_handler = join_body_stmts_into_vars(*vars_stmts_handler, _class=self._class)

            all_vars_handlers.append(v_merge(scope.variables, vars_handler, _class=self._class))  # TO CHECK: Is this right?

        self.variables = v_merge(self.variables, v_merge(vars_body, h_merge(*all_vars_handlers, vars_orelse), vars_finalbody, _class=self._class), _class=self._class)
        
    def visit_Try(self, node: ast.Try):
        self._visit_try(node)

    def visit_TryStar(self, node):
        self._visit_try(node)

    def visit_ClassDef(self, class_node: ast.ClassDef):
        visit_all(self, class_node.bases, class_node.keywords, class_node.decorator_list, getattr(class_node, "type_params", None))

        vars_stmts_body = get_vars_for_nodes(self, *class_node.body, _class=True)
        vars_body = join_body_stmts_into_vars(*vars_stmts_body, _class=True)  

        self.variables.input_variables |= (vars_body.input_variables - self.variables.output_variables)
        self.variables.inputs_variables_in_function_in_class |= (vars_body.inputs_variables_in_function_in_class)
        self.variables.output_variables |= set([class_node.name])
        self.variables.nonlocal_variables |= vars_body.nonlocal_variables
        self.variables.global_variables |= vars_body.global_variables


    def visit_Global(self, global_node):
        for name in global_node.names:
            self.variables.global_variables.add(name)

    def visit_Nonlocal(self, nonlocal_node):
        for name in nonlocal_node.names:
            self.variables.nonlocal_variables.add(name)

class ArgumentsVisitor(ast.NodeVisitor):
    """ Util visitor to handle args only """
    def __init__(self, expr_scope, arg_scope):
        self.expr_scope = expr_scope
        self.arg_scope = arg_scope

    def visit_arg(self, node):
        self.arg_scope.visit(node)
        visit_all(self.expr_scope, node.annotation, getattr(node, 'type_comment', None))

    def visit_arguments(self, node):
        super().generic_visit(node)

    def generic_visit(self, node):
        self.expr_scope.visit(node)





####################################################################################################
######################### HELPER FUNCTIONS ##########################################
####################################################################################################
        
def join_body_stmts_into_vars(*stmts: ExposedVariables, _class=False):
    return v_merge(*stmts, _class=_class)  # CHECK: Is this enough?

def get_vars_for_nodes(visitor: TempScopeVisitor, *nodes: ast.AST, _class=False):
    scopes = [TempScopeVisitor(ExposedVariables(), is_lhs_target=visitor.is_lhs_target, is_also_input_of_aug_assign=visitor.is_also_input_of_aug_assign, _class=visitor._class or _class) for _ in nodes]
    for scope, node in zip(scopes, nodes):
        scope.visit(node)
    return [scope.variables for scope in scopes]

def visit_all(visitor, *nodes):
    for node in nodes:
        if node is None:
            pass
        elif isinstance(node, list):
            visit_all(visitor, *node)
        else:
            visitor.visit(node)




def get_name(node):
    if type(node) in [ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]:
        return node.name
    elif type(node) is ast.Name:
        return node.id
    elif type(node) is ast.alias:
        return node.asname if node.asname is not None else node.name
    else:
        raise RuntimeError("Unknown node type")



def annotate(dag_nodes: ast.AST):
    annotator = TempScopeVisitor(ExposedVariables()) # TODO: class_binds_near
    annotator.visit(tree)
    return annotator.variables

def get_input_variables_for(annotated_variables: ExposedVariables):
    """
    Returns a list of all the variables that are referenced in the given scope
    """
    return annotated_variables.input_variables | annotated_variables.inputs_variables_in_function_in_class, set([])

def get_output_variables_for(annotated_variables):
    """
    Returns a list of all the variables that are referenced in the given scope
    """
    return annotated_variables.output_variables, set([])





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













######### BROKENS:


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
# TODO: arguably Broken . Should f be an output ?

# test_no_nonlocal
code = """
def f():
    x += a
"""
tree = ast.parse(code).body[0]
inputs, errors = get_input_variables_for(annotate(tree)); inputs
outputs, errors = get_output_variables_for(annotate(tree)); outputs
# Turns out it's CORRECT IN PYTHON that x doesnt work like this.. Neat!
# TODO: Broken !!


# test_no_nonlocal
code = """
def f(x):
    def g():
        x += 2
    return g
"""
tree = ast.parse(code).body[0]
# TODO: Broken !!

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
# TODO: Broken !!


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
# TODO: Broken !!



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
# TODO: Broken !!


# test_multiarg_lambda
code = """
lambda x, y, *args: x if y else args
"""
# TODO: Check...

# test_nested_lambdas
code = """
lambda x, y: lambda y, z: t + x + y + z
t = x = y = z = 0
"""
# TODO: Check...

















####################################################################################################
####################################################################################################
####################################################################################################


x = 1
y = 1
class A:
    x = 2
    y = 2
    def f(self):
        x = 3

        class B:
            x = 4
            y = 4
            def g(self):
                return x
            def h(self):
                return y
    
        return B()
    
    class B:
        x = 5
        def g(self):
            return x
        def gdef(self, v=x):
            return v
        
    a = B()



A().a.g()  # This is 1, meaning x IS an input
A().a.gdef()  # This is 5, meaning only what is in the BODY of a function becomes "Global"
A().f().g()  # This is 3, meaning x IS NOT an input
A().f().h()  # 1 or 2? In my current system (collapsing classVars @ Func level) would coe out 2...



x = "out"
xout = " xout"

class A:
    x = "in" 
    y = x + (lambda: x)() + xout

a = A().y
a


a = 5
def geta():
    return a

geta()
a = 6
geta()



def i(a, x):
    return x+" iout "

v = "vout"
k = "3"

class A:
    @staticmethod
    def i(self, x):
        return x + " iin " 
    
    def i2(self, x):
        return x + " i2in "+ "(" + i(0, "iwithini2:") + ")" "(vwithini2:" +v + ")"
    
    i3 = lambda x: x + " i3in "

    v = "5"

    v2 = v + " 100 " + k  +" " + i2(0, "3") + i(0, "4") + i3("5")

    def j(self):
        return self.i2("6") + v
    
a = A()
a.j()

a.v2