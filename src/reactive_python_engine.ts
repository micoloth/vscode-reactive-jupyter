export const scriptCode = `

import ast
import json
import hashlib
import itertools
import functools
import dataclasses 
from collections import abc, deque
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Callable


class ReactivePythonDagBuilderUtils__():
    """Everything is packaged in here in order not to pollute the global namespace.
    """

    @staticmethod
    def define_reactive_python_utils():
        """Everything is packaged in here in order not to pollute the global namespace.
        """
        
        # We define here the small subset of NetworkX objects that we need to use, 
        # in order to avoid pip-installing the whole NetworkX library on the user's machine:
        DiGraph, topological_sort, has_path = ReactivePythonDagBuilderUtils__.define_networkx_objects()

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


        ############################################################################################################################
        ########### THE AST VISITORS: extract input and output variables from Python statements: #######################
        ############################################################################################################################


        @dataclasses.dataclass
        class ExposedVariables():
            input_variables: Set[str] = dataclasses.field(default_factory=set)
            output_variables: Set[str] = dataclasses.field(default_factory=set)
            nonlocal_variables: Set[str] = dataclasses.field(default_factory=set)
            global_variables: Set[str] = dataclasses.field(default_factory=set)
            introduced_variables: Set[str] = dataclasses.field(default_factory=set) # These are the params in a function, or the target n a For or a With or exception, or even the variables in a class
            inputs_variables_in_function_in_class: Set[str] = dataclasses.field(default_factory=set)  # These are the variables that are used by the body of a function that is inside a class


        class TempScopeVisitor(ast.NodeVisitor):
            def __init__(self, variables: ExposedVariables, is_lhs_target=False, is_also_input_of_aug_assign=False, _class=False):
                """ Not how it receives 'variables' by REFERENCE !! """
                # self.node: ast.AST = node
                # self.parent: "TempScope" = parent
                self.variables: ExposedVariables = variables  # dataclasses.field(default_factory=ExposedVariables)
                self.is_lhs_target = is_lhs_target
                self.is_also_input_of_aug_assign = is_also_input_of_aug_assign
                self._class = _class

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

                input_vars = (vars_body.input_variables - argument_visitor.variables.introduced_variables - func_name_set) 
                inputs_variables_in_function_in_class = (vars_body.inputs_variables_in_function_in_class - argument_visitor.variables.introduced_variables - func_name_set) 
                globals_and_nonlocals = vars_body.global_variables | vars_body.nonlocal_variables

                if self._class:
                    self.variables.inputs_variables_in_function_in_class |= input_vars | globals_and_nonlocals
                else:
                    self.variables.input_variables |= (input_vars - self.variables.output_variables) | globals_and_nonlocals
                    self.variables.inputs_variables_in_function_in_class |= (inputs_variables_in_function_in_class - self.variables.output_variables) | globals_and_nonlocals

                self.variables.output_variables |= func_name_set
                # self.variables.nonlocal_variables |= vars_body.nonlocal_variables
                self.variables.global_variables |= vars_body.global_variables

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
                    visit_all(scope, handler.type)
                
                    vars_stmts_handler = get_vars_for_nodes(scope, *handler.body, _class=self._class)
                    vars_handler = join_body_stmts_into_vars(*vars_stmts_handler, _class=self._class)

                    # Remove the handler.name from the input variables AND output variables, by hand:
                    vars_handler.input_variables -= set([handler.name])
                    vars_handler.output_variables -= set([handler.name])
                    vars_handler.inputs_variables_in_function_in_class -= set([handler.name])

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

        def annotate(tree: ast.AST):
            annotator = TempScopeVisitor(ExposedVariables())
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

        reserved_terms = {
            "False", "def", "if", "raise", "None", "del", "import", "return", "True", "elif", "in", "try", "and", "else", "is", "while",
            "as", "except", "lambda", "with", "assert", "finally", "nonlocal", "yield", "break", "for", "not", "class", "from", "or", "continue", "global", "pass",
            "list", "set", "dict", "tuple", "str", "int", "float", "bool", "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
            "sum", "min", "max", "abs", "round", "len", "print", "input", "open", "close", "read", "write", "readline", "readlines", "seek", "tell", "split", "join", "replace", "startswith", "endswith",
            "find", "index", "rindex", "strip", "lower", "upper", "capitalize", "count", "format", "append", "extend", "insert", "pop", "remove", "clear", "copy",
            "update", "keys", "values", "items", "get", "add", "union", "intersection", "difference", "symmetric_difference", "issubset", "issuperset",
            "*", "**", "=", "==", "!=", "<", ">", "<=", ">=", "+", "-", "/", "//", "%",
        }

        def filter_reserved_terms(names: Set[str]):
            return names.difference(reserved_terms)


        ####################################################################################################
        ################   DAG FROM AST ##########################################
        ####################################################################################################


        @dataclasses.dataclass
        class DagNode:
            lineno: int
            end_lineno: int
            tree_ast: ast.AST
            text: str  # Honestly this is only used for debugging ...
            input_vars: Set[str]  # Represents the variables that need to be defined elsewhere to be used in the eval of this node
            output_vars: Set[str]  # Represents the variables that are defined in this node (either via assignment or function definition)  and can be used later
            errored_input_vars: Set[str]  # Represents the variables that are defined in this node (either via assignment or function definition)  and can be used later
            errored_output_vars: Set[str]  # Represents the variables that are defined in this node (either via assignment or function definition)  and can be used later
            stale: bool = False
            needstobestale: bool = False
            depends_on_stale_nodes_not_selected: bool = False  # Helper var used in a specific function to do a specific thing, it's just handy to have it here directly, but ignore for now
        
        def node_repr_hash(node):
            repr = f"{node['lineno']}-{node['end_lineno']}: {node['text'].strip()}"
            # Hash this string to some 8-digit number:
            return int(hashlib.sha256(repr.encode('utf-8')).hexdigest(), 16) % 10**8


        def ast_to_dagnodes(ast_nodes, full_code: Optional[str] = None) -> List[DagNode]:
            result: List[DagNode] = []
            splitted_code = full_code.splitlines() if full_code else None

            for node in ast_nodes.body:
                annotated_node = annotate(node)
                inputs_, error_inputs = get_input_variables_for(annotated_node)
                inputs_, error_inputs = filter_reserved_terms(inputs_), filter_reserved_terms(error_inputs)
                outputs_, error_outputs = get_output_variables_for(annotated_node)
                outputs_, error_outputs = filter_reserved_terms(outputs_), filter_reserved_terms(error_outputs)
                result.append(DagNode(
                    lineno=node.lineno-1,
                    end_lineno=node.end_lineno-1,
                    tree_ast=node,
                    # text="acapo".join(splitted_code[node.lineno-1:node.end_lineno]) if splitted_code else "",
                    # Recompute code by ast-dumping the node:
                    text=ast.unparse(node),
                    input_vars=inputs_,
                    output_vars=outputs_,
                    errored_input_vars=error_inputs,
                    errored_output_vars=error_outputs,
                ))

            return result


        def find_dagnodes_to_fuse_sameline(dag_nodes: DiGraph, full_code: str):
            nodes_to_fuse: List[List[int]] = []
            
            # Additional step: Identify all the groups of nodes that are joined in a single line
            # Ie, where the end_lineno of a node is THE SAME AS the lineno of the next node:
            # Then, we want to fuse those nodes, because they share a line
            # > THIS WORKS BECAUSE THE NODES IN THE dag_nodes ARE SUPPOSED TO BE SORTED BY LINE NUMBER:
            last_cheched_node = -1
            last_checked_end_lineno = -1
            for node in dag_nodes.nodes:
                if dag_nodes.nodes[node]['text']== '_START_' :
                    continue
                if dag_nodes.nodes[node]['lineno'] == last_checked_end_lineno:
                    # Check if there is already a list of nodes to fuse that contains the last_checked_node:
                    if len(nodes_to_fuse) > 0 and nodes_to_fuse[-1][-1] == last_cheched_node:
                        # If so, just append the current node to that list:
                        nodes_to_fuse[-1].append(node)
                    else:
                        # Otherwise, create a new list of nodes to fuse:
                        nodes_to_fuse.append([last_cheched_node, node])
                last_cheched_node = node
                last_checked_end_lineno = dag_nodes.nodes[node]['end_lineno']
            
            return nodes_to_fuse
        
        def find_dagnodes_to_fuse_cells(dag_nodes: DiGraph, full_code: str):
            # Additional step: Identify all the groups of nodes that are joined in a single Cell.
            # We identify cells with special comments: # % [   and # % ]
            # For example, if we have:
            # # % [
            # a = 1
            # b = c
            # a = b
            # # % ]
            # Then we want to identify that this is just 1 node, whose input is c and whose output is a and b
            splitted_code = full_code.splitlines()
            delimiter_start, delimiter_end = "# % [", "# % ]"
            cells: List[List[int]] = []
            # Find delimiter lines in full_code:
            delimiter_lines_start= [i for i, line in enumerate(splitted_code) if line.startswith(delimiter_start)]
            delimiter_lines_end= [i for i, line in enumerate(splitted_code) if line.startswith(delimiter_end)]
            if len(delimiter_lines_start) != len(delimiter_lines_end):
                print(f"Invalid number of delimiter lines in code: {len(delimiter_lines_end)} end lines and {len(delimiter_lines_start)} start lines")
                return cells
            last_end_line = -1
            for start_line, end_line in zip(delimiter_lines_start, delimiter_lines_end):
                if start_line <= last_end_line or end_line <= start_line:
                    print(f"Invalid delimiter lines in code: {start_line} {end_line}")
                    return cells
                cells.append([
                    node for node in dag_nodes.nodes  # TODO make not quadratic
                    if dag_nodes.nodes[node]['text']!= '_START_' and dag_nodes.nodes[node]['lineno'] >= start_line and (dag_nodes.nodes[node]['end_lineno'] <= end_line)
                ])
            return cells
        
        def fuse_two_nodes(graph, node1, node2):
            # For each edge Into node1, add an edge Into node2.
            # In particular, check if the edge already exixts, and if it does, add the "vars" field to it.
            # Same for all the edges Out of node1.
            # IMPORTANT: the node that gets removed is node1, and node2 is the one that remains.
            for pred in graph.predecessors(node2):
                if graph.has_edge(pred, node1):
                    graph.edges[pred, node1]['vars'] = list(set(graph.edges[pred, node1]['vars'] + graph.edges[pred, node2]['vars']))
                elif pred != node1:
                    graph.add_edge(pred, node1, vars=graph.edges[pred, node2]['vars'])
            for succ in graph.successors(node2):
                if graph.has_edge(node1, succ):
                    graph.edges[node1, succ]['vars'] = list(set(graph.edges[node1, succ]['vars'] + graph.edges[node2, succ]['vars']))
                elif succ != node1:
                    graph.add_edge(node1, succ, vars=graph.edges[node2, succ]['vars'])
            
            # Furthermore, modify the "input_vars" and "output_vars" fields of node2.
            n1, n2 = graph.nodes[node1], graph.nodes[node2]
            n1['input_vars'] = n2['input_vars'] | n1['input_vars']
            n1['output_vars'] = n2['output_vars'] | n1['output_vars']
            n1['errored_input_vars'] = n2['errored_input_vars'] | n1['errored_input_vars']
            n1['errored_output_vars'] = n2['errored_output_vars'] | n1['errored_output_vars']
            # and also the lines and text
            n1['text'] = n1['text'] + '\\n' + n2['text'] if n1['lineno'] <= n2['lineno'] else n2['text'] + '\\n' + n1['text']
            n1['lineno'] = min(n2['lineno'], n1['lineno'])
            n1['end_lineno'] = max(n2['end_lineno'], n1['end_lineno'])
            # and also the tree_ast
            n1['tree_ast'] = [n1['tree_ast'], n2['tree_ast']]

            # Finally, delete node1
            graph.remove_node(node2)

        def fuse_nodes(graph, full_code):
            all_nodes_to_fuse = find_dagnodes_to_fuse_sameline(graph, full_code)
            for nodes in all_nodes_to_fuse:
                for node in nodes[1:]:
                    fuse_two_nodes(graph, nodes[0], node)
            all_nodes_to_fuse = find_dagnodes_to_fuse_cells(graph, full_code)
            for nodes in all_nodes_to_fuse:
                for node in nodes[1:]:
                    fuse_two_nodes(graph, nodes[0], node)
            return graph


        def dagnodes_to_dag(dag_nodes, add_out_of_order=False, add_starting_node=False):
            # Initialize a graph with all the dag_nodes:
            graph = DiGraph()
            for i, node in enumerate(dag_nodes):
                graph.add_node(i, **node.__dict__)
            
            # Add edges:
            # The idea is: Iterate on the "dag_nodes" list in REVERSE order.
            # For each node, and for each input var of the node, find the LAST node that outputs that variable (ie the FIRST in reversed order)
            # Add an edge from that node to the current node

            # First, for each var, find all the nodes that output that variable:
            outputs_for_each_var: Dict[str, Set[int]] = {} # key is the var, value is the set of nodes that output that var
            for i, node in enumerate(dag_nodes):
                for var in node.output_vars | node.errored_output_vars:
                    if var not in outputs_for_each_var:
                        outputs_for_each_var[var] = set()
                    outputs_for_each_var[var].add(i)
            
            out_of_order_edges: List[Tuple[int, int, str]] = []

            # Now, iterate on the "dag_nodes" in REVERSE order:
            for i, node in enumerate(reversed(dag_nodes)):  # i is the index in the "dag_nodes" list FROM THE END !!
                node_index = len(dag_nodes) - i - 1
                for var in node.input_vars | node.errored_input_vars:
                    if var in outputs_for_each_var:
                        # Get the LAST node that outputs that variable, whose index is STRICTLY LESS THAN the current node index:
                        relevant_output = max({x for x in outputs_for_each_var[var] if x < node_index}, default=None)
                        if relevant_output is not None: 
                            # if there is no (relevant_output->node_index) edge, add it- otherwise add var to the "vars" list dataclasses.field:
                            if graph.has_edge(relevant_output, node_index):
                                graph[relevant_output][node_index]['vars'].append(var)
                            else:
                                graph.add_edge(relevant_output, node_index, vars=[var])
                        elif add_out_of_order and relevant_output is None and len(outputs_for_each_var[var]) == 1 and list(outputs_for_each_var[var])[0] != node_index:
                            # If there are no prev nodes that output the var, BUT there is ONLY ONE node that output it ever, THEN you can CONSIDER this for defining an out-of-order definition- 
                            # BUT, defer it to when you can check that you are not creating CYCLES!
                            out_of_order_edges.append((list(outputs_for_each_var[var])[0], node_index, var))
            
            # Now, add the out-of-order edges, but only if they don't create cycles:
            for edge in out_of_order_edges:
                if not has_path(graph, edge[1], edge[0]):
                    if graph.has_edge(edge[0], edge[1]):
                        graph[edge[0]][edge[1]]['vars'].append(edge[2])
                    else:
                        graph.add_edge(edge[0], edge[1], vars=[edge[2]])

            # Add a single node "_START_" that has an edge pointing to ALL THE NODES THAT HAVE NO PARENTS
            if add_starting_node:
                graph.add_node("_START_", stale=False, needstobestale=False, text="_START_")
                for i, node in enumerate(dag_nodes):
                    if len(list(graph.predecessors(i))) == 0:
                        graph.add_edge("_START_", i, vars=[])

            return graph

        def ast_equality(ast1: Union[ast.AST, List], ast2: Union[ast.AST, List]):
            if type(ast1) is list and type(ast2) is list:
                if len(ast1) != len(ast2):
                    return False
                for i in range(len(ast1)):
                    if not ast_equality(ast1[i], ast2[i]):
                        return False
                return True
            elif type(ast1) is not list and type(ast2) is not list:
                return ast.dump(ast1) == ast.dump(ast2)
            else:
                return False


        def update_staleness_info_in_new_dag(old_dag, new_dag):
            """old_dag has a set of nodes, each of which has a "stale" flag.

            new_dag is another DAG, which could be slighly different.

            Start from the "_START_" node of the new dag, and for each node, check if it is in the old_dag (Compare the "ast_tree" dataclasses.field by equality).

            Each node in the new_dag NEEDSTOBESTALE if and only if: 
                - ANY of its parents in the new_dag NEEDSTOBESTALE, OR
                - it Doesn't have a corresponding node in the old_dag (ie it's new/ has been changed).

            Subsequently, 
            Each node is STALE if:
                - it NEEDSTOBESTALE, OR
                - ANY of its parents in the new_dag NEEDSTOBESTALE, OR
                - it Doesn't have a corresponding node in the old_dag (ie it's new/ has been changed).
            Otherwise, it should retain the state it had in the old_dag.

            This double logic is to handle this case:
                - If a node is STALE and one of its childer in FRESH (not stale), but None of them NEEDSTOBESTALE, the the Children SHOULD remain Fresh...

            Args:
                old_dag (_type_): The old dag
                new_dag (_type_): The stale value of each node in the new_dag is updated in-place !!!
            """

            new_node_to_old: Dict[int, ast.AST] = {} # key is the new node, value is the old node that corresponds to it

            # Iterate on the new_dag in topological order:
            for new_node in topological_sort(new_dag):
                if new_node == '_START_': 
                    new_node_to_old['_START_'] = '_START_'; continue
                parents = list(new_dag.predecessors(new_node))
                
                # If any of the parents needstobestale, then needstobestale:
                if any(new_dag.nodes[x]['needstobestale'] for x in parents):
                    new_dag.nodes[new_node]['needstobestale'] = True
                    new_dag.nodes[new_node]['stale'] = True
                    continue
                
                # For all the parents, get the corresponding nodes in the old_dag: if any don't exist, it's stale:
                parents_in_old_dag = [new_node_to_old.get(parent) for parent in parents]
                if any(x is None for x in parents_in_old_dag):
                    new_dag.nodes[new_node]['needstobestale'] = True
                    new_dag.nodes[new_node]['stale'] = True
                    continue

                # Among all the nodes in old_graph that are Successors of All nodes in parents_in_old_dag,
                # Find the one that is the same as the new_node: 
                all_candidates_in_old = set.intersection(*[set(old_dag.successors(parent)) for parent in parents_in_old_dag])
                for candidate in all_candidates_in_old:
                    if ast_equality(old_dag.nodes[candidate]['tree_ast'], new_dag.nodes[new_node]['tree_ast']):
                        new_node_to_old[new_node] = candidate
                        break
                
                # If there is no matching node in the old_dag, then needs to be stale and it's stale:
                if new_node_to_old.get(new_node) is None:
                    new_dag.nodes[new_node]['needstobestale'] = True; 
                    new_dag.nodes[new_node]['stale'] = True; 
                    continue
                else:
                    # RETAIN THE PREVIOUS STATE:
                    new_dag.nodes[new_node]['stale'] = old_dag.nodes[new_node_to_old[new_node]]['stale']

            # Additional step: For all variables that are outputted by a stale node, mark all the nodes that use that variable as stale FROM THE BEGINNING:
            # If the assignements always use a new name, as it should be in a functional setting, this is not necessary.
            # But, if the same variable is reassigned, then it is necessary to mark all the nodes that use that variable as stale.
            # all_outputted_stale_vars = set(itertools.chain.from_iterable([new_dag.nodes[node]['output_vars'] for node in new_dag.nodes if new_dag.nodes[node]['stale']]))
            # for node in topological_sort(new_dag):
            #     if node == '_START_': 
            #         continue
            #     node_data = new_dag.nodes[node]
            #     node_vars = node_data['input_vars'] | node_data['errored_input_vars'] | node_data['output_vars'] | node_data['errored_output_vars']
            #     node_parents = list(new_dag.predecessors(node))
            #     if any(var in all_outputted_stale_vars for var in node_vars) or any(new_dag.nodes[parent]['stale'] for parent in node_parents):
            #         node_data['stale'] = True
        

        def draw_dag(graph):
            from webweb import Web
            # Give the webweb a title.
            web = Web(title='AST_sample')

            # Source and target index, + "var" label:
            edge_list = [
                (source, target, str(graph[source][target].get('vars', [])))
                for source, target in graph.edges
            ]

            # n: {'name': n, 'shape': 's' if n%2==0 else 'o', 'text': "AAA"} 
            nodes = {
                n: {
                    'text': graph.nodes[n].get("text", "..")[:20],
                    '_color': str(graph.nodes[n].get("stale", "BOH"))
                } 
                for n in graph.nodes
            }

            web.networks.my_nice_graph(  # Yes creating trees like this is fucking awful
                adjacency=edge_list,
                nodes=nodes,
            )
            web.display.networkName = 'my_nice_graph'
            web.display.networkLayer = 0
            web.display.colorBy = '_color'
            web.display.sizeBy = 'degree'
            web.display.gravity = .01
            web.display.charge = 30
            web.display.linkLength = 120
            web.display.linkStrenght = 60
            web.display.colorPalette = 'Set2'
            web.display.scaleLinkOpacity = False
            web.display.scaleLinkWidth = True
            web.show()

        def pr(node, is_current=False, include_code=False):
            return [
                node['lineno'], 
                node['end_lineno'], 
                "synced" if (not node['stale']) else "outdated" if (not node.get('depends_on_stale_nodes_not_selected')) else 'dependsonotherstalecode',
                "current" if is_current else "",
                node['text'] if include_code else "",
                node_repr_hash(node) if include_code else ""
            ] 

        def dag_to_node_ranges(current_dag, current_line: Optional[int] = None, get_upstream=True, get_downstream=True, stale_only=False, include_code=False):
            """If current_line is None, it will return ALL the nodes in the dag
            If it is a line number, it will return all the ancestors and/or descendants of the node that contains that line number
            """
            if not current_line:
                if stale_only:
                    nodes_to_return = [n for n in topological_sort(current_dag) if current_dag.nodes[n].get('stale') and current_dag.nodes[n]['text'] != '_START_']
                else:
                    nodes_to_return = [n for n in topological_sort(current_dag) if current_dag.nodes[n]['text'] != '_START_']
                current_node = None
            else:
                current_node = [n for n in current_dag.nodes if current_dag.nodes[n]['text'] != '_START_' and current_dag.nodes[n]['lineno'] <= current_line and current_dag.nodes[n]['end_lineno'] >= current_line]
                assert len(current_node) <=1, f"Attention: Multiple nodes with the same line number: {current_node}"
                if len(current_node) == 0:
                    return []
                current_node = current_node[0]
                
                nodes_to_return = []
                if get_upstream and not stale_only:
                    nodes_to_return += list(directed_ancestors_with_pruning(current_dag, current_node, pruning_condition=lambda x: x['text'] != '_START_'))
                elif get_upstream and stale_only:
                    nodes_to_return += list(directed_ancestors_with_pruning(current_dag, current_node, pruning_condition=lambda x: x['text'] != '_START_' and x.get('stale')))
                    # This^ will get ALL THE STALE ANCESTORS, until you get the FIRST NOT-STALE node, then you prune that branch since it's good.
                if (not get_downstream) or not stale_only or current_dag.nodes[current_node].get('stale'):
                    nodes_to_return += [current_node]  # IF YOU ARE NOT GETTING THE DESCENDANTS (So "up to(including) the current node"), this is inserted EVEN IF NOT STALE, which is very intentional
                if get_downstream and not stale_only:
                    nodes_to_return += list(directed_descendants_with_pruning(current_dag, current_node, pruning_condition=lambda x: x['text'] != '_START_'))
                elif get_downstream and stale_only:
                    nodes_to_return += list(directed_descendants_with_pruning(current_dag, current_node, pruning_condition=lambda x: x['text'] != '_START_' and x.get('stale')))
                    # This will get ALL THE STALE DESCENDANTS, until you get the FIRST NOT-STALE node, then you prune that branch since it's good.
                
                # When you need to Execute code, the order is important, so we sort the nodes by topological order. 
                # When you are just selecting, not so much.  # For now, I'm using the stale_only flag to indicate this difference.
                # Btw: It's unfortunate that directed_ancestors_with_pruning's bfs doesn't return nodes Already sorted, but I'm not gonna find out why...
                if stale_only:
                    nodes_to_return = [n for n in topological_sort(current_dag) if n in set(nodes_to_return)]

                # Additional step: If get_downstream OR if you are running a single block (neither downstream not upstream), 
                # you want to identify which codes are stale but actually depend on other stale code
                # that has NOTHING to do with current_line, and so will not be executed..
                if (get_downstream and stale_only) or (not get_downstream and not get_upstream):
                    # Remove the ones which depend on stale nodes that are Not in nodes_to_returm by setting their depends_on_stale_nodes_not_selected to True:
                    for n in nodes_to_return:
                        depends_on_stale_nodes_not_selected = False
                        for pred in current_dag.predecessors(n):
                            is_selected = pred in nodes_to_return  # Yes, it's quadratic. This is probably the only place where this code is quadratic?
                            if (current_dag.nodes[pred].get('text') != '_START_'
                                and ((not is_selected and current_dag.nodes[pred].get('stale')) or (is_selected and current_dag.nodes[pred].get('depends_on_stale_nodes_not_selected')))):
                                depends_on_stale_nodes_not_selected = True   
                        current_dag.nodes[n]['depends_on_stale_nodes_not_selected'] = depends_on_stale_nodes_not_selected
            
            result = [pr(current_dag.nodes[n], is_current=(n==current_node), include_code=include_code) for n in nodes_to_return]
            result = json.dumps(result)
            return result
        
        def set_all_descendants_to_stale(dag, node):
            # Additional step: Set ALL the descendants of node AND ALSO all nodes using its output and THEIR descendants to STALE: 
            # If the node has no Outputs, this has no effect, of course
            node_outputs = dag.nodes[node]['output_vars'] | dag.nodes[node]['errored_output_vars']
            if len(node_outputs) == 0:
                return
            nodes_to_set_as_stale = set()
            # Get ALl the nodes which use OR define the output variables of the node:
            for n in topological_sort(dag):
                if (
                        # any of the PARENTS is in nodes_to_set_as_stale set: # Remember & means "intersection"
                        (set(dag.predecessors(n)) & nodes_to_set_as_stale)
                        # Or uses the Output in any way:
                        or (dag.nodes[n].get('input_vars', set()) & node_outputs) 
                        or (dag.nodes[n].get('output_vars', set()) & node_outputs)
                        or (dag.nodes[n].get('errored_input_vars', set()) & node_outputs)
                        or (dag.nodes[n].get('errored_output_vars', set()) & node_outputs)
                    ):  
                    nodes_to_set_as_stale.add(n)
                    if n != node:
                        dag.nodes[n]['stale'] = True

        return dagnodes_to_dag, ast_to_dagnodes, draw_dag, update_staleness_info_in_new_dag, get_input_variables_for, get_output_variables_for, annotate, dag_to_node_ranges, fuse_nodes, node_repr_hash, set_all_descendants_to_stale
    

    def __init__(self):
        self.dagnodes_to_dag, self.ast_to_dagnodes, self.draw_dag, self.update_staleness_info_in_new_dag, self.get_input_variables_for, self.get_output_variables_for, self.annotate, self.dag_to_node_ranges, self.fuse_nodes, self.node_repr_hash, self.set_all_descendants_to_stale = self.define_reactive_python_utils()
        self.syntax_error_range = None  ## type: Optional[DiGraph]
        self.current_dag = None  ## type: Optional[DiGraph]
        self.locked_for_execution = None

    def update_dag(self, code: str, current_line: Optional[int] = None):
        try: 
            ast_tree = ast.parse(code, "", mode='exec', type_comments=True)
            self.syntax_error_range = None
        except SyntaxError as e:
            # Get the line number of the syntax error:
            line_number = e.lineno
            self.syntax_error_range = [line_number-1 if line_number else line_number, line_number]
            return
        
        try:
            dagnodes = self.ast_to_dagnodes(ast_tree, code)
            new_dag = self.dagnodes_to_dag(dagnodes, add_out_of_order=True, add_starting_node=True)
            new_dag = self.fuse_nodes(new_dag, code)
                
            if self.current_dag is not None:
                self.update_staleness_info_in_new_dag(self.current_dag, new_dag)
            else:
                # Put all nodes as stale
                for n in new_dag.nodes:
                    new_dag.nodes[n]['stale'] = True
            
            if not self.locked_for_execution:  # Should never happen, but better be careful
                self.current_dag = new_dag
            return
        except Exception as e:
            print(e)
            # raise e
            return e

    def update_dag_and_get_ranges(self, code: Optional[str] = None, current_line: Optional[int] = None, get_upstream=True, get_downstream=True, stale_only=False, include_code=False):
        if code and not self.locked_for_execution: 
            errors = self.update_dag(code) 
            if errors:
                return errors
        if self.syntax_error_range:
            line_number = self.syntax_error_range[0]
            other_selection = line_number and current_line and not -2 <= (line_number - current_line) <= 2
            if other_selection:
                syntax_error_range = [[line_number-1 if line_number else line_number, line_number, "syntaxerror", ""]]
                return json.dumps(syntax_error_range)
            else:
                return []
        if self.current_dag:
            return self.dag_to_node_ranges(self.current_dag, current_line, get_upstream=get_upstream, get_downstream=get_downstream, stale_only=stale_only, include_code=include_code)
        else:
            return []
    
    def ask_for_ranges_to_compute(self, code: Optional[str], current_line: Optional[int] = None, get_upstream=True, get_downstream=True, stale_only=False):
        if self.locked_for_execution:
            return []
        else:
            ranges = self.update_dag_and_get_ranges(code, current_line, get_upstream, get_downstream, stale_only, include_code=True)
            if ranges and not self.syntax_error_range:
                self.locked_for_execution = True
                return ranges
            return []

    def unlock(self):
        self.locked_for_execution = None
        return True
    
    def set_locked_range_as_synced(self, hash: str):
        if not self.locked_for_execution or not self.current_dag:
            return
        for node in self.current_dag:
            if self.node_repr_hash(self.current_dag.nodes[node]) == hash:
                self.current_dag.nodes[node]['stale'] = False

                # Additional step: Set ALL its descendants AND ALSO all nodes using this output and THEIR descendants to STALE: 
                # If the node has no Outputs, this has no effect, of course
                self.set_all_descendants_to_stale(self.current_dag, node)
                return True

    
    

    @staticmethod
    def define_networkx_objects():
        """We define here the small subset of NetworkX(2.8.8) objects that we need to use, 
        in order to avoid pip-installing the whole NetworkX library on the user's machine.

        Note that this is just the NetworkX code, with the documentation stripped out (to save space).
        Please refer to the NetworkX documentation for more information.
        """

        class AtlasView(abc.Mapping):
            """An AtlasView is a Read-only abc.Mapping of Mappings.

            It is a View into a dict-of-dict data structure.
            The inner level of dict is read-write. But the
            outer level is read-only.

            See Also
            ========
            AdjacencyView: View into dict-of-dict-of-dict
            MultiAdjacencyView: View into dict-of-dict-of-dict-of-dict
            """

            __slots__ = ("_atlas",)

            def __getstate__(self):
                return {"_atlas": self._atlas}

            def __setstate__(self, state):
                self._atlas = state["_atlas"]

            def __init__(self, d):
                self._atlas = d

            def __len__(self):
                return len(self._atlas)

            def __iter__(self):
                return iter(self._atlas)

            def __getitem__(self, key):
                return self._atlas[key]

            def copy(self):
                return {n: self[n].copy() for n in self._atlas}

            def __str__(self):
                return str(self._atlas)  # {nbr: self[nbr] for nbr in self})

            def __repr__(self):
                return f"{self.__class__.__name__}({self._atlas!r})"


        class AdjacencyView(AtlasView):
            """An AdjacencyView is a Read-only Map of Maps of Maps.

            It is a View into a dict-of-dict-of-dict data structure.
            The inner level of dict is read-write. But the
            outer levels are read-only.

            See Also
            ========
            AtlasView: View into dict-of-dict
            MultiAdjacencyView: View into dict-of-dict-of-dict-of-dict
            """

            __slots__ = ()  # Still uses AtlasView slots names _atlas

            def __getitem__(self, name):
                return AtlasView(self._atlas[name])

            def copy(self):
                return {n: self[n].copy() for n in self._atlas}

        # DegreeViews
        class DiDegreeView:

            def __init__(self, G, nbunch=None, weight=None):
                self._graph = G
                self._succ = G._succ if hasattr(G, "_succ") else G._adj
                self._pred = G._pred if hasattr(G, "_pred") else G._adj
                self._nodes = self._succ if nbunch is None else list(G.nbunch_iter(nbunch))
                self._weight = weight

            def __call__(self, nbunch=None, weight=None):
                if nbunch is None:
                    if weight == self._weight:
                        return self
                    return self.__class__(self._graph, None, weight)
                try:
                    if nbunch in self._nodes:
                        if weight == self._weight:
                            return self[nbunch]
                        return self.__class__(self._graph, None, weight)[nbunch]
                except TypeError:
                    pass
                return self.__class__(self._graph, nbunch, weight)

            def __getitem__(self, n):
                weight = self._weight
                succs = self._succ[n]
                preds = self._pred[n]
                if weight is None:
                    return len(succs) + len(preds)
                return sum(dd.get(weight, 1) for dd in succs.values()) + sum(
                    dd.get(weight, 1) for dd in preds.values()
                )

            def __iter__(self):
                weight = self._weight
                if weight is None:
                    for n in self._nodes:
                        succs = self._succ[n]
                        preds = self._pred[n]
                        yield (n, len(succs) + len(preds))
                else:
                    for n in self._nodes:
                        succs = self._succ[n]
                        preds = self._pred[n]
                        deg = sum(dd.get(weight, 1) for dd in succs.values()) + sum(
                            dd.get(weight, 1) for dd in preds.values()
                        )
                        yield (n, deg)

            def __len__(self):
                return len(self._nodes)

            def __str__(self):
                return str(list(self))

            def __repr__(self):
                return f"{self.__class__.__name__}({dict(self)})"


        class DegreeView(DiDegreeView):

            def __getitem__(self, n):
                weight = self._weight
                nbrs = self._succ[n]
                if weight is None:
                    return len(nbrs) + (n in nbrs)
                return sum(dd.get(weight, 1) for dd in nbrs.values()) + (
                    n in nbrs and nbrs[n].get(weight, 1)
                )

            def __iter__(self):
                weight = self._weight
                if weight is None:
                    for n in self._nodes:
                        nbrs = self._succ[n]
                        yield (n, len(nbrs) + (n in nbrs))
                else:
                    for n in self._nodes:
                        nbrs = self._succ[n]
                        deg = sum(dd.get(weight, 1) for dd in nbrs.values()) + (
                            n in nbrs and nbrs[n].get(weight, 1)
                        )
                        yield (n, deg)


        class OutDegreeView(DiDegreeView):
            """A DegreeView class to report out_degree for a DiGraph; See DegreeView"""

            def __getitem__(self, n):
                weight = self._weight
                nbrs = self._succ[n]
                if self._weight is None:
                    return len(nbrs)
                return sum(dd.get(self._weight, 1) for dd in nbrs.values())

            def __iter__(self):
                weight = self._weight
                if weight is None:
                    for n in self._nodes:
                        succs = self._succ[n]
                        yield (n, len(succs))
                else:
                    for n in self._nodes:
                        succs = self._succ[n]
                        deg = sum(dd.get(weight, 1) for dd in succs.values())
                        yield (n, deg)


        class InDegreeView(DiDegreeView):
            """A DegreeView class to report in_degree for a DiGraph; See DegreeView"""

            def __getitem__(self, n):
                weight = self._weight
                nbrs = self._pred[n]
                if weight is None:
                    return len(nbrs)
                return sum(dd.get(weight, 1) for dd in nbrs.values())

            def __iter__(self):
                weight = self._weight
                if weight is None:
                    for n in self._nodes:
                        preds = self._pred[n]
                        yield (n, len(preds))
                else:
                    for n in self._nodes:
                        preds = self._pred[n]
                        deg = sum(dd.get(weight, 1) for dd in preds.values())
                        yield (n, deg)

        class NodeDataView(abc.Set):

            __slots__ = ("_nodes", "_data", "_default")

            def __getstate__(self):
                return {"_nodes": self._nodes, "_data": self._data, "_default": self._default}

            def __setstate__(self, state):
                self._nodes = state["_nodes"]
                self._data = state["_data"]
                self._default = state["_default"]

            def __init__(self, nodedict, data=False, default=None):
                self._nodes = nodedict
                self._data = data
                self._default = default

            @classmethod
            def _from_iterable(cls, it):
                try:
                    return set(it)
                except TypeError as err:
                    if "unhashable" in str(err):
                        msg = " : Could be b/c data=True or your values are unhashable"
                        raise TypeError(str(err) + msg) from err
                    raise

            def __len__(self):
                return len(self._nodes)

            def __iter__(self):
                data = self._data
                if data is False:
                    return iter(self._nodes)
                if data is True:
                    return iter(self._nodes.items())
                return (
                    (n, dd[data] if data in dd else self._default)
                    for n, dd in self._nodes.items()
                )

            def __contains__(self, n):
                try:
                    node_in = n in self._nodes
                except TypeError:
                    n, d = n
                    return n in self._nodes and self[n] == d
                if node_in is True:
                    return node_in
                try:
                    n, d = n
                except (TypeError, ValueError):
                    return False
                return n in self._nodes and self[n] == d

            def __getitem__(self, n):
                if isinstance(n, slice):
                    raise Exception(
                        f"{type(self).__name__} does not support slicing, "
                        f"try list(G.nodes.data())[{n.start}:{n.stop}:{n.step}]"
                    )
                ddict = self._nodes[n]
                data = self._data
                if data is False or data is True:
                    return ddict
                return ddict[data] if data in ddict else self._default

            def __str__(self):
                return str(list(self))

            def __repr__(self):
                name = self.__class__.__name__
                if self._data is False:
                    return f"{name}({tuple(self)})"
                if self._data is True:
                    return f"{name}({dict(self)})"
                return f"{name}({dict(self)}, data={self._data!r})"


        # EdgeDataViews
        class OutEdgeDataView:
            """EdgeDataView for outward edges of DiGraph; See EdgeDataView"""

            __slots__ = (
                "_viewer",
                "_nbunch",
                "_data",
                "_default",
                "_adjdict",
                "_nodes_nbrs",
                "_report",
            )

            def __getstate__(self):
                return {
                    "viewer": self._viewer,
                    "nbunch": self._nbunch,
                    "data": self._data,
                    "default": self._default,
                }

            def __setstate__(self, state):
                self.__init__(**state)

            def __init__(self, viewer, nbunch=None, data=False, default=None):
                self._viewer = viewer
                adjdict = self._adjdict = viewer._adjdict
                if nbunch is None:
                    self._nodes_nbrs = adjdict.items
                else:
                    # dict retains order of nodes but acts like a set
                    nbunch = dict.fromkeys(viewer._graph.nbunch_iter(nbunch))
                    self._nodes_nbrs = lambda: [(n, adjdict[n]) for n in nbunch]
                self._nbunch = nbunch
                self._data = data
                self._default = default
                # Set _report based on data and default
                if data is True:
                    self._report = lambda n, nbr, dd: (n, nbr, dd)
                elif data is False:
                    self._report = lambda n, nbr, dd: (n, nbr)
                else:  # data is attribute name
                    self._report = (
                        lambda n, nbr, dd: (n, nbr, dd[data])
                        if data in dd
                        else (n, nbr, default)
                    )

            def __len__(self):
                return sum(len(nbrs) for n, nbrs in self._nodes_nbrs())

            def __iter__(self):
                return (
                    self._report(n, nbr, dd)
                    for n, nbrs in self._nodes_nbrs()
                    for nbr, dd in nbrs.items()
                )

            def __contains__(self, e):
                u, v = e[:2]
                if self._nbunch is not None and u not in self._nbunch:
                    return False  # this edge doesn't start in nbunch
                try:
                    ddict = self._adjdict[u][v]
                except KeyError:
                    return False
                return e == self._report(u, v, ddict)

            def __str__(self):
                return str(list(self))

            def __repr__(self):
                return f"{self.__class__.__name__}({list(self)})"


        class EdgeDataView(OutEdgeDataView):
            __slots__ = ()

            def __len__(self):
                return sum(1 for e in self)

            def __iter__(self):
                seen = {}
                for n, nbrs in self._nodes_nbrs():
                    for nbr, dd in nbrs.items():
                        if nbr not in seen:
                            yield self._report(n, nbr, dd)
                    seen[n] = 1
                del seen

            def __contains__(self, e):
                u, v = e[:2]
                if self._nbunch is not None and u not in self._nbunch and v not in self._nbunch:
                    return False  # this edge doesn't start and it doesn't end in nbunch
                try:
                    ddict = self._adjdict[u][v]
                except KeyError:
                    return False
                return e == self._report(u, v, ddict)


        class InEdgeDataView(OutEdgeDataView):
            """An EdgeDataView class for outward edges of DiGraph; See EdgeDataView"""

            __slots__ = ()

            def __iter__(self):
                return (
                    self._report(nbr, n, dd)
                    for n, nbrs in self._nodes_nbrs()
                    for nbr, dd in nbrs.items()
                )

            def __contains__(self, e):
                u, v = e[:2]
                if self._nbunch is not None and v not in self._nbunch:
                    return False  # this edge doesn't end in nbunch
                try:
                    ddict = self._adjdict[v][u]
                except KeyError:
                    return False
                return e == self._report(u, v, ddict)
            

        # NodeViews
        class NodeView(abc.Mapping, abc.Set):
            __slots__ = ("_nodes",)

            def __getstate__(self):
                return {"_nodes": self._nodes}

            def __setstate__(self, state):
                self._nodes = state["_nodes"]

            def __init__(self, graph):
                self._nodes = graph._node

            # abc.Mapping methods
            def __len__(self):
                return len(self._nodes)

            def __iter__(self):
                return iter(self._nodes)

            def __getitem__(self, n):
                if isinstance(n, slice):
                    raise Exception(
                        f"{type(self).__name__} does not support slicing, "
                        f"try list(G.nodes)[{n.start}:{n.stop}:{n.step}]"
                    )
                return self._nodes[n]

            # Set methods
            def __contains__(self, n):
                return n in self._nodes

            @classmethod
            def _from_iterable(cls, it):
                return set(it)

            # DataView method
            def __call__(self, data=False, default=None):
                if data is False:
                    return self
                return NodeDataView(self._nodes, data, default)

            def data(self, data=True, default=None):
                if data is False:
                    return self
                return NodeDataView(self._nodes, data, default)

            def __str__(self):
                return str(list(self))

            def __repr__(self):
                return f"{self.__class__.__name__}({tuple(self)})"
            

        # EdgeViews    have set operations and no data reported
        class OutEdgeView(abc.Set, abc.Mapping):
            """A EdgeView class for outward edges of a DiGraph"""

            __slots__ = ("_adjdict", "_graph", "_nodes_nbrs")

            def __getstate__(self):
                return {"_graph": self._graph, "_adjdict": self._adjdict}

            def __setstate__(self, state):
                self._graph = state["_graph"]
                self._adjdict = state["_adjdict"]
                self._nodes_nbrs = self._adjdict.items

            @classmethod
            def _from_iterable(cls, it):
                return set(it)

            dataview = OutEdgeDataView

            def __init__(self, G):
                self._graph = G
                self._adjdict = G._succ if hasattr(G, "succ") else G._adj
                self._nodes_nbrs = self._adjdict.items

            # Set methods
            def __len__(self):
                return sum(len(nbrs) for n, nbrs in self._nodes_nbrs())

            def __iter__(self):
                for n, nbrs in self._nodes_nbrs():
                    for nbr in nbrs:
                        yield (n, nbr)

            def __contains__(self, e):
                try:
                    u, v = e
                    return v in self._adjdict[u]
                except KeyError:
                    return False

            # abc.Mapping Methods
            def __getitem__(self, e):
                if isinstance(e, slice):
                    raise Exception(
                        f"{type(self).__name__} does not support slicing, "
                        f"try list(G.edges)[{e.start}:{e.stop}:{e.step}]"
                    )
                u, v = e
                return self._adjdict[u][v]

            # EdgeDataView methods
            def __call__(self, nbunch=None, data=False, default=None):
                if nbunch is None and data is False:
                    return self
                return self.dataview(self, nbunch, data, default)

            def data(self, data=True, default=None, nbunch=None):

                if nbunch is None and data is False:
                    return self
                return self.dataview(self, nbunch, data, default)

            # String Methods
            def __str__(self):
                return str(list(self))

            def __repr__(self):
                return f"{self.__class__.__name__}({list(self)})"


        class EdgeView(OutEdgeView):

            __slots__ = ()

            dataview = EdgeDataView

            def __len__(self):
                num_nbrs = (len(nbrs) + (n in nbrs) for n, nbrs in self._nodes_nbrs())
                return sum(num_nbrs) // 2

            def __iter__(self):
                seen = {}
                for n, nbrs in self._nodes_nbrs():
                    for nbr in list(nbrs):
                        if nbr not in seen:
                            yield (n, nbr)
                    seen[n] = 1
                del seen

            def __contains__(self, e):
                try:
                    u, v = e[:2]
                    return v in self._adjdict[u] or u in self._adjdict[v]
                except (KeyError, ValueError):
                    return False


        class InEdgeView(OutEdgeView):
            """A EdgeView class for inward edges of a DiGraph"""

            __slots__ = ()

            def __setstate__(self, state):
                self._graph = state["_graph"]
                self._adjdict = state["_adjdict"]
                self._nodes_nbrs = self._adjdict.items

            dataview = InEdgeDataView

            def __init__(self, G):
                self._graph = G
                self._adjdict = G._pred if hasattr(G, "pred") else G._adj
                self._nodes_nbrs = self._adjdict.items

            def __iter__(self):
                for n, nbrs in self._nodes_nbrs():
                    for nbr in nbrs:
                        yield (nbr, n)

            def __contains__(self, e):
                try:
                    u, v = e
                    return u in self._adjdict[v]
                except KeyError:
                    return False

            def __getitem__(self, e):
                if isinstance(e, slice):
                    raise Exception(
                        f"{type(self).__name__} does not support slicing, "
                        f"try list(G.in_edges)[{e.start}:{e.stop}:{e.step}]"
                    )
                u, v = e
                return self._adjdict[v][u]


        class _CachedPropertyResetterAdj:
            def __set__(self, obj, value):
                od = obj.__dict__
                od["_adj"] = value
                if "adj" in od:
                    del od["adj"]


        class _CachedPropertyResetterNode:
            def __set__(self, obj, value):
                od = obj.__dict__
                od["_node"] = value
                if "nodes" in od:
                    del od["nodes"]


        class Graph:

            _adj = _CachedPropertyResetterAdj()
            _node = _CachedPropertyResetterNode()

            node_dict_factory = dict
            node_attr_dict_factory = dict
            adjlist_outer_dict_factory = dict
            adjlist_inner_dict_factory = dict
            edge_attr_dict_factory = dict
            graph_attr_dict_factory = dict

            def to_undirected_class(self):
                return Graph

            def __init__(self, incoming_graph_data=None, **attr):
                self.graph = self.graph_attr_dict_factory()  # dictionary for graph attributes
                self._node = self.node_dict_factory()  # empty node attribute dict
                self._adj = self.adjlist_outer_dict_factory()  # empty adjacency dict

                # load graph attributes (must be after convert)
                self.graph.update(attr)

            @functools.cached_property
            def adj(self):
                return AdjacencyView(self._adj)

            @property
            def name(self):
                return self.graph.get("name", "")

            @name.setter
            def name(self, s):
                self.graph["name"] = s

            def __iter__(self):
                return iter(self._node)

            def __contains__(self, n):
                try:
                    return n in self._node
                except TypeError:
                    return False

            def __len__(self):
                return len(self._node)

            def __getitem__(self, n):
                return self.adj[n]

            def add_node(self, node_for_adding, **attr):
                if node_for_adding not in self._node:
                    if node_for_adding is None:
                        raise ValueError("None cannot be a node")
                    self._adj[node_for_adding] = self.adjlist_inner_dict_factory()
                    attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
                    attr_dict.update(attr)
                else:  # update attr even if node already exists
                    self._node[node_for_adding].update(attr)

            def add_nodes_from(self, nodes_for_adding, **attr):
                for n in nodes_for_adding:
                    try:
                        newnode = n not in self._node
                        newdict = attr
                    except TypeError:
                        n, ndict = n
                        newnode = n not in self._node
                        newdict = attr.copy()
                        newdict.update(ndict)
                    if newnode:
                        if n is None:
                            raise ValueError("None cannot be a node")
                        self._adj[n] = self.adjlist_inner_dict_factory()
                        self._node[n] = self.node_attr_dict_factory()
                    self._node[n].update(newdict)

            def remove_node(self, n):
                adj = self._adj
                try:
                    nbrs = list(adj[n])  # list handles self-loops (allows mutation)
                    del self._node[n]
                except KeyError as err:  # Exception if n not in self
                    raise Exception(f"The node {n} is not in the graph.") from err
                for u in nbrs:
                    del adj[u][n]  # remove all edges n-u in graph
                del adj[n]  # now remove node

            def remove_nodes_from(self, nodes):
                adj = self._adj
                for n in nodes:
                    try:
                        del self._node[n]
                        for u in list(adj[n]):  # list handles self-loops
                            del adj[u][n]  # (allows mutation of dict in loop)
                        del adj[n]
                    except KeyError:
                        pass

            @functools.cached_property
            def nodes(self):
                return NodeView(self)

            def number_of_nodes(self):
                return len(self._node)

            def order(self):
                return len(self._node)

            def has_node(self, n):
                try:
                    return n in self._node
                except TypeError:
                    return False

            def add_edge(self, u_of_edge, v_of_edge, **attr):
                u, v = u_of_edge, v_of_edge
                # add nodes
                if u not in self._node:
                    if u is None:
                        raise ValueError("None cannot be a node")
                    self._adj[u] = self.adjlist_inner_dict_factory()
                    self._node[u] = self.node_attr_dict_factory()
                if v not in self._node:
                    if v is None:
                        raise ValueError("None cannot be a node")
                    self._adj[v] = self.adjlist_inner_dict_factory()
                    self._node[v] = self.node_attr_dict_factory()
                # add the edge
                datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
                datadict.update(attr)
                self._adj[u][v] = datadict
                self._adj[v][u] = datadict

            def add_edges_from(self, ebunch_to_add, **attr):
                for e in ebunch_to_add:
                    ne = len(e)
                    if ne == 3:
                        u, v, dd = e
                    elif ne == 2:
                        u, v = e
                        dd = {}  # doesn't need edge_attr_dict_factory
                    else:
                        raise Exception(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
                    if u not in self._node:
                        if u is None:
                            raise ValueError("None cannot be a node")
                        self._adj[u] = self.adjlist_inner_dict_factory()
                        self._node[u] = self.node_attr_dict_factory()
                    if v not in self._node:
                        if v is None:
                            raise ValueError("None cannot be a node")
                        self._adj[v] = self.adjlist_inner_dict_factory()
                        self._node[v] = self.node_attr_dict_factory()
                    datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
                    datadict.update(attr)
                    datadict.update(dd)
                    self._adj[u][v] = datadict
                    self._adj[v][u] = datadict

            def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):
                self.add_edges_from(((u, v, {weight: d}) for u, v, d in ebunch_to_add), **attr)

            def remove_edge(self, u, v):
                try:
                    del self._adj[u][v]
                    if u != v:  # self-loop needs only one entry removed
                        del self._adj[v][u]
                except KeyError as err:
                    raise Exception(f"The edge {u}-{v} is not in the graph") from err

            def remove_edges_from(self, ebunch):
                adj = self._adj
                for e in ebunch:
                    u, v = e[:2]  # ignore edge data if present
                    if u in adj and v in adj[u]:
                        del adj[u][v]
                        if u != v:  # self loop needs only one entry removed
                            del adj[v][u]

            def update(self, edges=None, nodes=None):
                if edges is not None:
                    if nodes is not None:
                        self.add_nodes_from(nodes)
                        self.add_edges_from(edges)
                    else:
                        # check if edges is a Graph object
                        try:
                            graph_nodes = edges.nodes
                            graph_edges = edges.edges
                        except AttributeError:
                            # edge not Graph-like
                            self.add_edges_from(edges)
                        else:  # edges is Graph-like
                            self.add_nodes_from(graph_nodes.data())
                            self.add_edges_from(graph_edges.data())
                            self.graph.update(edges.graph)
                elif nodes is not None:
                    self.add_nodes_from(nodes)
                else:
                    raise Exception("update needs nodes or edges input")

            def has_edge(self, u, v):
                try:
                    return v in self._adj[u]
                except KeyError:
                    return False

            def neighbors(self, n):
                try:
                    return iter(self._adj[n])
                except KeyError as err:
                    raise Exception(f"The node {n} is not in the graph.") from err

            @functools.cached_property
            def edges(self):
                return EdgeView(self)

            def get_edge_data(self, u, v, default=None):
                try:
                    return self._adj[u][v]
                except KeyError:
                    return default

            def adjacency(self):
                return iter(self._adj.items())

            @functools.cached_property
            def degree(self):
                return DegreeView(self)

            def clear(self):
                self._adj.clear()
                self._node.clear()
                self.graph.clear()

            def clear_edges(self):
                for neighbours_dict in self._adj.values():
                    neighbours_dict.clear()

            def is_multigraph(self):
                """Returns True if graph is a multigraph, False otherwise."""
                return False

            def is_directed(self):
                """Returns True if graph is directed, False otherwise."""
                return False

            def size(self, weight=None):
                s = sum(d for v, d in self.degree(weight=weight))
                # If weight is None, the sum of the degrees is guaranteed to be
                # even, so we can perform integer division and hence return an
                # integer. Otherwise, the sum of the weighted degrees is not
                # guaranteed to be an integer, so we perform "real" division.
                return s // 2 if weight is None else s / 2

            def number_of_edges(self, u=None, v=None):
                if u is None:
                    return int(self.size())
                if v in self._adj[u]:
                    return 1
                return 0

            def nbunch_iter(self, nbunch=None):
                if nbunch is None:  # include all nodes via iterator
                    bunch = iter(self._adj)
                elif nbunch in self:  # if nbunch is a single node
                    bunch = iter([nbunch])
                else:  # if nbunch is a sequence of nodes

                    def bunch_iter(nlist, adj):
                        try:
                            for n in nlist:
                                if n in adj:
                                    yield n
                        except TypeError as err:
                            exc, message = err, err.args[0]
                            # capture error for non-sequence/iterator nbunch.
                            if "iter" in message:
                                exc = Exception(
                                    "nbunch is not a node or a sequence of nodes."
                                )
                            # capture error for unhashable node.
                            if "hashable" in message:
                                exc = Exception(
                                    f"ast.AST {n} in sequence nbunch is not a valid node."
                                )
                            raise exc

                    bunch = bunch_iter(nbunch, self._adj)
                return bunch

        class _CachedPropertyResetterAdjAndSucc:
            def __set__(self, obj, value):
                od = obj.__dict__
                od["_adj"] = value
                od["_succ"] = value
                # reset cached properties
                if "adj" in od:
                    del od["adj"]
                if "succ" in od:
                    del od["succ"]

        class _CachedPropertyResetterPred:
            def __set__(self, obj, value):
                od = obj.__dict__
                od["_pred"] = value
                if "pred" in od:
                    del od["pred"]

        class DiGraph(Graph):
            _adj = _CachedPropertyResetterAdjAndSucc()  # type: ignore
            _succ = _adj  # type: ignore
            _pred = _CachedPropertyResetterPred()

            def __init__(self, incoming_graph_data=None, **attr):
                self.graph = self.graph_attr_dict_factory()  # dictionary for graph attributes
                self._node = self.node_dict_factory()  # dictionary for node attr
                # We store two adjacency lists:
                # the predecessors of node n are stored in the dict self._pred
                # the successors of node n are stored in the dict self._succ=self._adj
                self._adj = self.adjlist_outer_dict_factory()  # empty adjacency dict successor
                self._pred = self.adjlist_outer_dict_factory()  # predecessor
                # Note: self._succ = self._adj  # successor

                # NOTE: I DISABLED CONVERSIONS...

                # load graph attributes (must be after convert)
                self.graph.update(attr)

            @functools.cached_property
            def adj(self):
                return AdjacencyView(self._succ)

            @functools.cached_property
            def succ(self):
                return AdjacencyView(self._succ)

            @functools.cached_property
            def pred(self):
                return AdjacencyView(self._pred)

            def add_node(self, node_for_adding, **attr):
                if node_for_adding not in self._succ:
                    if node_for_adding is None:
                        raise ValueError("None cannot be a node")
                    self._succ[node_for_adding] = self.adjlist_inner_dict_factory()
                    self._pred[node_for_adding] = self.adjlist_inner_dict_factory()
                    attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
                    attr_dict.update(attr)
                else:  # update attr even if node already exists
                    self._node[node_for_adding].update(attr)

            def add_nodes_from(self, nodes_for_adding, **attr):
                for n in nodes_for_adding:
                    try:
                        newnode = n not in self._node
                        newdict = attr
                    except TypeError:
                        n, ndict = n
                        newnode = n not in self._node
                        newdict = attr.copy()
                        newdict.update(ndict)
                    if newnode:
                        if n is None:
                            raise ValueError("None cannot be a node")
                        self._succ[n] = self.adjlist_inner_dict_factory()
                        self._pred[n] = self.adjlist_inner_dict_factory()
                        self._node[n] = self.node_attr_dict_factory()
                    self._node[n].update(newdict)

            def remove_node(self, n):
                try:
                    nbrs = self._succ[n]
                    del self._node[n]
                except KeyError as err:  # Exception if n not in self
                    raise Exception(f"The node {n} is not in the digraph.") from err
                for u in nbrs:
                    del self._pred[u][n]  # remove all edges n-u in digraph
                del self._succ[n]  # remove node from succ
                for u in self._pred[n]:
                    del self._succ[u][n]  # remove all edges n-u in digraph
                del self._pred[n]  # remove node from pred

            def remove_nodes_from(self, nodes):
                for n in nodes:
                    try:
                        succs = self._succ[n]
                        del self._node[n]
                        for u in succs:
                            del self._pred[u][n]  # remove all edges n-u in digraph
                        del self._succ[n]  # now remove node
                        for u in self._pred[n]:
                            del self._succ[u][n]  # remove all edges n-u in digraph
                        del self._pred[n]  # now remove node
                    except KeyError:
                        pass  # silent failure on remove

            def add_edge(self, u_of_edge, v_of_edge, **attr):
                u, v = u_of_edge, v_of_edge
                # add nodes
                if u not in self._succ:
                    if u is None:
                        raise ValueError("None cannot be a node")
                    self._succ[u] = self.adjlist_inner_dict_factory()
                    self._pred[u] = self.adjlist_inner_dict_factory()
                    self._node[u] = self.node_attr_dict_factory()
                if v not in self._succ:
                    if v is None:
                        raise ValueError("None cannot be a node")
                    self._succ[v] = self.adjlist_inner_dict_factory()
                    self._pred[v] = self.adjlist_inner_dict_factory()
                    self._node[v] = self.node_attr_dict_factory()
                # add the edge
                datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
                datadict.update(attr)
                self._succ[u][v] = datadict
                self._pred[v][u] = datadict

            def add_edges_from(self, ebunch_to_add, **attr):
                for e in ebunch_to_add:
                    ne = len(e)
                    if ne == 3:
                        u, v, dd = e
                    elif ne == 2:
                        u, v = e
                        dd = {}
                    else:
                        raise Exception(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
                    if u not in self._succ:
                        if u is None:
                            raise ValueError("None cannot be a node")
                        self._succ[u] = self.adjlist_inner_dict_factory()
                        self._pred[u] = self.adjlist_inner_dict_factory()
                        self._node[u] = self.node_attr_dict_factory()
                    if v not in self._succ:
                        if v is None:
                            raise ValueError("None cannot be a node")
                        self._succ[v] = self.adjlist_inner_dict_factory()
                        self._pred[v] = self.adjlist_inner_dict_factory()
                        self._node[v] = self.node_attr_dict_factory()
                    datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
                    datadict.update(attr)
                    datadict.update(dd)
                    self._succ[u][v] = datadict
                    self._pred[v][u] = datadict

            def remove_edge(self, u, v):
                try:
                    del self._succ[u][v]
                    del self._pred[v][u]
                except KeyError as err:
                    raise Exception(f"The edge {u}-{v} not in graph.") from err

            def remove_edges_from(self, ebunch):
                for e in ebunch:
                    u, v = e[:2]  # ignore edge data
                    if u in self._succ and v in self._succ[u]:
                        del self._succ[u][v]
                        del self._pred[v][u]

            def has_successor(self, u, v):
                """Returns True if node u has successor v.

                This is true if graph has the edge u->v.
                """
                return u in self._succ and v in self._succ[u]

            def has_predecessor(self, u, v):
                """Returns True if node u has predecessor v.

                This is true if graph has the edge u<-v.
                """
                return u in self._pred and v in self._pred[u]

            def successors(self, n):
                try:
                    return iter(self._succ[n])
                except KeyError as err:
                    raise Exception(f"The node {n} is not in the digraph.") from err

            # digraph definitions
            neighbors = successors

            def predecessors(self, n):
                try:
                    return iter(self._pred[n])
                except KeyError as err:
                    raise Exception(f"The node {n} is not in the digraph.") from err

            @functools.cached_property
            def edges(self):
                return OutEdgeView(self)

            # alias out_edges to edges
            @functools.cached_property
            def out_edges(self):
                return OutEdgeView(self)

            out_edges.__doc__ = edges.__doc__

            @functools.cached_property
            def in_edges(self):
                return InEdgeView(self)

            @functools.cached_property
            def degree(self):
                return DiDegreeView(self)

            @functools.cached_property
            def in_degree(self):
                return InDegreeView(self)

            @functools.cached_property
            def out_degree(self):
                return OutDegreeView(self)

            def clear(self):
                self._succ.clear()
                self._pred.clear()
                self._node.clear()
                self.graph.clear()

            def clear_edges(self):
                for predecessor_dict in self._pred.values():
                    predecessor_dict.clear()
                for successor_dict in self._succ.values():
                    successor_dict.clear()

            def is_multigraph(self):
                """Returns True if graph is a multigraph, False otherwise."""
                return False

            def is_directed(self):
                """Returns True if graph is directed, False otherwise."""
                return True



        def topological_generations(G):
            if not G.is_directed():
                raise Exception("Topological sort not defined on undirected graphs.")

            multigraph = G.is_multigraph()
            indegree_map = {v: d for v, d in G.in_degree() if d > 0}
            zero_indegree = [v for v, d in G.in_degree() if d == 0]

            while zero_indegree:
                this_generation = zero_indegree
                zero_indegree = []
                for node in this_generation:
                    if node not in G:
                        raise RuntimeError("Graph changed during iteration")
                    for child in G.neighbors(node):
                        try:
                            indegree_map[child] -= len(G[node][child]) if multigraph else 1
                        except KeyError as err:
                            raise RuntimeError("Graph changed during iteration") from err
                        if indegree_map[child] == 0:
                            zero_indegree.append(child)
                            del indegree_map[child]
                yield this_generation

            if indegree_map:
                raise Exception(
                    "Graph contains a cycle or graph changed during iteration"
                )


        def topological_sort(G):
            """Returns a generator of nodes in topologically sorted order.
            """
            for generation in topological_generations(G):
                yield from generation


        from collections import deque

        def generic_bfs_edges(G, source, neighbors=None, depth_limit=None, sort_neighbors=None):
            if callable(sort_neighbors):
                _neighbors = neighbors
                neighbors = lambda node: iter(sort_neighbors(_neighbors(node)))

            visited = {source}
            if depth_limit is None:
                depth_limit = len(G)
            queue = deque([(source, depth_limit, neighbors(source))])
            while queue:
                parent, depth_now, children = queue[0]
                try:
                    child = next(children)
                    if child not in visited:
                        yield parent, child
                        visited.add(child)
                        if depth_now > 1:
                            queue.append((child, depth_now - 1, neighbors(child)))
                except StopIteration:
                    queue.popleft()


        def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
            if reverse and G.is_directed():
                successors = G.predecessors
            else:
                successors = G.neighbors
            yield from generic_bfs_edges(G, source, successors, depth_limit, sort_neighbors)

        def ancestors(G, source):
            return {child for parent, child in bfs_edges(G, source, reverse=True)}

        def descendants(G, source):
            return {child for parent, child in bfs_edges(G, source)}
        
        def _bidirectional_pred_succ(G, source, target):
            """Bidirectional shortest path helper.

            Returns (pred, succ, w) where
            pred is a dictionary of predecessors from w to the source, and
            succ is a dictionary of successors from w to the target.
            """
            # does BFS from both source and target and meets in the middle
            if target == source:
                return ({target: None}, {source: None}, source)

            # handle either directed or undirected
            if G.is_directed():
                Gpred = G.pred
                Gsucc = G.succ
            else:
                Gpred = G.adj
                Gsucc = G.adj

            # predecesssor and successors in search
            pred = {source: None}
            succ = {target: None}

            # initialize fringes, start with forward
            forward_fringe = [source]
            reverse_fringe = [target]

            while forward_fringe and reverse_fringe:
                if len(forward_fringe) <= len(reverse_fringe):
                    this_level = forward_fringe
                    forward_fringe = []
                    for v in this_level:
                        for w in Gsucc[v]:
                            if w not in pred:
                                forward_fringe.append(w)
                                pred[w] = v
                            if w in succ:  # path found
                                return pred, succ, w
                else:
                    this_level = reverse_fringe
                    reverse_fringe = []
                    for v in this_level:
                        for w in Gpred[v]:
                            if w not in succ:
                                succ[w] = v
                                reverse_fringe.append(w)
                            if w in pred:  # found path
                                return pred, succ, w

            raise Exception(f"No path between {source} and {target}.")

        def bidirectional_shortest_path(G, source, target):

            if source not in G or target not in G:
                msg = f"Either source {source} or target {target} is not in G"
                raise Exception(msg)

            # call helper to do the real work
            results = _bidirectional_pred_succ(G, source, target)
            pred, succ, w = results

            # build path from pred+w+succ
            path = []
            # from source to w
            while w is not None:
                path.append(w)
                w = pred[w]
            path.reverse()
            # from w to target
            w = succ[path[-1]]
            while w is not None:
                path.append(w)
                w = succ[w]

            return path

        def has_path(G, source, target):
            try:
                bidirectional_shortest_path(G, source, target)
            except Exception as e:
                return False
            return True


        return DiGraph, topological_sort, has_path

reactive_python_dag_builder_utils__ = ReactivePythonDagBuilderUtils__()



`;