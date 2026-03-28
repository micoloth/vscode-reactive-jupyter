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

        # ==========================================================================================================
        # HELPER FUNCTIONS FOR MERGING EXPOSED VARIABLES
        # ==========================================================================================================

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
            return v_merge(*stmts, _class=_class)

        # ==========================================================================================================
        # ITERATIVE (NON-RECURSIVE) AST VISITOR IMPLEMENTATION
        # ==========================================================================================================
        # Instead of creating child TempScopeVisitor instances recursively, we use an explicit work stack.
        # Work items are processed in LIFO order. Two types of work:
        # 1. VISIT: Visit a node, store result in a slot
        # 2. CONTINUATION: Post-process collected results (merge, transform, etc.)
        # ==========================================================================================================

        # Work item type constants
        WORK_VISIT = 0          # Visit a node and store result in slot
        WORK_CONTINUATION = 1   # Run a continuation to process results

        # Continuation type constants (what post-processing to do)
        CONT_MERGE_VMERGE = 'v_merge'
        CONT_MERGE_HMERGE = 'h_merge'
        CONT_ASSIGN = 'assign'
        CONT_AUGASSIGN = 'augassign'
        CONT_FUNCTION = 'function'
        CONT_COMPREHENSION = 'comprehension'
        CONT_FOR = 'for'
        CONT_WITH = 'with'
        CONT_IF_WHILE = 'if_while'
        CONT_TRY = 'try'
        CONT_TRY_HANDLER = 'try_handler'
        CONT_MATCH_CASE = 'match_case'
        CONT_CLASS = 'class'
        CONT_ARGUMENTS = 'arguments'

        @dataclasses.dataclass
        class VisitorFrame:
            """A frame holding visitor state and result slots for collecting child results."""
            variables: ExposedVariables
            is_lhs_target: bool = False
            is_also_input_of_aug_assign: bool = False
            _class: bool = False
            # Result slots for collecting child visit results
            result_slots: List[Optional[ExposedVariables]] = dataclasses.field(default_factory=list)

        class IterativeScopeVisitor:
            """
            Non-recursive AST visitor that uses an explicit work stack instead of recursion.
            Replaces the recursive TempScopeVisitor pattern.
            """

            def __init__(self):
                # Work stack: list of (work_type, *work_data)
                self.work_stack: List[Tuple] = []
                # Frame stack: each frame holds context for a level of visiting
                self.frame_stack: List[VisitorFrame] = []

            def annotate(self, tree: ast.AST) -> ExposedVariables:
                """Main entry point - iteratively computes variables for an AST."""
                # Initialize root frame
                root_frame = VisitorFrame(ExposedVariables())
                self.frame_stack = [root_frame]
                
                # Push initial work: visit the tree
                self._push_visit(tree, frame_idx=0, slot_idx=None)
                
                # Process work stack until empty
                self._process_work_stack()
                
                return root_frame.variables

            def _push_visit(self, node: ast.AST, frame_idx: int, slot_idx: Optional[int],
                           is_lhs_target: bool = False, is_also_input_of_aug_assign: bool = False,
                           _class: bool = False, use_frame_flags: bool = False):
                """Push a visit work item onto the stack."""
                if node is None:
                    return
                if use_frame_flags:
                    frame = self.frame_stack[frame_idx]
                    is_lhs_target = frame.is_lhs_target
                    is_also_input_of_aug_assign = frame.is_also_input_of_aug_assign
                    _class = frame._class
                self.work_stack.append((WORK_VISIT, node, frame_idx, slot_idx,
                                       is_lhs_target, is_also_input_of_aug_assign, _class))

            def _push_visit_all(self, nodes: List[ast.AST], frame_idx: int,
                               is_lhs_target: bool = False, is_also_input_of_aug_assign: bool = False,
                               _class: bool = False, use_frame_flags: bool = False):
                """Push visit work items for all nodes (they share the frame's variables)."""
                for node in reversed(nodes):  # Reversed so first node is processed first
                    if node is not None:
                        if isinstance(node, list):
                            self._push_visit_all(node, frame_idx, is_lhs_target, is_also_input_of_aug_assign, _class, use_frame_flags)
                        else:
                            self._push_visit(node, frame_idx, slot_idx=None,
                                           is_lhs_target=is_lhs_target, is_also_input_of_aug_assign=is_also_input_of_aug_assign,
                                           _class=_class, use_frame_flags=use_frame_flags)

            def _push_continuation(self, cont_type: str, frame_idx: int, extra_data: Any = None):
                """Push a continuation work item onto the stack."""
                self.work_stack.append((WORK_CONTINUATION, cont_type, frame_idx, extra_data))

            def _create_child_frame(self, parent_frame_idx: int, fresh_variables: bool = True,
                                   is_lhs_target: bool = None, is_also_input_of_aug_assign: bool = None,
                                   _class: bool = None) -> int:
                """Create a new frame for child visits. Returns the frame index."""
                parent = self.frame_stack[parent_frame_idx]
                new_frame = VisitorFrame(
                    variables=ExposedVariables() if fresh_variables else parent.variables,
                    is_lhs_target=is_lhs_target if is_lhs_target is not None else parent.is_lhs_target,
                    is_also_input_of_aug_assign=is_also_input_of_aug_assign if is_also_input_of_aug_assign is not None else parent.is_also_input_of_aug_assign,
                    _class=_class if _class is not None else parent._class,
                )
                self.frame_stack.append(new_frame)
                return len(self.frame_stack) - 1

            def _allocate_slots(self, frame_idx: int, count: int) -> int:
                """Allocate result slots in a frame. Returns the starting slot index."""
                frame = self.frame_stack[frame_idx]
                start_idx = len(frame.result_slots)
                frame.result_slots.extend([None] * count)
                return start_idx

            def _get_slots(self, frame_idx: int, start_idx: int, count: int) -> List[ExposedVariables]:
                """Get results from slots."""
                frame = self.frame_stack[frame_idx]
                return [frame.result_slots[start_idx + i] or ExposedVariables() for i in range(count)]

            def _process_work_stack(self):
                """Main loop - process work items until stack is empty."""
                while self.work_stack:
                    work_item = self.work_stack.pop()
                    work_type = work_item[0]
                    
                    if work_type == WORK_VISIT:
                        _, node, frame_idx, slot_idx, is_lhs_target, is_also_input_of_aug_assign, _class = work_item
                        self._handle_visit(node, frame_idx, slot_idx, is_lhs_target, is_also_input_of_aug_assign, _class)
                    elif work_type == WORK_CONTINUATION:
                        _, cont_type, frame_idx, extra_data = work_item
                        self._handle_continuation(cont_type, frame_idx, extra_data)

            def _handle_visit(self, node: ast.AST, frame_idx: int, slot_idx: Optional[int],
                             is_lhs_target: bool, is_also_input_of_aug_assign: bool, _class: bool):
                """Handle visiting a single node."""
                frame = self.frame_stack[frame_idx]
                
                # If slot_idx is not None, we need to create a fresh frame for this visit
                if slot_idx is not None:
                    child_frame_idx = self._create_child_frame(
                        frame_idx, fresh_variables=True,
                        is_lhs_target=is_lhs_target, is_also_input_of_aug_assign=is_also_input_of_aug_assign,
                        _class=_class
                    )
                    # After visiting, store result in parent's slot
                    self.work_stack.append(('store_result', frame_idx, slot_idx, child_frame_idx))
                    # Visit in new frame
                    self._dispatch_node(node, child_frame_idx, is_lhs_target, is_also_input_of_aug_assign, _class)
                else:
                    # Visit directly in current frame
                    self._dispatch_node(node, frame_idx, is_lhs_target, is_also_input_of_aug_assign, _class)

            def _dispatch_node(self, node: ast.AST, frame_idx: int, is_lhs_target: bool,
                              is_also_input_of_aug_assign: bool, _class: bool):
                """Dispatch to the appropriate handler based on node type."""
                frame = self.frame_stack[frame_idx]
                # Update frame flags
                frame.is_lhs_target = is_lhs_target
                frame.is_also_input_of_aug_assign = is_also_input_of_aug_assign
                frame._class = _class
                
                node_type = type(node).__name__
                handler = getattr(self, f'_handle_{node_type}', None)
                if handler:
                    handler(node, frame_idx)
                else:
                    # Default: generic visit - visit all child nodes
                    self._handle_generic(node, frame_idx)

            def _handle_generic(self, node: ast.AST, frame_idx: int):
                """Default handler - visit all child nodes in AST order (left-to-right).
                
                IMPORTANT: Children must be pushed in REVERSE order because the work stack
                is LIFO (last-in-first-out). This ensures children are processed in the
                correct left-to-right order as in the original recursive visitor.
                """
                children = list(ast.iter_child_nodes(node))
                for child in reversed(children):
                    self._push_visit(child, frame_idx, slot_idx=None, use_frame_flags=True)

            # ==========================================================================================================
            # NODE HANDLERS - each handles a specific AST node type
            # ==========================================================================================================

            def _handle_Name(self, node: ast.Name, frame_idx: int):
                """Handle Name nodes."""
                frame = self.frame_stack[frame_idx]
                is_input = type(node.ctx) is ast.Load or frame.is_also_input_of_aug_assign or type(node.ctx) is ast.Del
                is_output = frame.is_lhs_target or type(node.ctx) is ast.Store
                
                if is_input and node.id not in frame.variables.output_variables:
                    frame.variables.input_variables.add(node.id)
                if is_output:
                    frame.variables.output_variables.add(node.id)

            def _handle_Subscript(self, node: ast.Subscript, frame_idx: int):
                """Handle Subscript nodes."""
                frame = self.frame_stack[frame_idx]
                if type(node.ctx) in [ast.Store, ast.Del] or frame.is_lhs_target:
                    self._push_visit(node.value, frame_idx, slot_idx=None,
                                    is_lhs_target=True, is_also_input_of_aug_assign=frame.is_also_input_of_aug_assign,
                                    _class=frame._class)
                elif type(node.ctx) is ast.Load:
                    self._push_visit(node.value, frame_idx, slot_idx=None, use_frame_flags=True)
                else:
                    raise RuntimeError(f"Unsupported node type: {node}")
                # Always visit the slice with Load semantics
                self._push_visit(node.slice, frame_idx, slot_idx=None,
                                is_lhs_target=False, is_also_input_of_aug_assign=False, _class=frame._class)

            def _handle_Attribute(self, node: ast.Attribute, frame_idx: int):
                """Handle Attribute nodes."""
                frame = self.frame_stack[frame_idx]
                if type(node.ctx) in [ast.Store, ast.Del] or frame.is_lhs_target:
                    self._push_visit(node.value, frame_idx, slot_idx=None,
                                    is_lhs_target=True, is_also_input_of_aug_assign=frame.is_also_input_of_aug_assign,
                                    _class=frame._class)
                elif type(node.ctx) is ast.Load:
                    self._push_visit(node.value, frame_idx, slot_idx=None, use_frame_flags=True)
                else:
                    raise RuntimeError(f"Unsupported node type: {node}")

            def _handle_Assign(self, node: ast.Assign, frame_idx: int):
                """Handle Assign nodes."""
                frame = self.frame_stack[frame_idx]
                # Allocate slots: 1 for value, N for targets
                num_targets = len(node.targets)
                slot_start = self._allocate_slots(frame_idx, 1 + num_targets)
                
                # Push continuation first (runs after all visits complete)
                self._push_continuation(CONT_ASSIGN, frame_idx, (slot_start, num_targets))
                
                # Push visits (in reverse order so value is visited first)
                for i, target in enumerate(reversed(node.targets)):
                    self._push_visit(target, frame_idx, slot_idx=slot_start + num_targets - i,
                                    is_lhs_target=frame.is_lhs_target, is_also_input_of_aug_assign=False,
                                    _class=frame._class)
                self._push_visit(node.value, frame_idx, slot_idx=slot_start,
                                is_lhs_target=frame.is_lhs_target, is_also_input_of_aug_assign=False,
                                _class=frame._class)

            def _handle_AugAssign(self, node: ast.AugAssign, frame_idx: int):
                """Handle AugAssign nodes (+=, -=, etc.)."""
                frame = self.frame_stack[frame_idx]
                # Visit target with augassign semantics (it's both input and output)
                if type(node.target.ctx) in [ast.Store, ast.Del] or frame.is_lhs_target:
                    self._push_visit(node.target, frame_idx, slot_idx=None,
                                    is_lhs_target=True, is_also_input_of_aug_assign=True, _class=frame._class)
                elif type(node.target.ctx) is ast.Load:
                    self._push_visit(node.target, frame_idx, slot_idx=None, use_frame_flags=True)
                else:
                    raise RuntimeError(f"Unsupported node type: {node}")
                # Visit value
                self._push_visit(node.value, frame_idx, slot_idx=None,
                                is_lhs_target=False, is_also_input_of_aug_assign=False, _class=frame._class)

            def _handle_List(self, node: ast.List, frame_idx: int):
                """Handle List nodes."""
                frame = self.frame_stack[frame_idx]
                if type(node.ctx) in [ast.Store, ast.Del] or frame.is_lhs_target:
                    self._push_visit_all(node.elts, frame_idx,
                                        is_lhs_target=True, is_also_input_of_aug_assign=frame.is_also_input_of_aug_assign,
                                        _class=frame._class)
                elif type(node.ctx) is ast.Load:
                    self._handle_generic(node, frame_idx)
                else:
                    raise RuntimeError(f"Unsupported node type: {node}")

            def _handle_Tuple(self, node: ast.Tuple, frame_idx: int):
                """Handle Tuple nodes."""
                frame = self.frame_stack[frame_idx]
                if type(node.ctx) in [ast.Store, ast.Del] or frame.is_lhs_target:
                    self._push_visit_all(node.elts, frame_idx,
                                        is_lhs_target=True, is_also_input_of_aug_assign=frame.is_also_input_of_aug_assign,
                                        _class=frame._class)
                elif type(node.ctx) is ast.Load:
                    self._handle_generic(node, frame_idx)
                else:
                    raise RuntimeError(f"Unsupported node type: {node}")

            def _handle_alias(self, node: ast.alias, frame_idx: int):
                """Handle alias nodes (for imports)."""
                frame = self.frame_stack[frame_idx]
                variable = node.asname if node.asname is not None else node.name
                frame.variables.output_variables.add(variable)

            def _handle_arg(self, node: ast.arg, frame_idx: int):
                """Handle arg nodes (function arguments)."""
                frame = self.frame_stack[frame_idx]
                frame.variables.introduced_variables.add(node.arg)

            def _handle_FunctionDef(self, node: ast.FunctionDef, frame_idx: int):
                """Handle FunctionDef nodes."""
                self._handle_function_common(node, frame_idx, func_name=node.name)

            def _handle_AsyncFunctionDef(self, node: ast.AsyncFunctionDef, frame_idx: int):
                """Handle AsyncFunctionDef nodes."""
                self._handle_function_common(node, frame_idx, func_name=node.name)

            def _handle_Lambda(self, node: ast.Lambda, frame_idx: int):
                """Handle Lambda nodes."""
                self._handle_function_common(node, frame_idx, func_name=None)

            def _handle_function_common(self, node, frame_idx: int, func_name: Optional[str]):
                """Common handler for all function-like nodes."""
                frame = self.frame_stack[frame_idx]
                
                # Slots: decorators/returns (slot 0), arguments (slot 1), body (slot 2)
                slot_start = self._allocate_slots(frame_idx, 3)
                
                # Push continuation
                self._push_continuation(CONT_FUNCTION, frame_idx, (slot_start, func_name, frame._class))
                
                # Push body visit (in a new frame with _class=False)
                if isinstance(node.body, list):
                    # FunctionDef/AsyncFunctionDef: body is a list of statements
                    body_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=False)
                    self.work_stack.append(('collect_body', frame_idx, slot_start + 2, body_frame_idx, len(node.body)))
                    for stmt in reversed(node.body):
                        self._push_visit(stmt, body_frame_idx, slot_idx=None, use_frame_flags=True)
                else:
                    # Lambda: body is a single expression
                    self._push_visit(node.body, frame_idx, slot_idx=slot_start + 2, _class=False)
                
                # Push arguments visit
                args_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True)
                self.work_stack.append(('store_result', frame_idx, slot_start + 1, args_frame_idx))
                self._push_arguments_visit(node.args, frame_idx, args_frame_idx)
                
                # Push decorators/type hints visit (only for non-Lambda)
                if not isinstance(node, ast.Lambda):
                    decorators_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True)
                    self.work_stack.append(('store_result', frame_idx, slot_start, decorators_frame_idx))
                    if hasattr(node, 'type_comment') and node.type_comment:
                        pass  # Type comments are strings, not AST nodes
                    self._push_visit_all(node.decorator_list, decorators_frame_idx, use_frame_flags=True)
                    if node.returns:
                        self._push_visit(node.returns, decorators_frame_idx, slot_idx=None, use_frame_flags=True)
                else:
                    # For Lambda, set slot 0 to empty
                    frame.result_slots[slot_start] = ExposedVariables()

            def _push_arguments_visit(self, args: ast.arguments, parent_frame_idx: int, args_frame_idx: int):
                """Visit function arguments, separating arg names from default value expressions."""
                # args_frame collects introduced_variables for argument names
                # parent_frame gets inputs from default values
                args_frame = self.frame_stack[args_frame_idx]
                
                # Collect all argument names as introduced variables
                all_args = (
                    (args.posonlyargs or []) + 
                    args.args + 
                    ([args.vararg] if args.vararg else []) + 
                    args.kwonlyargs + 
                    ([args.kwarg] if args.kwarg else [])
                )
                for arg in all_args:
                    args_frame.variables.introduced_variables.add(arg.arg)
                    # Visit annotations in parent frame (they're inputs)
                    if arg.annotation:
                        self._push_visit(arg.annotation, parent_frame_idx, slot_idx=None, use_frame_flags=True)
                
                # Visit default values in parent frame (they're inputs)
                for default in (args.defaults or []) + (args.kw_defaults or []):
                    if default:
                        self._push_visit(default, parent_frame_idx, slot_idx=None, use_frame_flags=True)

            def _handle_ListComp(self, node: ast.ListComp, frame_idx: int):
                """Handle ListComp nodes."""
                self._handle_comprehension_common([node.elt], node.generators, frame_idx)

            def _handle_SetComp(self, node: ast.SetComp, frame_idx: int):
                """Handle SetComp nodes."""
                self._handle_comprehension_common([node.elt], node.generators, frame_idx)

            def _handle_GeneratorExp(self, node: ast.GeneratorExp, frame_idx: int):
                """Handle GeneratorExp nodes."""
                self._handle_comprehension_common([node.elt], node.generators, frame_idx)

            def _handle_DictComp(self, node: ast.DictComp, frame_idx: int):
                """Handle DictComp nodes."""
                self._handle_comprehension_common([node.key, node.value], node.generators, frame_idx)

            def _handle_comprehension_common(self, targets: List[ast.AST], generators: List[ast.comprehension], frame_idx: int):
                """Common handler for comprehensions."""
                frame = self.frame_stack[frame_idx]
                
                # Count slots needed: for each generator (target, iter, ifs...), plus targets
                total_slots = 0
                gen_info = []  # (slot_start, num_ifs) for each generator
                for gen in generators:
                    gen_slot_start = total_slots
                    num_ifs = len(gen.ifs)
                    total_slots += 2 + num_ifs  # target, iter, ifs
                    gen_info.append((gen_slot_start, num_ifs))
                targets_slot_start = total_slots
                total_slots += len(targets)
                
                slot_start = self._allocate_slots(frame_idx, total_slots)
                
                # Push continuation
                self._push_continuation(CONT_COMPREHENSION, frame_idx, (slot_start, gen_info, len(targets)))
                
                # Push target visits
                for i, target in enumerate(reversed(targets)):
                    self._push_visit(target, frame_idx, slot_idx=slot_start + targets_slot_start + len(targets) - 1 - i,
                                    is_lhs_target=frame.is_lhs_target, _class=frame._class)
                
                # Push generator visits (in reverse order)
                for gen_idx, gen in enumerate(reversed(generators)):
                    actual_gen_idx = len(generators) - 1 - gen_idx
                    gen_slot_start, num_ifs = gen_info[actual_gen_idx]
                    
                    # ifs
                    for if_idx, if_clause in enumerate(reversed(gen.ifs)):
                        self._push_visit(if_clause, frame_idx, 
                                        slot_idx=slot_start + gen_slot_start + 2 + num_ifs - 1 - if_idx,
                                        is_lhs_target=frame.is_lhs_target, _class=frame._class)
                    # iter
                    self._push_visit(gen.iter, frame_idx, slot_idx=slot_start + gen_slot_start + 1,
                                    is_lhs_target=frame.is_lhs_target, _class=frame._class)
                    # target
                    self._push_visit(gen.target, frame_idx, slot_idx=slot_start + gen_slot_start,
                                    is_lhs_target=frame.is_lhs_target, _class=frame._class)

            def _handle_For(self, node: ast.For, frame_idx: int):
                """Handle For nodes."""
                self._handle_for_common(node, frame_idx)

            def _handle_AsyncFor(self, node: ast.AsyncFor, frame_idx: int):
                """Handle AsyncFor nodes."""
                self._handle_for_common(node, frame_idx)

            def _handle_for_common(self, node, frame_idx: int):
                """Common handler for For and AsyncFor."""
                frame = self.frame_stack[frame_idx]
                
                # Slots: iter (0), target (1), body (2), orelse (3)
                slot_start = self._allocate_slots(frame_idx, 4)
                
                # Push continuation
                self._push_continuation(CONT_FOR, frame_idx, (slot_start, frame._class))
                
                # orelse - visit as body
                orelse_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                self.work_stack.append(('collect_body', frame_idx, slot_start + 3, orelse_frame_idx, 2 + len(node.orelse)))
                for stmt in reversed(node.orelse):
                    self._push_visit(stmt, orelse_frame_idx, slot_idx=None, use_frame_flags=True)
                self._push_visit(node.iter, orelse_frame_idx, slot_idx=None, use_frame_flags=True)
                self._push_visit(node.target, orelse_frame_idx, slot_idx=None, use_frame_flags=True)
                
                # body - visit as body
                body_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                self.work_stack.append(('collect_body', frame_idx, slot_start + 2, body_frame_idx, 2 + len(node.body)))
                for stmt in reversed(node.body):
                    self._push_visit(stmt, body_frame_idx, slot_idx=None, use_frame_flags=True)
                self._push_visit(node.iter, body_frame_idx, slot_idx=None, use_frame_flags=True)
                self._push_visit(node.target, body_frame_idx, slot_idx=None, use_frame_flags=True)
                
                # target and iter
                self._push_visit(node.target, frame_idx, slot_idx=slot_start + 1,
                                is_lhs_target=frame.is_lhs_target, _class=frame._class)
                self._push_visit(node.iter, frame_idx, slot_idx=slot_start,
                                is_lhs_target=frame.is_lhs_target, _class=frame._class)

            def _handle_With(self, node: ast.With, frame_idx: int):
                """Handle With nodes."""
                self._handle_with_common(node, frame_idx)

            def _handle_AsyncWith(self, node: ast.AsyncWith, frame_idx: int):
                """Handle AsyncWith nodes."""
                self._handle_with_common(node, frame_idx)

            def _handle_with_common(self, node, frame_idx: int):
                """Common handler for With and AsyncWith."""
                frame = self.frame_stack[frame_idx]
                
                # Slots: body (0), items (1 per item)
                num_items = len(node.items)
                slot_start = self._allocate_slots(frame_idx, 1 + num_items)
                
                # Push continuation
                self._push_continuation(CONT_WITH, frame_idx, (slot_start, num_items, frame._class))
                
                # Push item visits (in reverse)
                for i, item in enumerate(reversed(node.items)):
                    item_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                    actual_i = num_items - 1 - i
                    self.work_stack.append(('store_result', frame_idx, slot_start + 1 + actual_i, item_frame_idx))
                    self._push_visit(item.context_expr, item_frame_idx, slot_idx=None, use_frame_flags=True)
                    if item.optional_vars:
                        self._push_visit(item.optional_vars, item_frame_idx, slot_idx=None, use_frame_flags=True)
                
                # Push body visit
                body_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                self.work_stack.append(('collect_body', frame_idx, slot_start, body_frame_idx, len(node.body)))
                for stmt in reversed(node.body):
                    self._push_visit(stmt, body_frame_idx, slot_idx=None, use_frame_flags=True)

            def _handle_If(self, node: ast.If, frame_idx: int):
                """Handle If nodes."""
                self._handle_if_while_common(node, frame_idx)

            def _handle_While(self, node: ast.While, frame_idx: int):
                """Handle While nodes."""
                self._handle_if_while_common(node, frame_idx)

            def _handle_if_while_common(self, node, frame_idx: int):
                """Common handler for If and While."""
                frame = self.frame_stack[frame_idx]
                
                # Slots: test (0), body (1), orelse (2)
                slot_start = self._allocate_slots(frame_idx, 3)
                
                # Push continuation
                self._push_continuation(CONT_IF_WHILE, frame_idx, (slot_start, frame._class))
                
                # orelse
                orelse_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                self.work_stack.append(('collect_body', frame_idx, slot_start + 2, orelse_frame_idx, len(node.orelse)))
                for stmt in reversed(node.orelse):
                    self._push_visit(stmt, orelse_frame_idx, slot_idx=None, use_frame_flags=True)
                
                # body
                body_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                self.work_stack.append(('collect_body', frame_idx, slot_start + 1, body_frame_idx, len(node.body)))
                for stmt in reversed(node.body):
                    self._push_visit(stmt, body_frame_idx, slot_idx=None, use_frame_flags=True)
                
                # test
                self._push_visit(node.test, frame_idx, slot_idx=slot_start, _class=frame._class)

            def _handle_Try(self, node: ast.Try, frame_idx: int):
                """Handle Try nodes."""
                self._handle_try_common(node, frame_idx)

            def _handle_TryStar(self, node, frame_idx: int):
                """Handle TryStar nodes (Python 3.11+)."""
                self._handle_try_common(node, frame_idx)

            def _handle_try_common(self, node, frame_idx: int):
                """Common handler for Try and TryStar."""
                frame = self.frame_stack[frame_idx]
                
                # Slots: body (0), orelse (1), finalbody (2), handlers (3+)
                num_handlers = len(node.handlers)
                slot_start = self._allocate_slots(frame_idx, 3 + num_handlers)
                
                # Push continuation
                handler_names = [h.name for h in node.handlers]
                self._push_continuation(CONT_TRY, frame_idx, (slot_start, num_handlers, handler_names, frame._class))
                
                # Handlers (in reverse)
                for i, handler in enumerate(reversed(node.handlers)):
                    actual_i = num_handlers - 1 - i
                    handler_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                    self.work_stack.append(('collect_try_handler', frame_idx, slot_start + 3 + actual_i, 
                                           handler_frame_idx, handler.name, len(handler.body)))
                    # Visit handler.type in handler frame
                    if handler.type:
                        self._push_visit(handler.type, handler_frame_idx, slot_idx=None, use_frame_flags=True)
                    # Visit handler body
                    for stmt in reversed(handler.body):
                        self._push_visit(stmt, handler_frame_idx, slot_idx=None, use_frame_flags=True)
                
                # finalbody
                finalbody_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                self.work_stack.append(('collect_body', frame_idx, slot_start + 2, finalbody_frame_idx, len(node.finalbody)))
                for stmt in reversed(node.finalbody):
                    self._push_visit(stmt, finalbody_frame_idx, slot_idx=None, use_frame_flags=True)
                
                # orelse
                orelse_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                self.work_stack.append(('collect_body', frame_idx, slot_start + 1, orelse_frame_idx, len(node.orelse)))
                for stmt in reversed(node.orelse):
                    self._push_visit(stmt, orelse_frame_idx, slot_idx=None, use_frame_flags=True)
                
                # body
                body_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                self.work_stack.append(('collect_body', frame_idx, slot_start, body_frame_idx, len(node.body)))
                for stmt in reversed(node.body):
                    self._push_visit(stmt, body_frame_idx, slot_idx=None, use_frame_flags=True)

            def _handle_Match(self, node, frame_idx: int):
                """Handle Match nodes (Python 3.10+)."""
                frame = self.frame_stack[frame_idx]
                
                # Visit subject
                self._push_visit(node.subject, frame_idx, slot_idx=None, use_frame_flags=True)
                
                # Visit each case
                for case in node.cases:
                    # Get pattern captures
                    pattern_outputs = self._get_pattern_captures(case.pattern, frame_idx)
                    frame.variables.output_variables |= pattern_outputs
                    
                    # Visit guard if present
                    if case.guard:
                        self._push_visit(case.guard, frame_idx, slot_idx=None, use_frame_flags=True)
                    
                    # Visit case body - need to merge results properly
                    slot_start = self._allocate_slots(frame_idx, 1)
                    self._push_continuation(CONT_MATCH_CASE, frame_idx, (slot_start, frame._class))
                    
                    body_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                    self.work_stack.append(('collect_body', frame_idx, slot_start, body_frame_idx, len(case.body)))
                    for stmt in reversed(case.body):
                        self._push_visit(stmt, body_frame_idx, slot_idx=None, use_frame_flags=True)

            def _get_pattern_captures(self, pattern, frame_idx: int) -> Set[str]:
                """Extract captured variable names from a match pattern (non-recursive)."""
                captures = set()
                if pattern is None:
                    return captures
                
                # Use a stack to process patterns iteratively
                pattern_stack = [pattern]
                frame = self.frame_stack[frame_idx]
                
                while pattern_stack:
                    pat = pattern_stack.pop()
                    if pat is None:
                        continue
                    
                    # MatchAs: "case x:" or "case _ as x:"
                    if hasattr(ast, 'MatchAs') and isinstance(pat, ast.MatchAs):
                        if pat.name:
                            captures.add(pat.name)
                        if pat.pattern:
                            pattern_stack.append(pat.pattern)
                    # MatchStar: "case [*rest]:"
                    elif hasattr(ast, 'MatchStar') and isinstance(pat, ast.MatchStar):
                        if pat.name:
                            captures.add(pat.name)
                    # MatchMapping: "case {**rest}:"
                    elif hasattr(ast, 'MatchMapping') and isinstance(pat, ast.MatchMapping):
                        if pat.rest:
                            captures.add(pat.rest)
                        for val in pat.patterns:
                            pattern_stack.append(val)
                    # MatchSequence: "case [a, b, c]:"
                    elif hasattr(ast, 'MatchSequence') and isinstance(pat, ast.MatchSequence):
                        for p in pat.patterns:
                            pattern_stack.append(p)
                    # MatchOr: "case a | b:"
                    elif hasattr(ast, 'MatchOr') and isinstance(pat, ast.MatchOr):
                        for p in pat.patterns:
                            pattern_stack.append(p)
                    # MatchClass: "case Point(x=a, y=b):"
                    elif hasattr(ast, 'MatchClass') and isinstance(pat, ast.MatchClass):
                        # The cls is an input (we need to visit it)
                        self._push_visit(pat.cls, frame_idx, slot_idx=None, use_frame_flags=True)
                        for p in pat.patterns:
                            pattern_stack.append(p)
                        for p in pat.kwd_patterns:
                            pattern_stack.append(p)
                    # MatchValue: has inputs but no captures
                    elif hasattr(ast, 'MatchValue') and isinstance(pat, ast.MatchValue):
                        self._push_visit(pat.value, frame_idx, slot_idx=None, use_frame_flags=True)
                
                return captures

            def _handle_ClassDef(self, node: ast.ClassDef, frame_idx: int):
                """Handle ClassDef nodes."""
                frame = self.frame_stack[frame_idx]
                
                # Slots: decorators/bases (0), body (1)
                slot_start = self._allocate_slots(frame_idx, 2)
                
                # Push continuation
                self._push_continuation(CONT_CLASS, frame_idx, (slot_start, node.name))
                
                # Body with _class=True
                body_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=True)
                self.work_stack.append(('collect_body', frame_idx, slot_start + 1, body_frame_idx, len(node.body)))
                for stmt in reversed(node.body):
                    self._push_visit(stmt, body_frame_idx, slot_idx=None, use_frame_flags=True)
                
                # Decorators, bases, keywords in parent scope
                deco_frame_idx = self._create_child_frame(frame_idx, fresh_variables=True, _class=frame._class)
                self.work_stack.append(('store_result', frame_idx, slot_start, deco_frame_idx))
                self._push_visit_all(node.bases, deco_frame_idx, use_frame_flags=True)
                self._push_visit_all(node.keywords, deco_frame_idx, use_frame_flags=True)
                self._push_visit_all(node.decorator_list, deco_frame_idx, use_frame_flags=True)
                if hasattr(node, 'type_params') and node.type_params:
                    self._push_visit_all(node.type_params, deco_frame_idx, use_frame_flags=True)

            def _handle_Global(self, node: ast.Global, frame_idx: int):
                """Handle Global nodes."""
                frame = self.frame_stack[frame_idx]
                for name in node.names:
                    frame.variables.global_variables.add(name)

            def _handle_Nonlocal(self, node: ast.Nonlocal, frame_idx: int):
                """Handle Nonlocal nodes."""
                frame = self.frame_stack[frame_idx]
                for name in node.names:
                    frame.variables.nonlocal_variables.add(name)

            def _handle_NamedExpr(self, node, frame_idx: int):
                """Handle NamedExpr (walrus operator :=)."""
                frame = self.frame_stack[frame_idx]
                # The target is an output
                frame.variables.output_variables.add(node.target.id)
                # Visit the value
                self._push_visit(node.value, frame_idx, slot_idx=None, use_frame_flags=True)

            # ==========================================================================================================
            # CONTINUATION HANDLERS - post-processing after child visits
            # ==========================================================================================================

            def _handle_continuation(self, cont_type: str, frame_idx: int, extra_data: Any):
                """Handle continuation work items."""
                if cont_type == CONT_ASSIGN:
                    self._cont_assign(frame_idx, extra_data)
                elif cont_type == CONT_FUNCTION:
                    self._cont_function(frame_idx, extra_data)
                elif cont_type == CONT_COMPREHENSION:
                    self._cont_comprehension(frame_idx, extra_data)
                elif cont_type == CONT_FOR:
                    self._cont_for(frame_idx, extra_data)
                elif cont_type == CONT_WITH:
                    self._cont_with(frame_idx, extra_data)
                elif cont_type == CONT_IF_WHILE:
                    self._cont_if_while(frame_idx, extra_data)
                elif cont_type == CONT_TRY:
                    self._cont_try(frame_idx, extra_data)
                elif cont_type == CONT_MATCH_CASE:
                    self._cont_match_case(frame_idx, extra_data)
                elif cont_type == CONT_CLASS:
                    self._cont_class(frame_idx, extra_data)

            def _cont_assign(self, frame_idx: int, extra_data):
                """Continuation for Assign nodes."""
                slot_start, num_targets = extra_data
                frame = self.frame_stack[frame_idx]
                
                value_vars = frame.result_slots[slot_start] or ExposedVariables()
                target_vars = [frame.result_slots[slot_start + 1 + i] or ExposedVariables() for i in range(num_targets)]
                
                # v_merge(self.variables, *value, h_merge(*targets))
                targets_merged = h_merge(*target_vars)
                frame.variables = v_merge(frame.variables, value_vars, targets_merged, _class=frame._class)

            def _cont_function(self, frame_idx: int, extra_data):
                """Continuation for function nodes."""
                slot_start, func_name, parent_class = extra_data
                frame = self.frame_stack[frame_idx]
                
                decorators_vars = frame.result_slots[slot_start] or ExposedVariables()
                argument_vars = frame.result_slots[slot_start + 1] or ExposedVariables()
                body_vars = frame.result_slots[slot_start + 2] or ExposedVariables()
                
                # Merge decorator/annotation inputs
                frame.variables.input_variables |= decorators_vars.input_variables
                frame.variables.output_variables |= decorators_vars.output_variables
                frame.variables.input_variables |= argument_vars.input_variables
                frame.variables.output_variables |= argument_vars.output_variables
                
                # Function name set
                func_name_set = {func_name} if func_name else set()
                
                # Compute body inputs, removing introduced variables and function name
                input_vars = body_vars.input_variables - argument_vars.introduced_variables - func_name_set
                inputs_in_class = body_vars.inputs_variables_in_function_in_class - argument_vars.introduced_variables - func_name_set
                globals_and_nonlocals = body_vars.global_variables | body_vars.nonlocal_variables
                
                if parent_class:
                    frame.variables.inputs_variables_in_function_in_class |= input_vars | globals_and_nonlocals
                else:
                    frame.variables.input_variables |= (input_vars - frame.variables.output_variables) | globals_and_nonlocals
                    frame.variables.inputs_variables_in_function_in_class |= (inputs_in_class - frame.variables.output_variables) | globals_and_nonlocals
                
                frame.variables.output_variables |= func_name_set
                frame.variables.global_variables |= body_vars.global_variables

            def _cont_comprehension(self, frame_idx: int, extra_data):
                """Continuation for comprehension nodes."""
                slot_start, gen_info, num_targets = extra_data
                frame = self.frame_stack[frame_idx]
                
                # Collect all generator vars
                comprehension_scopes = []
                all_vars_target = []
                for gen_slot_start, num_ifs in gen_info:
                    vars_target = frame.result_slots[slot_start + gen_slot_start] or ExposedVariables()
                    vars_iter = frame.result_slots[slot_start + gen_slot_start + 1] or ExposedVariables()
                    vars_ifs = [frame.result_slots[slot_start + gen_slot_start + 2 + i] or ExposedVariables() 
                               for i in range(num_ifs)]
                    comprehension_scopes.append(v_merge(vars_target, vars_iter, *vars_ifs))
                    all_vars_target.extend(vars_target.output_variables)
                
                # Get target vars
                targets_slot_start = gen_info[-1][0] + 2 + gen_info[-1][1] if gen_info else 0
                targets_scopes = [frame.result_slots[slot_start + targets_slot_start + i] or ExposedVariables()
                                 for i in range(num_targets)]
                
                # Merge
                vars_scope = v_merge(*comprehension_scopes, h_merge(*targets_scopes))
                vars_scope.output_variables -= set(all_vars_target)
                frame.variables = v_merge(frame.variables, vars_scope)

            def _cont_for(self, frame_idx: int, extra_data):
                """Continuation for For nodes."""
                slot_start, _class = extra_data
                frame = self.frame_stack[frame_idx]
                
                vars_iter = frame.result_slots[slot_start] or ExposedVariables()
                vars_target = frame.result_slots[slot_start + 1] or ExposedVariables()
                vars_body = frame.result_slots[slot_start + 2] or ExposedVariables()
                vars_orelse = frame.result_slots[slot_start + 3] or ExposedVariables()
                
                # Join body statements
                vars_body = join_body_stmts_into_vars(vars_body, _class=_class)
                vars_orelse = join_body_stmts_into_vars(vars_orelse, _class=_class)
                
                frame.variables = v_merge(
                    frame.variables,
                    v_merge(vars_iter, vars_target, h_merge(vars_body, vars_orelse), _class=_class),
                    _class=_class
                )

            def _cont_with(self, frame_idx: int, extra_data):
                """Continuation for With nodes."""
                slot_start, num_items, _class = extra_data
                frame = self.frame_stack[frame_idx]
                
                vars_body = frame.result_slots[slot_start] or ExposedVariables()
                vars_body = join_body_stmts_into_vars(vars_body, _class=_class)
                
                items_vars = [frame.result_slots[slot_start + 1 + i] or ExposedVariables() for i in range(num_items)]
                items_merged = v_merge(*[v_merge(v) for v in items_vars], _class=_class)
                
                frame.variables = v_merge(frame.variables, v_merge(items_merged, vars_body, _class=_class), _class=_class)

            def _cont_if_while(self, frame_idx: int, extra_data):
                """Continuation for If/While nodes."""
                slot_start, _class = extra_data
                frame = self.frame_stack[frame_idx]
                
                vars_test = frame.result_slots[slot_start] or ExposedVariables()
                vars_body = frame.result_slots[slot_start + 1] or ExposedVariables()
                vars_orelse = frame.result_slots[slot_start + 2] or ExposedVariables()
                
                vars_body = join_body_stmts_into_vars(vars_body, _class=_class)
                vars_orelse = join_body_stmts_into_vars(vars_orelse, _class=_class)
                
                frame.variables = v_merge(
                    frame.variables,
                    v_merge(vars_test, h_merge(vars_body, vars_orelse), _class=_class),
                    _class=_class
                )

            def _cont_try(self, frame_idx: int, extra_data):
                """Continuation for Try nodes."""
                slot_start, num_handlers, handler_names, _class = extra_data
                frame = self.frame_stack[frame_idx]
                
                vars_body = frame.result_slots[slot_start] or ExposedVariables()
                vars_orelse = frame.result_slots[slot_start + 1] or ExposedVariables()
                vars_finalbody = frame.result_slots[slot_start + 2] or ExposedVariables()
                
                vars_body = join_body_stmts_into_vars(vars_body, _class=_class)
                vars_orelse = join_body_stmts_into_vars(vars_orelse, _class=_class)
                vars_finalbody = join_body_stmts_into_vars(vars_finalbody, _class=_class)
                
                all_vars_handlers = []
                for i in range(num_handlers):
                    vars_handler = frame.result_slots[slot_start + 3 + i] or ExposedVariables()
                    handler_name = handler_names[i]
                    # Remove handler name from input/output
                    if handler_name:
                        vars_handler.input_variables -= {handler_name}
                        vars_handler.output_variables -= {handler_name}
                        vars_handler.inputs_variables_in_function_in_class -= {handler_name}
                    all_vars_handlers.append(vars_handler)
                
                frame.variables = v_merge(
                    frame.variables,
                    v_merge(vars_body, h_merge(*all_vars_handlers, vars_orelse), vars_finalbody, _class=_class),
                    _class=_class
                )

            def _cont_match_case(self, frame_idx: int, extra_data):
                """Continuation for match case bodies."""
                slot_start, _class = extra_data
                frame = self.frame_stack[frame_idx]
                
                vars_body = frame.result_slots[slot_start] or ExposedVariables()
                vars_body = join_body_stmts_into_vars(vars_body, _class=_class)
                frame.variables = v_merge(frame.variables, vars_body, _class=_class)

            def _cont_class(self, frame_idx: int, extra_data):
                """Continuation for ClassDef nodes."""
                slot_start, class_name = extra_data
                frame = self.frame_stack[frame_idx]
                
                decorators_vars = frame.result_slots[slot_start] or ExposedVariables()
                body_vars = frame.result_slots[slot_start + 1] or ExposedVariables()
                body_vars = join_body_stmts_into_vars(body_vars, _class=True)
                
                # Merge decorators/bases
                frame.variables.input_variables |= decorators_vars.input_variables
                frame.variables.output_variables |= decorators_vars.output_variables
                
                # Merge body
                frame.variables.input_variables |= (body_vars.input_variables - frame.variables.output_variables)
                frame.variables.inputs_variables_in_function_in_class |= body_vars.inputs_variables_in_function_in_class
                frame.variables.output_variables |= {class_name}
                frame.variables.nonlocal_variables |= body_vars.nonlocal_variables
                frame.variables.global_variables |= body_vars.global_variables

            def _process_work_stack(self):
                """Main loop - process work items until stack is empty."""
                while self.work_stack:
                    work_item = self.work_stack.pop()
                    work_type = work_item[0]
                    
                    if work_type == WORK_VISIT:
                        _, node, frame_idx, slot_idx, is_lhs_target, is_also_input_of_aug_assign, _class = work_item
                        self._handle_visit(node, frame_idx, slot_idx, is_lhs_target, is_also_input_of_aug_assign, _class)
                    elif work_type == WORK_CONTINUATION:
                        _, cont_type, frame_idx, extra_data = work_item
                        self._handle_continuation(cont_type, frame_idx, extra_data)
                    elif work_type == 'store_result':
                        # Store child frame's variables into parent's slot
                        _, parent_frame_idx, slot_idx, child_frame_idx = work_item
                        child_vars = self.frame_stack[child_frame_idx].variables
                        self.frame_stack[parent_frame_idx].result_slots[slot_idx] = child_vars
                    elif work_type == 'collect_body':
                        # Collect body frame variables (joined)
                        _, parent_frame_idx, slot_idx, body_frame_idx, num_stmts = work_item
                        body_vars = self.frame_stack[body_frame_idx].variables
                        self.frame_stack[parent_frame_idx].result_slots[slot_idx] = body_vars
                    elif work_type == 'collect_try_handler':
                        # Collect handler frame variables
                        _, parent_frame_idx, slot_idx, handler_frame_idx, handler_name, num_stmts = work_item
                        handler_vars = self.frame_stack[handler_frame_idx].variables
                        self.frame_stack[parent_frame_idx].result_slots[slot_idx] = handler_vars


        # ==========================================================================================================
        # The old recursive TempScopeVisitor implementation has been fully replaced by the new IterativeScopeVisitor.
        # All variable extraction is now handled non-recursively for performance and maintainability.

        # ArgumentsVisitor is no longer needed; argument handling is now integrated into the iterative visitor.


        ####################################################################################################
        ######################### UTILITY FUNCTIONS ##########################################
        ####################################################################################################

        def annotate(tree: ast.AST):
            """Annotate an AST tree with input/output variables using the iterative visitor."""
            visitor = IterativeScopeVisitor()
            return visitor.annotate(tree)

        # The old recursive annotate_recursive is removed; only the iterative visitor is supported now.

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
            already_matched_old_nodes: Set[int] = set()  # Track which old nodes have been matched to prevent double-matching

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
                
                # Filter out candidates that have already been matched to other new nodes
                available_candidates = [c for c in all_candidates_in_old if c not in already_matched_old_nodes]
                
                # Find all candidates with matching AST
                matching_candidates = [
                    candidate for candidate in available_candidates 
                    if ast_equality(old_dag.nodes[candidate]['tree_ast'], new_dag.nodes[new_node]['tree_ast'])
                ]
                
                # If there are multiple matches, prefer the one with the closest line number
                if len(matching_candidates) > 1:
                    new_lineno = new_dag.nodes[new_node].get('lineno', 0)
                    matching_candidates.sort(key=lambda c: abs(old_dag.nodes[c].get('lineno', 0) - new_lineno))
                
                if matching_candidates:
                    best_match = matching_candidates[0]
                    new_node_to_old[new_node] = best_match
                    already_matched_old_nodes.add(best_match)
                
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