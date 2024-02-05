// import * as vscode from 'vscode';
// import {
//     languages,
//     Disposable,
//     window,
//     Range,
//     workspace,
//     TextEditor,
//     Position,
//     Selection,
//     CodeLensProvider
// } from 'vscode';

// import type * as nbformat from '@jupyterlab/nbformat';
// import { integer } from 'vscode-languageserver-protocol';
// import { joinPath } from './platform/vscode-path/resources';
// import { IFileSystem } from './platform/common/platform/types';
// import { IExtensionContext } from './platform/common/types';

// import { INotebookWatcher } from './webviews/extension-side/variablesView/types';
// import { IServiceContainer, IServiceManager } from './platform/ioc/types';
// import { IKernel } from './kernels/types';
// import { executeSilently, SilentExecutionErrorOptions } from './kernels/helpers';
// import { SessionDisposedError } from './platform/errors/sessionDisposedError';

// // This from SqlTools:
// // Actually a Good and important idea!
// // const keyName = 'attachedFilesMap';
// // export const getAttachedConnection = (file: Uri | string) => {
// //   return Context.workspaceState.get(keyName, {})[file.toString()];   // I had used globalState, but probably workspaceState is better
// // }  // I want to assign a >>Notebook<< to the TextEditor, not a Kernel, and then get the Kernel from the Notebook, so that i can ALSO use the Notebook for adding Cells etc

// // import { IExtensionContext } from './platform/common/types';

// ///////////////////////////////////////////////////////////////////////////////////////
// //// THIS: IS WHAT WE ARE DOING NOW: VERY GOOD:
// ///////////////////////////////////////////////////////////////////////////////////////

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // PYTHON COMMANDS AND SNIPPETS
// ////////////////////////////////////////////////////////////////////////////////////////////////////

// type AnnotatedRange = {
//     range: Range;
//     state: string; // Remember this exists too: 'synced' | 'outdated';
//     current: boolean;
//     text?: string;
//     hash?: string; // Hash is used so that when you send a node Back to Python, you can check if it actually him or not
// };

// let __reactivePythonEngineScript: string | undefined = undefined;

// /* Read script content from file: */
// async function generateCodeToGetVariableTypes(
//     context: IExtensionContext,
//     serviceContainer: IServiceContainer
// ): Promise<string> {
//     let fs = serviceContainer.get<IFileSystem>(IFileSystem);
//     if (__reactivePythonEngineScript) {
//         return __reactivePythonEngineScript;
//     }
//     const scriptPath = joinPath(context.extensionUri, 'src', 'reactive_python_engine.py');
//     const scriptCode = await fs.readFile(scriptPath);
//     __reactivePythonEngineScript = scriptCode;

//     // const initializeCode = ``;
//     // const cleanupWhoLsCode = dedent``;
//     // let code =  `${VariableFunc}("types", ${isDebugging}, _VSCODE_rwho_ls)`;
//     // code = `${initializeCode}${code}\n\n${cleanupCode}\n${cleanupWhoLsCode}`
//     return scriptCode;
// }

// export const getCommandToGetRangeToSelectFromPython = (currentQuery: string): string => {
//     /*
//     FAKE
//     */
//     // Pass currentQuery as a SIMPLE STRING, ie all the newlines should be passed in as explicit \n and so on:
//     // Turn currentQuery in a string that can be passed to Python, by sanitizing all the Newlines, Quotes and Indentations
//     // (ie, keeping them but in a way that Python can receive as a string):
//     return 'get_commands_to_execute("""' + currentQuery + '""")';
// };

// export const getCommandToGetAllRanges = (
//     text: string | null,
//     current_line: integer | null,
//     upstream: boolean,
//     downstream: boolean,
//     stale_only: boolean,
//     to_launch_compute: boolean = false
// ): string => {
//     let text_ = text || 'None';
//     let current_line_str: string = current_line ? current_line.toString() : 'None';
//     let upstream_param: string = upstream ? 'True' : 'False';
//     let downstream_param: string = downstream ? 'True' : 'False';
//     let stale_only_param: string = stale_only ? 'True' : 'False';
//     // return `if "reactive_python_dag_builder_utils__" in globals():\n\treactive_python_dag_builder_utils__.update_dag_and_get_ranges(code= ${text_}, current_line=${current_line_str}, get_upstream=${upstream_param}, get_downstream=${downstream_param}, stale_only=${stale_only_param})\nelse:\n\t[]`;
//     if (to_launch_compute) {
//         return `reactive_python_dag_builder_utils__.ask_for_ranges_to_compute(code= ${text_}, current_line=${current_line_str}, get_upstream=${upstream_param}, get_downstream=${downstream_param}, stale_only=${stale_only_param})`;
//     } else {
//         return `reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code= ${text_}, current_line=${current_line_str}, get_upstream=${upstream_param}, get_downstream=${downstream_param}, stale_only=${stale_only_param})`;
//     }
// };

// export const getSyncRangeCommand = (range: AnnotatedRange): string => {
//     return `reactive_python_dag_builder_utils__.set_locked_range_as_synced(${range.hash})`;
// };

// export const getUnlockCommand = (): string => {
//     return `reactive_python_dag_builder_utils__.unlock()`;
// };

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // UTILS
// ////////////////////////////////////////////////////////////////////////////////////////////////////

// import { ExtensionContext } from 'vscode';
// import { IDataScienceCodeLensProvider } from './interactive-window/editor-integration/types';

// let currentContext: ExtensionContext & { set: typeof setCurrentContext; onRegister: typeof onRegister } = {} as any;

// const queue: any[] = [];

// const onRegister = (cb: () => void) => queue.push(cb);

// export const setCurrentContext = (ctx: ExtensionContext) => {
//     currentContext = ctx as typeof currentContext;
//     queue.forEach((cb) => cb());
// };

// const handler = {
//     get(_: never, prop: string) {
//         if (prop === 'set') return setCurrentContext;
//         if (prop === 'onRegister') return onRegister;
//         return currentContext[prop];
//     },
//     set() {
//         throw new Error('Cannot set values to extension context directly!');
//     }
// };

// const Context = new Proxy<typeof currentContext>(currentContext, handler);

// export default Context;

// async function safeExecuteSilently(
//     kernel: IKernel,
//     { code, initializeCode, cleanupCode }: { code: string; initializeCode?: string; cleanupCode?: string },
//     errorOptions?: SilentExecutionErrorOptions
// ): Promise<nbformat.IOutput[]> {
//     if (kernel.disposed || kernel.disposing || !kernel.session || !kernel.session.kernel || kernel.session.disposed) {
//         return [];
//     }
//     try {
//         if (initializeCode) {
//             await executeSilently(kernel.session, initializeCode, errorOptions);
//         }
//         return await executeSilently(kernel.session, code, errorOptions);
//     } catch (ex) {
//         if (ex instanceof SessionDisposedError) {
//             return [];
//         }
//         throw ex;
//     } finally {
//         if (cleanupCode) {
//             await executeSilently(kernel.session, cleanupCode, errorOptions);
//         }
//     }
// }

// async function safeExecuteSilentlyInKernelAndReturnOutput(code: string, kernel: IKernel): Promise<string | null> {
//     return safeExecuteSilently(kernel, { code: code, initializeCode: '', cleanupCode: '' }).then((result) => {
//         // Understand if the result is an Error:
//         if (result && result[0] && result[0].output_type && result[0].output_type === 'error') {
//             vscode.window.showInformationMessage(
//                 'Reactive Python: Failed to execute code in Python Kernel: ' +
//                     result[0].evalue +
//                     ' - Traceback: ' +
//                     // Concat all the lines of the traceback:
//                     result[0].traceback.reduce((acc, curr) => acc + curr, '')
//             );
//             return null;
//         } else if (result && result[0] && result[0].data && result[0].data['text/plain'] !== undefined) {
//             console.log('>>>>>>>>Success! ');
//             let result_str: string = result[0].data['text/plain'];
//             console.log('Result: ' + result[0].data['text/plain']);
//             return result_str;
//         } else {
//             console.log('>>>>>>>>Failure! While executing code ' + code + 'Only got this result: ' + result);
//             vscode.window.showInformationMessage(
//                 'Reactive Python: Failed to execute code in Python Kernel while executing code ' +
//                     code +
//                     '. Retrieved: ' +
//                     result
//             );
//             return null;
//         }
//     });
// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // MY UTILS
// ////////////////////////////////////////////////////////////////////////////////////////////////////

// function getKernel(serviceManager: IServiceManager): IKernel | undefined {
//     // let getter = serviceManager.get<IVariableViewProvider>(IVariableViewProvider);
//     // let notebookWatcher = getter.notebookWatcher; // Idea: VariableViewProvider has it tho
//     let notebookWatcher = serviceManager.get<INotebookWatcher>(INotebookWatcher); // Would this work ???
//     let kernel = notebookWatcher.activeKernel;

//     // OR do it like this (at which point YOU DON'T EVEN NEED the ServiceManager):
//     // import { extensions, workspace } from 'vscode';
//     // const jupyterExt = extensions.getExtension<JupyterAPI>('ms-toolsai.jupyter');
//     // await jupyterExt.activate();
//     // const kernelService = await jupyterExt.exports.getKernelService();
//     // const activeKernels = kernelService.getActiveKernels();
//     // ? // Profit

//     return kernel;
// }

// async function safeGetKernel(serviceManager: IServiceManager): Promise<IKernel> {
//     /*
//     If Kernel is undefined, starts a new one by calling command 'jupyter.createnewinteractive'
//     */
//     let kernel = getKernel(serviceManager);
//     // Try at most 4 times:
//     for (let i = 0; i < 4; i++) {
//         if (kernel === undefined) {
//             vscode.window.showInformationMessage('Reactive Python: Starting Python Kernel: attempt ' + (i + 1));
//             await vscode.commands.executeCommand('jupyter.createnewinteractive');
//             // Repeat 3 times:
//             for (let j = 0; j < 3; j++) {
//                 // Sleep for 3 seconds:
//                 await new Promise((resolve) => setTimeout(resolve, 3000));
//                 kernel = getKernel(serviceManager);
//                 if (kernel !== undefined) {
//                     vscode.window.showInformationMessage('Reactive Python: Successfully started Python Kernel!');
//                     break;
//                 }
//             }
//         } else {
//             return kernel;
//         }
//     }
//     vscode.window.showInformationMessage('Reactive Python: Failed to start Python Kernel with the Jupiter extension.');
//     throw new Error('Could not get kernel');
// }

// async function getIWAndRunText(serviceManager: IServiceManager, editor: vscode.TextEditor, text: string) {
//     let getter = serviceManager.get<IDataScienceCodeLensProvider>(IDataScienceCodeLensProvider);
//     let codeWatcher = getter.getCodeWatcher(editor.document);
//     let result = await codeWatcher?.runSelectionOrLine(editor, text);
//     return result;

//     // OR is there a better way to do this? ???????
// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // HIGHLIGHTING UTILS
// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // import { TextEditor, Range } from 'vscode';

// export const getEditorAllText = (editor: TextEditor): { text: string | null } => {
//     if (!editor || !editor.document || editor.document.uri.scheme === 'output') {
//         return {
//             text: null
//         };
//     }
//     const text = editor.document.getText();
//     return { text };
// };

// function formatTextAsPythonString(text: string) {
//     text = text.replace(/\\/g, '\\\\');
//     // >> You MIGHT be interested in
//     // /Users/michele.tasca/Documents/vscode-extensions/vscode-reactive-jupyter/src/platform/terminals/codeExecution/codeExecutionHelper.node.ts
//     //  >> CodeExecutionHelper >> normalizeLines  ...
//     text = text.replace(/'/g, "\\'");
//     text = text.replace(/"/g, '\\"');
//     text = text.replace(/\n/g, '\\n');
//     text = text.replace(/\r/g, '\\r');
//     text = text.replace(/\t/g, '\\t');
//     text = '"""' + text + '"""';
//     return text;
// }

// export const getEditorCurrentLineNum = (editor: TextEditor): integer | null => {
//     if (!editor || !editor.document || editor.document.uri.scheme === 'output') {
//         return null;
//     }
//     const currentLineNum = editor.selection.active.line;
//     return currentLineNum;
// };

// export const getEditorCurrentText = (editor: TextEditor): { currentQuery: string; currentRange: Range | null } => {
//     if (!editor || !editor.document || editor.document.uri.scheme === 'output') {
//         return {
//             currentQuery: '',
//             currentRange: null
//         };
//     }
//     if (!editor.selection.isEmpty) {
//         return {
//             currentQuery: editor.document.getText(editor.selection),
//             currentRange: editor.selection
//         };
//     }
//     const currentLine = editor.document
//         .getText(new Range(Math.max(0, editor.selection.active.line - 4), 0, editor.selection.active.line + 1, 0))
//         .replace(/[\n\r\s]/g, '');
//     if (currentLine.length === 0)
//         return {
//             currentQuery: '',
//             currentRange: editor.selection
//         };
//     const text = editor.document.getText();
//     const currentOffset = editor.document.offsetAt(editor.selection.active);
//     const prefix = text.slice(0, currentOffset + 1);
//     const allQueries = text; // parse(text);
//     const prefixQueries = prefix; // parse(prefix);
//     const currentQuery = allQueries[prefixQueries.length - 1];
//     const startIndex = prefix.lastIndexOf(prefixQueries[prefixQueries.length - 1]);
//     const startPos = editor.document.positionAt(startIndex);
//     const endPos = editor.document.positionAt(startIndex + currentQuery.length);
//     return {
//         currentQuery,
//         currentRange: new Range(startPos, endPos)
//     };
// };

// export const parseResultFromPythonAndGetRange = (resultFromPython: string): AnnotatedRange[] | null => {
//     // ResultFromPython is a string of the form: "[[startLine, endline, state], [startLine, endline, state], ...]" (the length is indefinite)
//     // Parse it and return the list of ranges to select:

//     // Result is returned as String, so remove the first and last character to get the Json-parsable string:
//     resultFromPython = resultFromPython.substring(1, resultFromPython.length - 1);
//     // Sanitize: To be clear, I'm ALMOST SURE this is a terrible idea...
//     resultFromPython = resultFromPython.replace(/\\\\/g, '\\');
//     resultFromPython = resultFromPython.replace(/\\"/g, '\\"');
//     resultFromPython = resultFromPython.replace(/\\'/g, "'");
//     resultFromPython = resultFromPython.replace(/\\n/g, '\\n');
//     resultFromPython = resultFromPython.replace(/\\r/g, '\\r');
//     resultFromPython = resultFromPython.replace(/\\t/g, '\\t');

//     // Json parse:
//     let resultFromPythonParsed;
//     try {
//         resultFromPythonParsed = JSON.parse(resultFromPython);
//     } catch (e) {
//         console.log('Failed to parse result from Python: ' + resultFromPython);
//         vscode.window.showErrorMessage('Reactive Python: Failed to parse JSON result from Python: ' + resultFromPython);
//         return null;
//     }
//     // Assert that it worked:
//     if (resultFromPythonParsed === undefined) {
//         console.log('Failed to parse result from Python: ' + resultFromPython);
//         return null;
//     }
//     // Convert to Range[]:
//     let ranges: AnnotatedRange[] = [];
//     for (let i = 0; i < resultFromPythonParsed.length; i++) {
//         let startLine = resultFromPythonParsed[i][0];
//         let endLine = resultFromPythonParsed[i][1];
//         let state = resultFromPythonParsed[i][2];
//         let current = resultFromPythonParsed[i][3] == 'current';
//         let text_ = resultFromPythonParsed[i].length > 4 ? resultFromPythonParsed[i][4] : undefined;
//         let hash = resultFromPythonParsed[i].length > 5 ? resultFromPythonParsed[i][5] : undefined;
//         if (startLine >= 0 && endLine >= 0 && startLine <= endLine && (state === 'synced' || state === 'outdated')) {
//             // Parse as int:
//             startLine = parseInt(startLine);
//             endLine = parseInt(endLine);
//             ranges.push({
//                 range: new Range(new Position(startLine, 0), new Position(endLine + 1, 0)),
//                 state: state,
//                 current: current,
//                 text: text_,
//                 hash: hash
//             });
//         }
//     }
//     return ranges;
// };

// const getCurrentRangesFromPython = async (
//     editor: TextEditor,
//     serviceManager: IServiceManager,
//     {
//         rebuild,
//         current_line = undefined,
//         upstream = true,
//         downstream = true,
//         stale_only = false,
//         to_launch_compute = false
//     }: {
//         rebuild: boolean;
//         current_line?: integer | undefined | null;
//         upstream?: boolean;
//         downstream?: boolean;
//         stale_only?: boolean;
//         to_launch_compute?: boolean;
//     },
//     kernel: IKernel | undefined = undefined
// ): Promise<AnnotatedRange[] | undefined> => {
//     if (current_line === null) {
//         console.log('getCurrentRangesFromPython: current_line is none');
//     }
//     let linen = current_line === undefined ? getEditorCurrentLineNum(editor) : current_line;
//     let text: string | null = rebuild ? getEditorAllText(editor).text : null;
//     if (!text && !linen) return;
//     text = text ? formatTextAsPythonString(text) : null;
//     let command = getCommandToGetAllRanges(text, linen, upstream, downstream, stale_only, to_launch_compute);
//     if (!kernel) {
//         kernel = await getKernel(serviceManager);
//     }
//     if (!kernel) return;
//     const result = await safeExecuteSilentlyInKernelAndReturnOutput(command, kernel);
//     if (!result || result == '[]') return;
//     if (to_launch_compute) {
//         console.log('Result from Python: ' + result);
//     }
//     const ranges_out = await parseResultFromPythonAndGetRange(result);
//     if (!ranges_out) return;
//     return ranges_out;
// };

// export const getTextInRanges = (ranges: AnnotatedRange[]): string[] => {
//     let text: string[] = [];
//     let editor = window.activeTextEditor;
//     if (!editor) return text;
//     for (let i = 0; i < ranges.length; i++) {
//         let range = ranges[i].range;
//         let textInRange = editor.document.getText(range);
//         text.push(textInRange);
//     }
//     return text;
// };

// const HighlightSynced = window.createTextEditorDecorationType({
//     backgroundColor: { id: `jupyter.syncedCell` },
//     borderColor: { id: `jupyter.syncedCell` },
//     borderWidth: '0px',
//     borderStyle: 'solid'
// });
// const HighlightSyncedCurrent = window.createTextEditorDecorationType({
//     backgroundColor: { id: `jupyter.syncedCurrentCell` },
//     borderColor: { id: `jupyter.syncedCurrentCell` },
//     borderWidth: '0px',
//     borderStyle: 'solid'
// });
// const HighlightOutdated = window.createTextEditorDecorationType({
//     backgroundColor: { id: `jupyter.outdatedCell` },
//     borderColor: { id: `jupyter.outdatedCell` },
//     borderWidth: '0px',
//     borderStyle: 'solid'
// });
// const HighlightOutdatedCurrent = window.createTextEditorDecorationType({
//     backgroundColor: { id: `jupyter.outdatedCurrentCell` },
//     borderColor: { id: `jupyter.outdatedCurrentCell` },
//     borderWidth: '0px',
//     borderStyle: 'solid'
// });

// let updateDecorations = async (editor: TextEditor, ranges_out: AnnotatedRange[]) => {
//     // if (!Config.highlightQuery) return;
//     if (
//         !editor ||
//         !editor.document ||
//         editor.document.uri.scheme === 'output'
//         //  || !this.registeredLanguages.includes(editor.document.languageId) // <--- COMES FROM this.registeredLanguages = Config.codelensLanguages;  // See Config below
//     ) {
//         return;
//     }
//     try {
//         // const { currentRange, currentQuery } = getEditorCurrentText(editor);
//         // if (!currentRange || !currentQuery) return;
//         // let command = getCommandToGetRangeToSelectFromPython(currentRange.start.line.toString());

//         console.log('>>>>>>>>Did it! ');
//         // Save the range associated with this editor:
//         // 1. Get the Kernel uuid:
//         // let kernel_uuid = editor.document.uri;
//         // // 2. Set current editor ranges in global state:
//         // await globalState.update(kernel_uuid.toString() + '_ranges', ranges_out);
//         // console.log('>>>>>>>>Global State updated');

//         // Set highlight on all the ranges in ranges_out with state == 'synced'
//         let sync_ranges = ranges_out.filter((r) => r.state == 'synced' && !r.current).map((r) => r.range);
//         editor.setDecorations(HighlightSynced, sync_ranges);
//         let sync_curr_ranges = ranges_out.filter((r) => r.state == 'synced' && r.current).map((r) => r.range);
//         editor.setDecorations(HighlightSyncedCurrent, sync_curr_ranges);
//         let out_ranges = ranges_out.filter((r) => r.state == 'outdated' && !r.current).map((r) => r.range);
//         editor.setDecorations(HighlightOutdated, out_ranges);
//         let out_curr_ranges = ranges_out
//             .filter((r) => (r.state == 'outdated' || r.state == 'syntaxerror') && r.current)
//             .map((r) => r.range);
//         editor.setDecorations(HighlightOutdatedCurrent, out_curr_ranges);
//     } catch (error) {
//         console.log('update decorations failed: %O', error);
//     }
// };

// // let updateDecorations = async (editor: TextEditor, globalState: vscode.Memento) => {
// //     try {
// //         // Get the Kernel uuid:
// //         let kernel_uuid = editor.document.uri;
// //         // Get the ranges from global state:
// //         let ranges_out: { range: Range; state: string }[] | undefined = await globalState.get(
// //             kernel_uuid.toString() + '_ranges'
// //         );
// //         console.log('>>>>>>>>REFRESHING the ranges: ');
// //         console.log(ranges_out);

// //         if (ranges_out) {
// //             console.log('>>>>>>>>YEE its NOT undefined! !!');

// //         }
// //         console.log('\n\n');
// //     } catch (error) {
// //         console.log('update decorations failed: %O', error);
// //     }
// // };

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // COMMANDS
// ////////////////////////////////////////////////////////////////////////////////////////////////////

// async function queueComputation(
//     current_ranges: AnnotatedRange[] | undefined,
//     serviceManager: IServiceManager,
//     kernel: IKernel,
//     activeTextEditor: TextEditor
// ) {
//     console.log('>> CURRENT RANGES::::: ' + current_ranges);
//     console.log('>> \n\n\n');
//     if (current_ranges) {
//         for (let range of current_ranges) {
//             if (!range.text) break;
//             console.log('>> TEXT: ', range.text);
//             console.log('>> \n\n\n');
//             let res = await getIWAndRunText(serviceManager, activeTextEditor, range.text);
//             console.log('>> RESULT: ', res);
//             console.log('>> \n\n\n');
//             // Check if res is an Error:
//             if (!res) {
//                 break;
//             }
//             let updateRange_command = getSyncRangeCommand(range);
//             const update_result = await safeExecuteSilentlyInKernelAndReturnOutput(updateRange_command, kernel);
//             if (!update_result) {
//                 vscode.window.showErrorMessage("Failed to update the range's state in Python");
//                 break;
//             }
//         }
//     }
//     let unlockCommand = getUnlockCommand();
//     const update_result = await safeExecuteSilentlyInKernelAndReturnOutput(unlockCommand, kernel);
//     if (!update_result) {
//         vscode.window.showErrorMessage('Failed to unlock the Python kernel');
//     }
//     // Trigger a onDidChangeTextEditorSelection event:
//     const refreshed_ranges = await getCurrentRangesFromPython(activeTextEditor, serviceManager, {
//         rebuild: false
//     });
//     if (refreshed_ranges) {
//         updateDecorations(activeTextEditor, refreshed_ranges);
//     }
// }

// export function createPreparePythonEnvForReactivePython(serviceManager: IServiceManager) {
//     async function preparePythonEnvForReactivePython() {
//         /* IDEA:
//         we DON'T want to run this on Activation, but WE ARE DOING IT FOR NOW. for easier testing.
//         */
//         let command = __reactivePythonEngineScript + '\n\n\n"Reactive Python Activated"\n';
//         let kernel = await safeGetKernel(serviceManager);
//         const result = await safeExecuteSilentlyInKernelAndReturnOutput(command, kernel).then((result) => {
//             return result;
//         });
//         if (result) {
//             vscode.window.showInformationMessage('Result: ' + result);
//             if (window.activeTextEditor) {
//                 let refreshed_ranges = await getCurrentRangesFromPython(window.activeTextEditor, serviceManager, {
//                     rebuild: true,
//                     current_line: null
//                 }); // Do this in order to immediatly recompute the dag in the python kernel
//                 if (refreshed_ranges) {
//                     updateDecorations(window.activeTextEditor, refreshed_ranges);
//                 }
//             }
//         }
//         // if (vscode.window.activeTextEditor) {
//         //     await updateDecorations(vscode.window.activeTextEditor, serviceManager);
//         //     updateDecorations(vscode.window.activeTextEditor);
//         // }
//     }
//     return preparePythonEnvForReactivePython;
// }

// export function createComputeAllDownstreamAction(serviceManager: IServiceManager) {
//     async function computeAllDownstreamAction() {
//         if (!window.activeTextEditor) return;
//         let kernel = await safeGetKernel(serviceManager);
//         if (!kernel) return;
//         const current_ranges = await getCurrentRangesFromPython(
//             window.activeTextEditor,
//             serviceManager,
//             {
//                 rebuild: true,
//                 upstream: false,
//                 downstream: true,
//                 stale_only: true,
//                 to_launch_compute: true
//             },
//             kernel
//         );
//         await queueComputation(current_ranges, serviceManager, kernel, window.activeTextEditor);
//     }
//     return computeAllDownstreamAction;
// }
// export function createComputeAllUpstreamAction(serviceManager: IServiceManager) {
//     async function computeAllUpstreamAction() {
//         if (!window.activeTextEditor) return;
//         let kernel = await safeGetKernel(serviceManager);
//         if (!kernel) return;
//         const current_ranges = await getCurrentRangesFromPython(
//             window.activeTextEditor,
//             serviceManager,
//             {
//                 rebuild: true,
//                 upstream: true,
//                 downstream: false,
//                 stale_only: true,
//                 to_launch_compute: true
//             },
//             kernel
//         );
//         await queueComputation(current_ranges, serviceManager, kernel, window.activeTextEditor);
//     }
//     return computeAllUpstreamAction;
// }
// export function createComputeAllDownstreamAndUpstreamAction(serviceManager: IServiceManager) {
//     async function computeAllAction() {
//         if (!window.activeTextEditor) return;
//         let kernel = await safeGetKernel(serviceManager);
//         if (!kernel) return;
//         const current_ranges = await getCurrentRangesFromPython(
//             window.activeTextEditor,
//             serviceManager,
//             {
//                 rebuild: true,
//                 upstream: true,
//                 downstream: true,
//                 stale_only: true,
//                 to_launch_compute: true
//             },
//             kernel
//         );
//         await queueComputation(current_ranges, serviceManager, kernel, window.activeTextEditor);
//     }
//     return computeAllAction;
// }

// export function createComputeAllAction(serviceManager: IServiceManager) {
//     async function computeAllAction() {
//         if (!window.activeTextEditor) return;
//         let kernel = await safeGetKernel(serviceManager);
//         if (!kernel) return;
//         const current_ranges = await getCurrentRangesFromPython(
//             window.activeTextEditor,
//             serviceManager,
//             {
//                 rebuild: true,
//                 current_line: null,
//                 upstream: true,
//                 downstream: true,
//                 stale_only: true,
//                 to_launch_compute: true
//             },
//             kernel
//         );
//         await queueComputation(current_ranges, serviceManager, kernel, window.activeTextEditor);
//     }
//     return computeAllAction;
// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // CODELENS
// ////////////////////////////////////////////////////////////////////////////////////////////////////

// export class CellCodelensProvider implements vscode.CodeLensProvider {
//     private codeLenses: vscode.CodeLens[] = [];
//     private range: Range | undefined;
//     private _onDidChangeCodeLenses: vscode.EventEmitter<void> = new vscode.EventEmitter<void>();
//     public readonly onDidChangeCodeLenses: vscode.Event<void> = this._onDidChangeCodeLenses.event;

//     change_range(new_range: Range | undefined) {
//         // console.log('>>>>>>>>>>>>>>>>>>NICE, FIRED RESET!. ');
//         if (new_range && new_range != this.range) {
//             this.range = new_range;
//             this._onDidChangeCodeLenses.fire();
//         } else if (!new_range && this.range) {
//             this.range = undefined;
//             this._onDidChangeCodeLenses.fire();
//         }
//     }
//     constructor() {
//         vscode.workspace.onDidChangeConfiguration((_) => {
//             this._onDidChangeCodeLenses.fire();
//         });
//         // Context.subscriptions.push(this._onDidChangeCodeLenses);
//     }

//     public provideCodeLenses(
//         document: vscode.TextDocument,
//         token: vscode.CancellationToken
//     ): vscode.CodeLens[] | Thenable<vscode.CodeLens[]> {
//         // console.log('>>>>>>>>>>>>>>>>>>NICE, INVOKED ONCE. ');
//         let editor = vscode.window.activeTextEditor;
//         if (editor && this.range && editor.document.uri == document.uri) {
//             // console.log('>>>>>>>>>>>>>>>>>>THESE URIS ARE THE SAME! ');
//             // Current line:
//             this.codeLenses = [
//                 new vscode.CodeLens(new vscode.Range(this.range.start.line, 0, this.range.end.line, 0), {
//                     title: 'sync upstream',
//                     tooltip: 'Run all outdated code upstream, including this cell',
//                     command: 'jupyter.sync-upstream',
//                     arguments: [this.range]
//                 }),
//                 new vscode.CodeLens(new vscode.Range(this.range.start.line, 0, this.range.end.line, 0), {
//                     title: 'sync downstream',
//                     tooltip: 'Run all outdated code downstream, including this cell',
//                     command: 'jupyter.sync-downstream',
//                     arguments: [this.range]
//                 }),
//                 new vscode.CodeLens(new vscode.Range(this.range.start.line, 0, this.range.end.line, 0), {
//                     title: 'sync all',
//                     tooltip: 'Run all outdated code upstream and downstream, including this cell',
//                     command: 'jupyter.sync-all',
//                     arguments: [this.range]
//                 })
//             ];
//             return this.codeLenses;
//         }
//         return [];
//     }
// }

// export class InitialCodelensProvider implements vscode.CodeLensProvider {
//     private started_at_least_once: vscode.CodeLens[] = [];
//     public provideCodeLenses(
//         document: vscode.TextDocument,
//         token: vscode.CancellationToken
//     ): vscode.CodeLens[] | Thenable<vscode.CodeLens[]> {
//         // console.log('>>>>>>>>>>>>>>>>>>NICE, INVOKED ONCE. ');
//         let editor = vscode.window.activeTextEditor;
//         if (editor && editor.document.uri == document.uri) {
//             let codeLenses = [
//                 new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), {
//                     title: '$(debug-start) Initialize Reactive Python',
//                     tooltip: 'Initialize Reactive Python on the current file',
//                     command: 'jupyter.initialize-reactive-python-extension'
//                     // arguments: [this.range] // Wanna pass the editor uri?
//                 }),
//                 new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), {
//                     title: 'Sync all Stale code',
//                     tooltip: 'Sync all Stale code in current file',
//                     command: 'jupyter.sync-all'
//                     // arguments: [this.range] // Wanna pass the editor uri?
//                 })
//             ];
//             return codeLenses;
//         }
//         return [];
//     }
// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////
// // ACTIVATION
// ////////////////////////////////////////////////////////////////////////////////////////////////////   class_weight={0: 0.9}

// export async function defineAllCommands(
//     context: IExtensionContext,
//     serviceManager: IServiceManager,
//     serviceContainer: IServiceContainer
// ) {
//     // Get context global state:
//     let globalState = context.globalState;

//     // Read Python initialization script:
//     let code = await generateCodeToGetVariableTypes(context, serviceContainer);
//     // if (code) {
//     //     console.log('code: ', code);
//     // } else {
//     //     console.log('<<<<<<<<<<<<<<<<<< code is null');
//     // }

//     // The command has been defined in the package.json file // Now provide the implementation of the command with registerCommand // The commandId parameter must match the command field in package.json
//     context.subscriptions.push(
//         vscode.commands.registerCommand(
//             'jupyter.initialize-reactive-python-extension',
//             createPreparePythonEnvForReactivePython(serviceManager)
//         )
//     );
//     context.subscriptions.push(
//         vscode.commands.registerCommand('jupyter.sync-upstream', createComputeAllUpstreamAction(serviceManager))
//     );
//     context.subscriptions.push(
//         vscode.commands.registerCommand('jupyter.sync-downstream', createComputeAllDownstreamAction(serviceManager))
//     );
//     context.subscriptions.push(
//         vscode.commands.registerCommand(
//             'jupyter.sync-upstream-and-downstream',
//             createComputeAllDownstreamAndUpstreamAction(serviceManager)
//         )
//     );

//     context.subscriptions.push(
//         vscode.commands.registerCommand('jupyter.sync-all', createComputeAllAction(serviceManager))
//     );

//     // await preparePythonEnvForReactivePython(serviceManager);

//     ///////// Codelens: ///////////////////////

//     const codelensProvider = new CellCodelensProvider();
//     languages.registerCodeLensProvider('python', codelensProvider);
//     languages.registerCodeLensProvider('python', new InitialCodelensProvider());

//     ///////// Document Highlights: ///////////////////////

//     window.onDidChangeActiveTextEditor(
//         async (editor) => {
//             if (editor) {
//                 const current_ranges = await getCurrentRangesFromPython(editor, serviceManager, { rebuild: true });
//                 if (!current_ranges) return;
//                 await updateDecorations(editor, current_ranges);
//             }
//         },
//         null,
//         Context.subscriptions
//     );
//     workspace.onDidChangeTextDocument(
//         // This: Exists too! >> onDidSaveTextDocument  >> (even if should be included in the onDidChangeTextDocument one ? )
//         // editors, undo/ReactDOM, save, etc
//         async (event) => {
//             if (window.activeTextEditor && event.document === window.activeTextEditor.document) {
//                 const current_ranges = await getCurrentRangesFromPython(window.activeTextEditor, serviceManager, {
//                     rebuild: true
//                 });
//                 if (!current_ranges) return;
//                 await updateDecorations(window.activeTextEditor, current_ranges);
//             }
//         },
//         null,
//         Context.subscriptions
//     );
//     window.onDidChangeTextEditorSelection(
//         async (event) => {
//             if (
//                 event.textEditor &&
//                 window.activeTextEditor &&
//                 event.textEditor.document === window.activeTextEditor.document &&
//                 window.activeTextEditor.selection.isEmpty
//             ) {
//                 const current_ranges = await getCurrentRangesFromPython(window.activeTextEditor, serviceManager, {
//                     rebuild: false
//                 });
//                 if (current_ranges) {
//                     updateDecorations(window.activeTextEditor, current_ranges);
//                 }
//                 let codelense_range = current_ranges ? current_ranges.filter((r) => r.current).map((r) => r.range) : [];
//                 codelensProvider.change_range(codelense_range.length > 0 ? codelense_range[0] : undefined);
//             } else {
//                 updateDecorations(event.textEditor, []);
//                 codelensProvider.change_range(undefined);
//             }
//         },
//         null,
//         Context.subscriptions
//     );
//     ///////// Do highlighting for the first time right now: ///////////////////////
//     // if (window.activeTextEditor) {
//     //     updateDecorations(window.activeTextEditor, serviceManager);
//     // } else {
//     //     vscode.window.showInformationMessage(
//     //         'No active text editor on Activation Time. Try to retrigger the highlighter somehow.'
//     //     );
//     // }

//     ///////// CodeLenses: ///////////////////////
//     // Add a codelens above the line where the cursor is, that launches the "jupyter.test-command" command:
//     // const codelensProvider = new MyCodeLensProvider();
//     // const disposable = languages.registerCodeLensProvider({ language: 'python' }, codelensProvider);
//     // context.subscriptions.push(disposable);
//     // HINT: ONE version of this is /Users/michele.tasca/Documents/vscode-extensions/vscode-reactive-jupyter/src/interactive-window/editor-integration/codelensprovider.ts !!
// }

// // Define the MyCodeLensProvider class:
// // class MyCodeLensProvider implements CodeLensProvider {
// //     provideCodeLenses(document: TextDocument, token: CancellationToken): CodeLens[] {
// //         const codelens = new CodeLens(new Range(0, 0, 0, 0));
// //         return [codelens];
// //     }

// //     resolveCodeLens(codeLens: CodeLens, token: CancellationToken): CodeLens {
// //         codeLens.command = {
// //             title: 'Compute Downstream',
// //             command: 'jupyter.test-command',
// //             tooltip: 'Compute Downstream'
// //         };
// //         return codeLens;
// //     }
// // }

// // Some idea of the way sqltools did Highlighting:
// //   function ext_setSelection = (arg: Selection | Selection[]) => {  // This is just called via a COMMAND ...
// //     if (!window.activeTextEditor || !arg) {return;}
// //     window.activeTextEditor.selections = (Array.isArray(arg) ? arg : [arg]).filter(Boolean);
// //   }
// //   function register(extension: IExtension) {
// //     Context.subscriptions.push(this);
// //     extension
// //       .registerCommand('setSelection', this.ext_setSelection)
// //     this.createCodelens();
// //     this.createDecorations();
// //     Config.addOnUpdateHook(({ event }) => {
// //       if (event.affectsConfig('codelensLanguages')) {
// //         this.createCodelens();
// //       }
// //       if (event.affectsConfig('highlightQuery')) {
// //         this.updateDecorations(window.activeTextEditor);
// //       }
// //     });
// //   }
// // AND: remember Config is in Documents/vscode-extensions/vscode-sqltools/packages/util/config-manager/vscode.ts

// ///////////////////////////////////////////////////////////////////////////////////////////////////
// ///////////// COOL EXAMPLE of adding decorations, if you want to!!!
// ///////////////////////////////////////////////////////////////////////////////////////////////////

// // private computeDecorations() {
// //     this.currentCellTopUnfocused = this.documentManager.createTextEditorDecorationType({
// //         borderColor: new vscode.ThemeColor('interactive.inactiveCodeBorder'),
// //         borderWidth: '2px 0px 0px 0px',
// //         borderStyle: 'solid',
// //         isWholeLine: true
// //     });
// //     this.currentCellBottomUnfocused = this.documentManager.createTextEditorDecorationType({
// //         borderColor: new vscode.ThemeColor('interactive.inactiveCodeBorder'),
// //         borderWidth: '0px 0px 1px 0px',
// //         borderStyle: 'solid',
// //         isWholeLine: true
// //     });
// //     this.currentCellTop = this.documentManager.createTextEditorDecorationType({
// //         borderColor: new vscode.ThemeColor('interactive.activeCodeBorder'),
// //         borderWidth: '2px 0px 0px 0px',
// //         borderStyle: 'solid',
// //         isWholeLine: true
// //     });
// //     this.currentCellBottom = this.documentManager.createTextEditorDecorationType({
// //         borderColor: new vscode.ThemeColor('interactive.activeCodeBorder'),
// //         borderWidth: '0px 0px 1px 0px',
// //         borderStyle: 'solid',
// //         isWholeLine: true
// //     });
// // }

// // private cellDecorationEnabled(settings: IJupyterSettings) {
// //     // check old true/false value for this setting
// //     if ((settings.decorateCells as unknown as boolean) === false) {
// //         return false;
// //     }

// //     return settings.decorateCells === 'currentCell' || settings.decorateCells === 'allCells';
// // }

// // /**
// //  *
// //  * @param editor The editor to update cell decorations in.
// //  * If left undefined, this function will update all visible text editors.
// //  */
// // private update(editor: vscode.TextEditor | undefined) {
// //     // Don't look through all visible editors unless we have to i.e. the active editor has changed
// //     const editorsToCheck = editor === undefined ? this.documentManager.visibleTextEditors : [editor];
// //     for (const editor of editorsToCheck) {
// //         if (
// //             editor &&
// //             editor.document &&
// //             editor.document.languageId === PYTHON_LANGUAGE &&
// //             !getAssociatedJupyterNotebook(editor.document) &&
// //             this.currentCellTop &&
// //             this.currentCellBottom &&
// //             this.currentCellTopUnfocused &&
// //             this.currentCellBottomUnfocused &&
// //             this.extensionChecker.isPythonExtensionInstalled
// //         ) {
// //             const settings = this.configuration.getSettings(editor.document.uri);
// //             if (this.cellDecorationEnabled(settings)) {
// //                 // Find all of the cells
// //                 const cells = generateCellRangesFromDocument(editor.document, settings);
// //                 // Find the range for our active cell.
// //                 const currentRange = cells.map((c) => c.range).filter((r) => r.contains(editor.selection.anchor));
// //                 const rangeTop =
// //                     currentRange.length > 0 ? [new vscode.Range(currentRange[0].start, currentRange[0].start)] : [];
// //                 // no need to decorate the bottom if we're decorating all cells
// //                 const rangeBottom =
// //                     settings.decorateCells !== 'allCells' && currentRange.length > 0
// //                         ? [new vscode.Range(currentRange[0].end, currentRange[0].end)]
// //                         : [];
// //                 const nonCurrentCells: vscode.Range[] = [];
// //                 if (settings.decorateCells === 'allCells')
// //                     cells.forEach((cell) => {
// //                         const cellTop = cell.range.start;
// //                         if (cellTop !== currentRange[0].start) {
// //                             nonCurrentCells.push(new vscode.Range(cellTop, cellTop));
// //                         }
// //                     });
// //                 if (this.documentManager.activeTextEditor === editor) {
// //                     editor.setDecorations(this.currentCellTop, rangeTop);
// //                     editor.setDecorations(this.currentCellBottom, rangeBottom);
// //                     editor.setDecorations(this.currentCellTopUnfocused, nonCurrentCells);
// //                     editor.setDecorations(this.currentCellBottomUnfocused, []);
// //                 } else {
// //                     editor.setDecorations(this.currentCellTop, []);
// //                     editor.setDecorations(this.currentCellBottom, []);
// //                     editor.setDecorations(this.currentCellTopUnfocused, [...nonCurrentCells, ...rangeTop]);
// //                     editor.setDecorations(this.currentCellBottomUnfocused, rangeBottom);
// //                 }
// //             } else {
// //                 editor.setDecorations(this.currentCellTop, []);
// //                 editor.setDecorations(this.currentCellBottom, []);
// //                 editor.setDecorations(this.currentCellTopUnfocused, []);
// //                 editor.setDecorations(this.currentCellBottomUnfocused, []);
// //             }
// //         }
// //     }
// // }
// // }
