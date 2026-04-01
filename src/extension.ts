
// You can use this Python script to keep the TypeScript file in sync with the Python file:
// path_in = "src/reactive_python_engine.py"
// path_out = "src/reactive_python_engine.ts"
// with open(path_in, 'r') as file:
//     content = file.read().replace("\\n", "\\\\n")
// with open(path_out, 'w') as file:
//     file.write(f'export const scriptCode = `\n{content}\n`;')

import {
    Uri,
    Range,
    window,
    Position,
    languages,
    workspace,
    TextEditor,
    extensions,
    NotebookCell,
    OutputChannel,
    ExtensionContext,
    NotebookDocument,
    NotebookCellOutputItem,
    CancellationTokenSource,
    commands,
    Selection,
    ViewColumn,
    Disposable,
    QuickPickItem,
    NotebookEditor,
    CodeLensProvider,
    NotebookCellOutput,
    NotebookCellExecutionSummary
} from 'vscode';
import * as vscode from 'vscode';

import { TextDecoder } from 'text-encoding';
import { scriptCode } from './reactive_python_engine';
import { Jupyter, Kernel } from '@vscode/jupyter-extension';
import { generateInteractiveCode } from './codeStolenFromJupyter/generateInteractiveCode';
// import { CellOutputDisplayIdTracker } from './codeStolenFromJupyter/cellExecutionMessageHandler';


////////////////////////////////////////////////////////////////////////////////////////////////////
// UTILS
////////////////////////////////////////////////////////////////////////////////////////////////////


const ErrorMimeType = NotebookCellOutputItem.error(new Error('')).mime;
const StdOutMimeType = NotebookCellOutputItem.stdout('').mime;
const StdErrMimeType = NotebookCellOutputItem.stderr('').mime;
const MarkdownMimeType = 'text/markdown';
const HtmlMimeType = 'text/html';
const textDecoder = new TextDecoder();

// Debug mode - set to true to show more alerts
let DEBUG_MODE = false;

/**
 * Log debug messages and optionally show VS Code alerts
 */
function debugLog(message: string, level: 'info' | 'warn' | 'error' = 'info', alwaysAlert: boolean = false) {
    const fullMessage = `Reactive Jupyter: ${message}`;
    
    if (level === 'error') {
        console.error(fullMessage);
        if (DEBUG_MODE || alwaysAlert) {
            window.showErrorMessage(fullMessage);
        }
    } else if (level === 'warn') {
        console.warn(fullMessage);
        if (DEBUG_MODE || alwaysAlert) {
            window.showWarningMessage(fullMessage);
        }
    } else {
        console.log(fullMessage);
        if (DEBUG_MODE) {
            window.showInformationMessage(fullMessage);
        }
    }
}

/**
 * Wrap an async function with error handling that prevents the extension from getting stuck
 */
function withErrorRecovery<T extends any[], R>(
    fn: (...args: T) => Promise<R>,
    operationName: string,
    globals: Map<string, string>,
    getEditor?: (...args: T) => TextEditor | undefined
): (...args: T) => Promise<R | undefined> {
    return async (...args: T): Promise<R | undefined> => {
        try {
            return await fn(...args);
        } catch (error: any) {
            const errorMessage = error?.message || String(error);
            debugLog(`Error in ${operationName}: ${errorMessage}`, 'error', true);
            
            // Try to reset state for the editor if we can identify it
            if (getEditor) {
                const editor = getEditor(...args);
                if (editor) {
                    debugLog(`Resetting state for editor due to error in ${operationName}`, 'warn');
                    forceResetEditorState(globals, editor);
                }
            }
            
            return undefined;
        }
    };
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//    RUN PYTHON CODE: 2 WAYS: IN KERNEL AND IN INTERACTIVE WINDOW
////////////////////////////////////////////////////////////////////////////////////////////////////


async function* executeCodeStreamInKernel(code: string, kernel: Kernel, output_channel: OutputChannel | null): AsyncGenerator<string | ExecutionError, void, unknown> {
    /*
    Currently, it ALWAYS logs a line (for the user) if output_channel is defined, 
    and it returns the result if it is NOT AN ERROR, else Undefined. 
    If you need Debugging traces, use console.log(). (currently it logs errors)
    */

    if (output_channel) {output_channel.appendLine(`Executing code against kernel ${code}`);}
    const tokenSource = new CancellationTokenSource();
    try {
        for await (const output of kernel.executeCode(code, tokenSource.token)) {
            for (const outputItem of output.items) {
                const decoded = textDecoder.decode(outputItem.data);
                if (outputItem.mime === ErrorMimeType) {
                    const error = JSON.parse(decoded) as Error;
                    if (output_channel) {
                        if (output_channel) {output_channel.appendLine(`Error executing code ${error.name}: ${error.message},/n ${error.stack}`);}
                    }
                    yield { name: error.name, message: error.message, stack: error.stack } as ExecutionError;
                } else {
                    if (output_channel) {
                        if (output_channel) {output_channel.appendLine(`${outputItem.mime} Output: ${decoded}`);}
                    }
                    yield decoded;
                    if (output_channel) {
                        if (output_channel) {output_channel.appendLine('Code execution completed');}
                    }
                }
            }
        }
    }
    catch (error: any) {
        const errorMessage = error?.message || String(error);
        debugLog(`Kernel execution failed: ${errorMessage}`, 'error', true);
        yield { name: 'KernelExecutionError', message: errorMessage, stack: error?.stack || '' } as ExecutionError;
    }
    finally {
        tokenSource.dispose();
    }
}

async function executeCodeInKernel(code: string, kernel: Kernel, output_channel: OutputChannel | null): Promise<string | ExecutionError> {
    let result = '';
    for await (const output of executeCodeStreamInKernel(code, kernel, output_channel)) {
        if (isExecutionError(output)) {
            return output;
        }
        else if (output !== undefined) {
            result += output;
        }
    }
    return result;
}

enum IWExecutionResult {
    Succeeded = 'Succeeded',
    Failed = 'Failed',
    NotebookClosed = 'NotebookClosed'
}

async function executeCodeInInteractiveWindow(
    text: string,
    notebook: NotebookDocument,
    output: OutputChannel | null,
    targetTextDocument: vscode.TextDocument | undefined = undefined,
    targetViewColumn: ViewColumn | undefined = undefined, // The original viewColumn where the editor was - ensures we open in the SAME tab group
): Promise<IWExecutionResult> {
    // Currently returns an IWExecutionResult. TODO: Rethink if you need something more...

    // IMPORTANT: jupyter.execSelectionInteractive uses the ACTIVE TEXT EDITOR to determine which 
    // interactive window to use (when interactiveWindow.creationMode is perFile).
    // So we need to ensure the correct text editor is active before executing.
    // By specifying viewColumn, we ensure it opens in the ORIGINAL tab group (not a new one if user moved away).
    if (targetTextDocument) {
        try {
            await window.showTextDocument(targetTextDocument, { 
                viewColumn: targetViewColumn, 
                preserveFocus: false, // Need to actually take focus for jupyter.execSelectionInteractive to work
                preview: false 
            });
        } catch (e: any) {
            debugLog(`Failed to set active text editor before execution: ${e?.message || e}`, 'warn');
            // Continue anyway - maybe it will still work
        }
    }

    try {
        await vscode.commands.executeCommand('jupyter.execSelectionInteractive', text);
    } catch (e: any) {
        debugLog(`jupyter.execSelectionInteractive command failed: ${e?.message || e}`, 'error', true);
        return IWExecutionResult.Failed;
    }
    // OTHER THINGS I TRIED:
    // let res = await getIWAndRunText(serviceManager, activeTextEditor, text);
    // let res = await executeCodeInKernel(text, kernel, output);
    // let res = await interactiveWindow.addNotebookCell( text, textEditor.document.uri, textEditor.selection.start.line, notebook )
    
    // ^ In particular, it would be a REALLY GOOD IDEA to do addNotebookCell ^ (which already works)
    // and then TAKE CONTROL OF THE WHOLE cellExecutionQueue mechanism, an Reimplement it here, by ALWAYS SENDING THINGS TO THE KERNEL IMPLICITELY 
    // and then streaming the output to the NotebookDocyment ourselves, the same way Jupyter does it.. 
    // PROBLEM: That's a Lot of work. For now, I'll take advantage of the execSelectionInteractive Command because it's Easier...
    
    let cell: NotebookCell | undefined = undefined;
    const MAX_CELL_WAIT_ITERATIONS = 80;  // ~20 seconds max wait for cell to appear
    for (let i = 0; i < MAX_CELL_WAIT_ITERATIONS; i++) {
        await new Promise((resolve) => setTimeout(resolve, 250));
        try {
            if (notebook.isClosed) {
                debugLog('Notebook was closed while waiting for cell', 'warn');
                return IWExecutionResult.NotebookClosed;
            }
            let lastCell = notebook.cellAt(notebook.cellCount - 1);
            let last_index = notebook.cellCount - 1;
            cell = getBestMatchingCell(lastCell, notebook, last_index, text);
            if (cell) { break; }
        } catch (e: any) {
            debugLog(`Error while waiting for cell: ${e?.message || e}`, 'warn');
            // Continue trying
        }
    }

    if (!cell) {
        debugLog('Failed to execute code in Interactive Window: No matching cell was identified after timeout', 'error', true);
        return IWExecutionResult.NotebookClosed;
    }

    let cell_state = CellState.Undefined;
    await new Promise((resolve) => setTimeout(resolve, 250));
    
    const MAX_EXECUTION_WAIT_ITERATIONS = 600;  // ~150 seconds max wait for execution (long-running cells)
    for (let i = 0; i < MAX_EXECUTION_WAIT_ITERATIONS; i++) {
        try {
            if (!notebook || notebook.isClosed || cell.notebook.isClosed) {
                return IWExecutionResult.NotebookClosed;
            }
            cell_state = getCellState(cell);
            if (cell_state === CellState.Success) { 
                if (has_error_mime(cell)) {
                    return IWExecutionResult.Failed;
                }
                return IWExecutionResult.Succeeded; 
            }
            else if (cell_state === CellState.Error) { 
                return IWExecutionResult.Failed; 
            }
        } catch (e: any) {
            debugLog(`Error while checking cell state: ${e?.message || e}`, 'warn');
            // Continue trying
        }
        await new Promise((resolve) => setTimeout(resolve, 250));
    }

    debugLog('Cell execution did not complete within timeout (150s). You may want to restart.', 'error', true);
    return IWExecutionResult.Failed;
}

async function safeExecuteCodeInKernel(
    command: string,
    editor: TextEditor,
    output: OutputChannel | null,
    globals: Map<string, string>,
    expected_initial_state: KernelState = KernelState.extension_available,
    return_to_initial_state: boolean = true
) {
    try {
        displayInitializationMessageIfNeeded(globals, editor);
        if (getKernelState(globals, editor) !== expected_initial_state) { return; }
        if (!checkSettings(globals, editor)) { return; }

        let notebookAndKernel = await getNotebookAndKernel(globals, editor, true);
        if (!notebookAndKernel) {
            debugLog("Lost Connection to this editor's Kernel. Please use 'Initialize Reactive Jupyter' or 'Force Reset Reactive Jupyter' command.", 'error', true);
            updateKernelState(globals, editor, KernelState.initializable_messaged);
            return;
        }
        let [notebook, kernel] = notebookAndKernel;
        updateKernelState(globals, editor, KernelState.implicit_execution_started);
        const result = await executeCodeInKernel(command, kernel, null);  // output ?
        if (return_to_initial_state) { updateKernelState(globals, editor, expected_initial_state); }
        return result;
    } catch (error: any) {
        debugLog(`safeExecuteCodeInKernel failed: ${error?.message || error}. Resetting to initializable state.`, 'error', true);
        // Reset to a safe state
        try {
            updateKernelState(globals, editor, KernelState.initializable_messaged);
        } catch {
            // If even that fails, force reset
            forceResetEditorState(globals, editor);
        }
        return undefined;
    }
}
async function safeExecuteCodeInKernelForInitialization(
    command: string,
    editor: TextEditor,
    output: OutputChannel | null,
    globals: Map<string, string>
): Promise<boolean> {
    try {
        // It's SLIGHTLY different from the above one, in ways I didn't bother to reconcile...
        if (getKernelState(globals, editor) !== KernelState.kernel_available) { return false; }
        if (!checkSettings(globals, editor)) { return false; }

        let notebookAndKernel = await getNotebookAndKernel(globals, editor, true);
        if (!notebookAndKernel) {
            debugLog("Kernel Initialization succeeded, but we lost it already... Please try again.", 'error', true);
            updateKernelState(globals, editor, KernelState.initializable_messaged);
            return false;
        }
        let [notebook, kernel] = notebookAndKernel;
        updateKernelState(globals, editor, KernelState.instantialization_started);
        const result = await executeCodeInKernel(command, kernel, null);  // output ?
        if (!isExecutionError(result)) {
            updateKernelState(globals, editor, KernelState.extension_available);
            window.showInformationMessage('Reactive Jupyter: The extension is ready to use.');
            return true;
        } else {
            updateKernelState(globals, editor, KernelState.kernel_available);
            debugLog('The initialization code could not be executed in the Python Kernel. Try restarting the kernel.', 'error', true);
            return false;
        }
    } catch (error: any) {
        debugLog(`safeExecuteCodeInKernelForInitialization failed: ${error?.message || error}`, 'error', true);
        // Reset to a safe state
        try {
            updateKernelState(globals, editor, KernelState.initializable_messaged);
        } catch {
            forceResetEditorState(globals, editor);
        }
        return false;
    }
}

async function safeExecuteCodeInInteractiveWindow(
    command: string,
    editor: TextEditor,
    output: OutputChannel | null,
    globals: Map<string, string>,
    expected_initial_state: KernelState = KernelState.extension_available,
    return_to_initial_state: boolean = true,
    targetTextDocument: vscode.TextDocument | undefined = undefined,
    targetViewColumn: ViewColumn | undefined = undefined, // The original viewColumn - ensures we open in the SAME tab group
) {
    try {
        displayInitializationMessageIfNeeded(globals, editor);
        if (getKernelState(globals, editor) !== expected_initial_state) { return; }
        if (!checkSettings(globals, editor)) { return; }

        let notebookAndKernel = await getNotebookAndKernel(globals, editor, true);
        if (!notebookAndKernel) {
            debugLog("Lost Connection to this editor's Notebook. Please use 'Initialize Reactive Jupyter' or 'Force Reset Reactive Jupyter' command.", 'error', true);
            updateKernelState(globals, editor, KernelState.initializable_messaged);
            return;
        }
        let [notebook, kernel] = notebookAndKernel;
        
        updateKernelState(globals, editor, KernelState.explicit_execution_started);
        // Pass the target text document + viewColumn so the execution happens in the correct interactive window
        const result = await executeCodeInInteractiveWindow(command, notebook, output, targetTextDocument, targetViewColumn);
        if (result == IWExecutionResult.NotebookClosed) {
            debugLog("Lost Connection to this editor's Notebook. Please use 'Initialize Reactive Jupyter' or 'Force Reset Reactive Jupyter' command.", 'error', true);
            updateKernelState(globals, editor, KernelState.initializable_messaged);
        }
        else if (return_to_initial_state) { updateKernelState(globals, editor, expected_initial_state); }
        return result;
    } catch (error: any) {
        debugLog(`safeExecuteCodeInInteractiveWindow failed: ${error?.message || error}. Resetting to initializable state.`, 'error', true);
        // Reset to a safe state
        try {
            updateKernelState(globals, editor, KernelState.initializable_messaged);
        } catch {
            forceResetEditorState(globals, editor);
        }
        return undefined;
    }
}


type AnnotatedRange = {
    range: Range;
    state: string; // Remember this exists too: 'synced' | 'outdated';
    current: boolean;
    text?: string;
    hash?: string; // Hash is used so that when you send a node Back to Python, you can check if it actually him or not
    has_children?: boolean;
};

async function queueComputation(
    current_ranges: AnnotatedRange[] | undefined,
    activeTextEditor: TextEditor,
    globals: Map<string, string>,
    output: OutputChannel,
) {
    // IMPORTANT: Capture the text document AND viewColumn at queue time. 
    // jupyter.execSelectionInteractive uses the ACTIVE TEXT EDITOR to determine which interactive window to use
    // (when interactiveWindow.creationMode is perFile). So we need to refocus this editor before each execution.
    // By also capturing viewColumn, we ensure showTextDocument opens it in the SAME tab group (not a new one).
    const queueTargetTextDocument = activeTextEditor.document;
    const queueTargetViewColumn = activeTextEditor.viewColumn;
    
    try {
        if (current_ranges) {
            let said_dependsonotherstalecode_message = false;
            for (let range of current_ranges) {
                if (!range.text) break;

                if (range.state === 'dependsonotherstalecode') {
                    if (!said_dependsonotherstalecode_message) {
                        let text = range.text.slice(0, 100);
                        debugLog(text + ' depends on other code that is outdated. Please update the other code first.', 'error', true);
                        said_dependsonotherstalecode_message = true;
                    }   
                    continue;
                }

                // Check if the target document is still open before each execution
                if (queueTargetTextDocument.isClosed) {
                    debugLog('The target Python file was closed during queue execution. Aborting remaining commands.', 'error', true);
                    break;
                }

                let res = await safeExecuteCodeInInteractiveWindow(
                    range.text, 
                    activeTextEditor, 
                    output, 
                    globals,
                    KernelState.extension_available,
                    true,
                    queueTargetTextDocument,  // Pass the captured text document to ensure all commands go to the same interactive window
                    queueTargetViewColumn     // Pass the captured viewColumn to ensure it opens in the SAME tab group
                );
                if ( res != IWExecutionResult.Succeeded ) { break; }

                const update_result = await safeExecuteCodeInKernel(getSyncRangeCommand(range), activeTextEditor, output, globals);
                if (!update_result) {
                    debugLog("Failed to update the range's state in Python: " + range.hash + " -- " + update_result, 'error');
                    break;
                }
                const refreshed_ranges = await getCurrentRangesFromPython(activeTextEditor, output, globals, {
                    rebuild: false
                });
                if (refreshed_ranges) {
                    updateDecorations(activeTextEditor, refreshed_ranges);
                }
                else {
                    updateDecorations(activeTextEditor, []);
                }
            }
        }
        const update_result = await safeExecuteCodeInKernel(getUnlockCommand(), activeTextEditor, output, globals);
        if (getKernelState(globals, activeTextEditor) == KernelState.extension_available && !update_result) {
            debugLog('Failed to unlock the Python kernel: ' + update_result, 'error');
        }
    } catch (error: any) {
        debugLog(`queueComputation failed: ${error?.message || error}. Attempting to unlock and reset.`, 'error', true);
        // Try to unlock anyway to prevent deadlocks
        try {
            await safeExecuteCodeInKernel(getUnlockCommand(), activeTextEditor, output, globals);
        } catch {
            // If unlock fails, we may need a force reset
            debugLog("Failed to unlock after error. Consider using 'Force Reset Reactive Jupyter' command.", 'warn', true);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//    SOME HELPER FUNCTIONS:
////////////////////////////////////////////////////////////////////////////////////////////////////


const errorMimeTypes = ['application/vnd.code.notebook.error']; // 'application/vnd.code.notebook.stderr' ? But like, TQDM outputs this even tho it succedes..

function has_error_mime(cell: NotebookCell): boolean {
    for (let i = 0; i < cell.outputs.length; i++) {
        for (let j = 0; j <  cell.outputs[i].items.length; j++) {
            if (errorMimeTypes.includes( cell.outputs[i].items[j].mime)) { 
                return true; 
            }
        }
    }
    return false;
}

// An Error type:
type ExecutionError = {
    name: string;
    message: string;
    stack: string;
};

function isExecutionError(obj: any): obj is ExecutionError {  // Typeguard
    return obj && (obj as ExecutionError).stack !== undefined; 
}

enum CellState {
    Success = 'success',
    Error = 'error',
    Undefined = 'undefined'
}

function getCellState(cell: NotebookCell): CellState {
    if (cell.executionSummary && cell.executionSummary.success !== undefined) { return cell.executionSummary.success ? CellState.Success : CellState.Error; }
    else { return CellState.Undefined; }
}

function getBestMatchingCell(cell: NotebookCell, nb: NotebookDocument, last_idx: number, text: string): NotebookCell | undefined {
    // First check the cell at last_idx. If text doesnt match, go BACK FROM THE LAST ONE until you find a match, else undefined.
    if (cell.document.getText() === generateInteractiveCode(text)) { return cell; }
    let cell_id = nb.cellAt(last_idx);
    if (cell_id.document.getText() === generateInteractiveCode(text)) { return cell_id; }
    let num_cells = nb.cellCount;
    for (let i = num_cells - 1; i >= 0; i--) {
        let cell_i = nb.cellAt(i);
        if (cell_i.document.getText() === generateInteractiveCode(text)) { return cell_i; }
    }
    return undefined;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//    JUST A STATE MACHINE
////////////////////////////////////////////////////////////////////////////////////////////////////


function getState<STATES>(globals: Map<string, string>, key: string): STATES | boolean {
    return (globals.get(key) as STATES) || false;
}

/**
 * Attempts to update state following valid transitions. If the transition is invalid,
 * it logs a warning but does NOT throw - this prevents the extension from getting stuck.
 * Returns true if the transition was valid, false otherwise.
 */
function updateState<STATES>(globals: Map<string, string>, key: string, newState_: string, stateTransitions: Map<STATES, STATES[]>, initialStates: STATES[] = []): boolean {
    let newState = newState_ as STATES;
    if (!newState) {
        console.error('Reactive Jupyter: Invalid state value: ' + newState);
        window.showWarningMessage('Reactive Jupyter: Encountered invalid state. You can use "Force Reset Reactive Jupyter" command to recover.');
        return false;
    }
    let currentState = (globals.get(key) as STATES) || false;
    if (!currentState) {
        if (initialStates.includes(newState)) {
            globals.set(key, newState as string);
            return true;
        }
        else {
            console.warn('Reactive Jupyter: Invalid initial state transition: ' + currentState + ' -> ' + newState);
            // Don't throw - just set to the initial state anyway to allow recovery
            globals.set(key, newState as string);
            return false;
        }
    }
    let acceptedTransitions = stateTransitions.get(currentState as STATES);
    if (acceptedTransitions && acceptedTransitions.includes(newState)) {
        globals.set(key, newState as string);
        return true;
    }
    else {
        console.warn('Reactive Jupyter: Invalid state transition: ' + currentState + ' -> ' + newState + '. Allowing anyway to prevent getting stuck.');
        // Don't throw - set the state anyway to prevent getting stuck
        globals.set(key, newState as string);
        return false;
    }
}

/**
 * Force reset all state for a given editor back to initial state.
 * This allows recovery from any stuck state.
 */
function forceResetEditorState(globals: Map<string, string>, editor: TextEditor) {
    const editorUri = editor.document.uri.toString();
    
    // Clear all state keys related to this editor
    globals.delete(editorConnectionStateKey(editorUri));
    globals.delete(editorRebuildKey(editorUri));
    globals.delete(editorLastEventTimestampKey(editorUri));
    globals.delete(editorToIWKey(editorUri));
    globals.delete(editorToKernelKey(editorUri));
    
    console.log('Reactive Jupyter: Force reset all state for editor: ' + editorUri);
}

/**
 * Force reset ALL global state - nuclear option for recovery
 */
function forceResetAllState(globals: Map<string, string>) {
    globals.clear();
    console.log('Reactive Jupyter: Force reset ALL global state');
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//    CONNECT TO INTERACTIVE WINDOW AS WELL AS KERNEL:
////////////////////////////////////////////////////////////////////////////////////////////////////


enum KernelState {
    settings_not_ok = 'settings_not_ok',
    initializable = 'initializable',
    initializable_messaged = 'initializable_messaged',
    initialization_started = 'initialization_started',
    kernel_found = 'kernel_found',
    kernel_available = 'kernel_available',
    instantialization_started = 'instantialization_started',
    extension_available = 'extension_available',
    explicit_execution_started = 'explicit_execution_started',
    implicit_execution_started = 'implicit_execution_started',
}

const kernelIinitialStates = [KernelState.settings_not_ok, KernelState.initializable_messaged];

const kernelStateTransitions: Map<KernelState, KernelState[]> = new Map([
    [KernelState.settings_not_ok, [KernelState.initializable]],
    [KernelState.initializable, [KernelState.initialization_started].concat(kernelIinitialStates)],
    [KernelState.initializable_messaged, [KernelState.initialization_started, KernelState.settings_not_ok]],
    [KernelState.initialization_started, [KernelState.kernel_found].concat(kernelIinitialStates)],
    [KernelState.kernel_found, [KernelState.kernel_available].concat(kernelIinitialStates)],
    [KernelState.kernel_available, [KernelState.instantialization_started, KernelState.extension_available].concat(kernelIinitialStates)],
    [KernelState.instantialization_started, [KernelState.extension_available].concat(kernelIinitialStates)],
    [KernelState.extension_available, [KernelState.explicit_execution_started, KernelState.implicit_execution_started].concat(kernelIinitialStates)],
    [KernelState.explicit_execution_started, [KernelState.extension_available].concat(kernelIinitialStates)],
    [KernelState.implicit_execution_started, [KernelState.extension_available].concat(kernelIinitialStates)],
]);

function updateKernelState(globals: Map<string, string>, editor: TextEditor, newState_: string) {
    const key = editorConnectionStateKey(editor.document.uri.toString());
    updateState<KernelState>(globals, key, newState_, kernelStateTransitions, kernelIinitialStates);
}

function getKernelState(globals: Map<string, string>, editor: TextEditor): KernelState | boolean {
    return getState<KernelState>(globals, editorConnectionStateKey(editor.document.uri.toString()));
}


function checkSettings(globals: Map<string, string>, editor: TextEditor,) {
    // After this function, you are: in settings_not_ok if they are not ok, or in THE SAME PREVIOUS STATES if they are, except if you were in settings_not_ok, in which case you are in initializable
    // Obviously, returns True if settings are ok, else False

    const resourceUri = editor.document.uri;
    const creationMode = vscode.workspace.getConfiguration('jupyter', resourceUri).get<string>('interactiveWindow.creationMode');
    const shiftEnter = vscode.workspace.getConfiguration('jupyter', resourceUri).get<boolean>('interactiveWindow.textEditor.executeSelection');
    const perFileMode = creationMode && ['perFile', 'perfile'].includes(creationMode);
    const shiftEnterOff = shiftEnter === false;

    if (perFileMode && shiftEnterOff && getKernelState(globals, editor) === KernelState.settings_not_ok) {
        updateKernelState(globals, editor, KernelState.initializable);
        return true;
    }
    else if (!perFileMode || !shiftEnterOff) {
        if (getKernelState(globals, editor) !== KernelState.settings_not_ok) { 
            updateKernelState(globals, editor, KernelState.settings_not_ok); 
            window.showErrorMessage('Reactive Jupyter: To use Reactive Jupyter please set the Jupyter extension settings to:  - "jupyter.interactiveWindow.creationMode": "perFile"   - "jupyter.interactiveWindow.textEditor.executeSelection": false');
        }
        return false;
    }
    else {
        return true;
    }
}

function displayInitializationMessageIfNeeded(globals: Map<string, string>, editor: TextEditor) {
    if (getKernelState(globals, editor) === KernelState.initializable) {
        window.showInformationMessage("Reactive Jupyter: Initialize the extension on this File with the command: 'Initialize Reactive Jupyter' or the CodeLens at the top");
        updateKernelState(globals, editor, KernelState.initializable_messaged);
    }
}

async function getKernelNotebook(document: NotebookDocument): Promise<Kernel | undefined> {
    const extension = extensions.getExtension<Jupyter>('ms-toolsai.jupyter');
    if (!extension) {
        window.showErrorMessage('Reactive Jupyter: Jupyter extension not installed');
        throw new Error('Reactive Jupyter: Jupyter extension not installed');
    }
    if (!extension.isActive) { await extension.activate(); }
    const api = extension.exports;
    return new Promise<Kernel | undefined>(async (resolve) => {
        const kernel = await api.kernels.getKernel(document.uri);
        if (kernel && (kernel as any).language === 'python') { resolve(kernel); } else { resolve(undefined); }
    });
}

function isThereANewNotebook(oldNotebooks: readonly CachedNotebookDocument[], newNotebooks: readonly NotebookDocument[]): NotebookDocument | undefined {
    // This returns a notebook, if either there is in BOTH OLD AND NEW, but in New it has ONE MORE CELL.
    // Or it is ONLY IN NEW, and it has at least one cell.
    let newNotebook: NotebookDocument | undefined = undefined;
    for (let i = 0; i < newNotebooks.length; i++) {
        let oldNotebook = oldNotebooks.find((doc) => doc.uri.toString() === newNotebooks[i].uri.toString());
        if (oldNotebook && newNotebooks[i].cellCount > oldNotebook.cellCount) { newNotebook = newNotebooks[i]; break; }
        else if (!oldNotebook && newNotebooks[i].cellCount > 0) { newNotebook = newNotebooks[i]; break; }
    }
    return newNotebook;
}


async function getNotebookAndKernel(globals: Map<string, string>, editor: TextEditor, notify: Boolean = false): Promise<[NotebookDocument, Kernel] | undefined> {
    let notebook_uri = globals.get(editorToIWKey(editor.document.uri.toString()));
    let iWsWCorrectUri = vscode.workspace.notebookDocuments.filter((doc: NotebookDocument) => doc.uri.toString() === notebook_uri);
    if (iWsWCorrectUri.length === 0) {
        if (notify) { window.showErrorMessage("Reactive Jupyter: Lost connection to this editor's Interactive Window. Please initialize it with the command: 'Initialize Reactive Jupyter' or the CodeLens at the top ") }
        return undefined;
    }
    let notebook = iWsWCorrectUri[0];
    let kernel = await getKernelNotebook(notebook);
    if (!kernel) {
        if (notify) { window.showErrorMessage("Reactive Jupyter: Lost connection to this editor's Python Kernel. Please initialize it with the command 'Initialize Reactive Jupyter' or the CodeLens at the top ") }
        return undefined;
    }
    return [notebook, kernel];
}


const editorToIWKey = (editorUri: string) => 'editorToIWKey' + editorUri;
const editorToKernelKey = (editorUri: string) => 'editorToIWKey' + editorUri;
const editorConnectionStateKey = (editorUri: string) => 'state' + editorUri;
const editorRebuildKey = (editorUri: string) => 'editorRebuild' + editorUri;
const editorLastEventTimestampKey = (editorUri: string) => 'editorLastEventTimestamp' + editorUri;


type CachedNotebookDocument = { cellCount: number, uri: Uri };
const toMyNotebookDocument = (doc: NotebookDocument): CachedNotebookDocument => ({ cellCount: doc.cellCount, uri: doc.uri });


async function initializeInteractiveWindowAndKernel(globals: Map<string, string>, editor: TextEditor) {

    let currentState = getKernelState(globals, editor)
    if (currentState !== KernelState.initializable && currentState !== KernelState.initializable_messaged) {
        debugLog('Invalid state for initialization: ' + currentState + '. Use "Force Reset Reactive Jupyter" to reset.', 'warn');
        return false;
    }

    // Start initializing:
    updateKernelState(globals, editor, KernelState.initialization_started);

    let n_attempts = 5;
    while (n_attempts > 0) {
        try {
            const notebookDocuments: CachedNotebookDocument[] = vscode.workspace.notebookDocuments.map(toMyNotebookDocument);
            // Wreck some Havoc: This should ALWAYS RESULT IN A RUNNING KERNEL, AND ALSO A NEW CELL SOMEWHERE, EVENTUALLY. This is the idea, at least...
            await vscode.commands.executeCommand('jupyter.execSelectionInteractive', autoreloadText);
            await vscode.commands.executeCommand('jupyter.execSelectionInteractive', welcomeText);
            // Other things I tried:
            // let resIW = await vscode.commands.executeCommand('jupyter.createnewinteractive') as Uri;
            // let resIW = await interactiveWindow.createEditor(undefined, editor.document.uri);

            let newNotebook: NotebookDocument | undefined = undefined;
            for (let i = 0; i < 50; i++) {  // Try 50 times to read the amount of notebooks open:
                await new Promise((resolve) => setTimeout(resolve, 100));
                newNotebook = isThereANewNotebook(notebookDocuments, vscode.workspace.notebookDocuments);
                if (newNotebook) { break; }
            }
            if (!newNotebook) { 
                n_attempts -= 1; 
                debugLog(`No new notebook found, ${n_attempts} attempts remaining...`, 'warn');
                continue;
            }

            globals.set(editorToIWKey(editor.document.uri.toString()), newNotebook.uri.toString());
            await new Promise((resolve) => setTimeout(resolve, 2000));
            let okFoundKernel: [NotebookDocument, Kernel] | undefined = undefined;
            for (let i = 0; i < 10; i++) {
                okFoundKernel = await getNotebookAndKernel(globals, editor,);
                if (okFoundKernel) { break; }
                window.showInformationMessage('Waiting for the Python Kernel to start...');
                await new Promise((resolve) => setTimeout(resolve, 2000));
            }

            if (!okFoundKernel) { 
                n_attempts -= 1; 
                debugLog(`No kernel found, ${n_attempts} attempts remaining...`, 'warn');
                continue;
            }
            updateKernelState(globals, editor, KernelState.kernel_found);

            let [notebook, kernel] = okFoundKernel;
            let is_last_cell_ok = false;
            for (let i = 0; i < 20; i++) {  // Try 20 times to read the last cell:
                await new Promise((resolve) => setTimeout(resolve, 100));
                try {
                    let lastCell = notebook.cellAt(notebook.cellCount - 1);
                    is_last_cell_ok = lastCell.document.getText() === welcomeText;
                    if (is_last_cell_ok) { break; }
                } catch (e) {
                    // Notebook cell access might fail if notebook is being modified
                    continue;
                }
            }
            if (!is_last_cell_ok) { 
                n_attempts -= 1; 
                updateKernelState(globals, editor, KernelState.initialization_started);
                debugLog(`Welcome cell not found, ${n_attempts} attempts remaining...`, 'warn');
                continue;
            }
            updateKernelState(globals, editor, KernelState.kernel_available);
            break;
        } catch (error: any) {
            n_attempts -= 1;
            debugLog(`Initialization attempt failed: ${error?.message || error}. ${n_attempts} attempts remaining.`, 'warn');
            if (n_attempts === 0) {
                break;
            }
        }
    }

    // After this, the only possible states SHOULD be KernelState.kernel_available or KernelState.initialization_started:

    let state_now = getKernelState(globals, editor)
    if (state_now === KernelState.initialization_started) {
        debugLog('Failed to initialize the Interactive Window and the Python Kernel after all attempts. Try "Force Reset Reactive Jupyter".', 'error', true);
        updateKernelState(globals, editor, KernelState.initializable_messaged);
        return false;
    }
    else if (state_now === KernelState.kernel_available) {
        window.showInformationMessage('Reactive Jupyter: Successfully initialized the Interactive Window and the Python Kernel');
        return true;
    }
    else {
        // Don't throw - just reset and return false
        debugLog('Unexpected state after initialization: ' + state_now + '. Resetting.', 'error', true);
        forceResetEditorState(globals, editor);
        return false;
    }
}



async function preparePythonEnvForReactivePython(editor: TextEditor, globals: Map<string, string>, output: OutputChannel) {
    try {
        let command = scriptCode + '\n\n\n"Reactive Jupyter Activated"\n';
        // if (output) { output.show(true); }

        if (getKernelState(globals, editor) == false) { globals.set(editorConnectionStateKey(editor.document.uri.toString()), KernelState.initializable); }

        checkSettings(globals, editor);

        if (editor.document.languageId !== 'python') {
            debugLog('This extension only works when editing Python files. Please open a Python file and try again.', 'error', true);
            return;
        }

        if (getKernelState(globals, editor) == (KernelState.initializable) || getKernelState(globals, editor) == (KernelState.initializable_messaged)) {
            let success = await initializeInteractiveWindowAndKernel(globals, editor);
            if (!success) { return; }
        }
        // Here, you should ALWAYS be in KernelState.kernel_available ...

        let instantiated_script = await safeExecuteCodeInKernelForInitialization(command, editor, output, globals);
        if (!instantiated_script) { return; }

        // Immediately start coloring ranges:

        // updateEditingState(globals, editor, EditingState.rebuilt);
        // I'm taking the liberty of overriding the State Machine here, because I can, and because inititializingthe extension should Always reset that state anyway
        const key = editorRebuildKey(editor.document.uri.toString());
        globals.set(key, EditingState.rebuilt as string)
        let refreshed_ranges = await getCurrentRangesFromPython(editor, null, globals, {
            rebuild: true,
            current_line: null
        }); 
        if (refreshed_ranges) {
            updateDecorations(editor, refreshed_ranges);
        }
        else{
            updateDecorations(editor, []);
        }
    } catch (error: any) {
        debugLog(`preparePythonEnvForReactivePython failed: ${error?.message || error}`, 'error', true);
        // Reset to allow recovery
        forceResetEditorState(globals, editor);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// PYTHON COMMANDS AND SNIPPETS
////////////////////////////////////////////////////////////////////////////////////////////////////

const autoreloadText = `
%load_ext autoreload
%autoreload 2`

const welcomeText = "# Welcome to Reactive Jupyter";

const getCommandToGetAllRanges = (
    text: string | null,
    current_line: number | null,
    upstream: boolean,
    downstream: boolean,
    stale_only: boolean,
    to_launch_compute: boolean = false
): string => {
    let text_ = text || 'None';
    let current_line_str: string = current_line ? current_line.toString() : 'None';
    let upstream_param: string = upstream ? 'True' : 'False';
    let downstream_param: string = downstream ? 'True' : 'False';
    let stale_only_param: string = stale_only ? 'True' : 'False';
    if (to_launch_compute) {
        return `reactive_python_dag_builder_utils__.ask_for_ranges_to_compute(code= ${text_}, current_line=${current_line_str}, get_upstream=${upstream_param}, get_downstream=${downstream_param}, stale_only=${stale_only_param})`;
    } else {
        return `reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code= ${text_}, current_line=${current_line_str}, get_upstream=${upstream_param}, get_downstream=${downstream_param}, stale_only=${stale_only_param})`;
    }
};

const getSyncRangeCommand = (range: AnnotatedRange): string => {
    return `reactive_python_dag_builder_utils__.set_locked_range_as_synced(${range.hash})`;
};

const getUnlockCommand = (): string => {
    return `reactive_python_dag_builder_utils__.unlock()`;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// GET RANGES FROM PYTHON
////////////////////////////////////////////////////////////////////////////////////////////////////

const getEditorAllText = (editor: TextEditor): { text: string | null } => {
    if (!editor || !editor.document || editor.document.uri.scheme === 'output') {
        return {
            text: null
        };
    }
    const text = editor.document.getText();
    return { text };
};

function formatTextAsPythonString(text: string) {
    text = text.replace(/\\/g, '\\\\');
    // >> You MIGHT be interested in vscode-jupyter/src/platform/terminals/codeExecution/codeExecutionHelper.node.ts  >> CodeExecutionHelper >> normalizeLines  ...
    text = text.replace(/'/g, "\\'");
    text = text.replace(/"/g, '\\"');
    text = text.replace(/\n/g, '\\n');
    text = text.replace(/\r/g, '\\r');
    text = text.replace(/\t/g, '\\t');
    text = '"""' + text + '"""';
    return text;
}

const getEditorCurrentLineNum = (editor: TextEditor): number | null => {
    if (!editor || !editor.document || editor.document.uri.scheme === 'output') {
        return null;
    }
    const currentLineNum = editor.selection.active.line;
    return currentLineNum;
};

const recognized_states = ['synced', 'outdated', 'syntaxerror', 'dependsonotherstalecode']

const parseResultFromPythonAndGetRange = (resultFromPython: string): AnnotatedRange[] | null => {
    // ResultFromPython is a string of the form: "[[startLine, endline, state, current, text, hash], [startLine, endline, state, current, text, hash], ...]" (the length is indefinite)
    // Parse it and return the list of ranges to select:

    try {
        // Result is returned as String, so remove the first and last character to get the Json-parsable string:
        resultFromPython = resultFromPython.substring(1, resultFromPython.length - 1);
        // Sanitize: To be clear, I'm ALMOST SURE this is a terrible idea...
        resultFromPython = resultFromPython.replace(/\\\\/g, '\\');
        resultFromPython = resultFromPython.replace(/\\"/g, '\\"');
        resultFromPython = resultFromPython.replace(/\\'/g, "'");
        resultFromPython = resultFromPython.replace(/\\n/g, '\\n');
        resultFromPython = resultFromPython.replace(/\\r/g, '\\r');
        resultFromPython = resultFromPython.replace(/\\t/g, '\\t');

        // Json parse:
        let resultFromPythonParsed;
        try {
            resultFromPythonParsed = JSON.parse(resultFromPython);
        } catch (e: any) {
            debugLog(`Failed to parse JSON result from Python: ${e?.message || e}. Raw: ${resultFromPython.slice(0, 200)}...`, 'error');
            return null;
        }
        // Assert that it worked:
        if (resultFromPythonParsed === undefined) {
            debugLog('Failed to parse result from Python: result is undefined', 'warn');
            return null;
        }
        // Convert to Range[]:
        let ranges: AnnotatedRange[] = [];
        for (let i = 0; i < resultFromPythonParsed.length; i++) {
            try {
                let startLine = resultFromPythonParsed[i][0];
                let endLine = resultFromPythonParsed[i][1];
                let state = resultFromPythonParsed[i][2];
                let current = resultFromPythonParsed[i][3] == 'current';
                let text_ = resultFromPythonParsed[i].length > 4 ? resultFromPythonParsed[i][4] : undefined;
                let hash = resultFromPythonParsed[i].length > 5 ? resultFromPythonParsed[i][5] : undefined;
                if (startLine >= 0 && endLine >= 0 && startLine <= endLine && recognized_states.includes(state)) {
                    // Parse as int:
                    startLine = parseInt(startLine);
                    endLine = parseInt(endLine);
                    ranges.push({
                        range: new Range(new Position(startLine, 0), new Position(endLine + 1, 0)),
                        state: state,
                        current: current,
                        text: text_,
                        hash: hash
                    });
                }
            } catch (e: any) {
                debugLog(`Failed to parse range ${i}: ${e?.message || e}`, 'warn');
                // Continue with other ranges
            }
        }
        return ranges;
    } catch (error: any) {
        debugLog(`parseResultFromPythonAndGetRange failed: ${error?.message || error}`, 'error');
        return null;
    }
};

const getCurrentRangesFromPython = async (
    editor: TextEditor,
    output: OutputChannel | null,
    globals: Map<string, string>,
    {
        rebuild,
        current_line = undefined,
        upstream = true,
        downstream = true,
        stale_only = false,
        to_launch_compute = false
    }: {
        rebuild: boolean;
        current_line?: number | undefined | null;
        upstream?: boolean;
        downstream?: boolean;
        stale_only?: boolean;
        to_launch_compute?: boolean;
    },
): Promise<AnnotatedRange[] | undefined> => {
    try {
        if (current_line === null) {
        }
        let linen = current_line === undefined ? getEditorCurrentLineNum(editor) : current_line;
        let text: string | null = rebuild ? getEditorAllText(editor).text : null;
        if (!text && !linen) return;
        text = text ? formatTextAsPythonString(text) : null;
        let command = getCommandToGetAllRanges(text, linen, upstream, downstream, stale_only, to_launch_compute);
        const result = await safeExecuteCodeInKernel(command, editor, output, globals);
        if (result === undefined || result == '[]' || isExecutionError(result)) return;
        if (to_launch_compute) {
        }
        const ranges_out = await parseResultFromPythonAndGetRange(result);
        if (!ranges_out) return;
        return ranges_out;
    } catch (error: any) {
        debugLog(`getCurrentRangesFromPython failed: ${error?.message || error}`, 'warn');
        return undefined;
    }
};

const getTextInRanges = (ranges: AnnotatedRange[]): string[] => {
    let text: string[] = [];
    let editor = window.activeTextEditor;
    if (!editor) return text;
    for (let i = 0; i < ranges.length; i++) {
        let range = ranges[i].range;
        let textInRange = editor.document.getText(range);
        text.push(textInRange);
    }
    return text;
};


const getEditorCurrentText = (editor: TextEditor): { currentQuery: string; currentRange: Range | null } => {
    // Currently not used..

    if (!editor || !editor.document || editor.document.uri.scheme === 'output') {
        return {
            currentQuery: '',
            currentRange: null
        };
    }
    if (!editor.selection.isEmpty) {
        return {
            currentQuery: editor.document.getText(editor.selection),
            currentRange: editor.selection
        };
    }
    const currentLine = editor.document
        .getText(new Range(Math.max(0, editor.selection.active.line - 4), 0, editor.selection.active.line + 1, 0))
        .replace(/[\n\r\s]/g, '');
    if (currentLine.length === 0)
        return {
            currentQuery: '',
            currentRange: editor.selection
        };
    const text = editor.document.getText();
    const currentOffset = editor.document.offsetAt(editor.selection.active);
    const prefix = text.slice(0, currentOffset + 1);
    const allQueries = text; // parse(text);
    const prefixQueries = prefix; // parse(prefix);
    const currentQuery = allQueries[prefixQueries.length - 1];
    const startIndex = prefix.lastIndexOf(prefixQueries[prefixQueries.length - 1]);
    const startPos = editor.document.positionAt(startIndex);
    const endPos = editor.document.positionAt(startIndex + currentQuery.length);
    return {
        currentQuery,
        currentRange: new Range(startPos, endPos)
    };
};



////////////////////////////////////////////////////////////////////////////////////////////////////
// HIGHLIGHTING UTILS
////////////////////////////////////////////////////////////////////////////////////////////////////



const HighlightSynced = window.createTextEditorDecorationType({
    backgroundColor: { id: `reactivejupyter.syncedCell` },
    borderColor: { id: `reactivejupyter.syncedCell` },
    borderWidth: '0px',
    borderStyle: 'solid'
});
const HighlightSyncedCurrent = window.createTextEditorDecorationType({
    backgroundColor: { id: `reactivejupyter.syncedCurrentCell` },
    borderColor: { id: `reactivejupyter.syncedCurrentCell` },
    borderWidth: '0px',
    borderStyle: 'solid'
});
const HighlightOutdated = window.createTextEditorDecorationType({
    backgroundColor: { id: `reactivejupyter.outdatedCell` },
    borderColor: { id: `reactivejupyter.outdatedCell` },
    borderWidth: '0px',
    borderStyle: 'solid'
});
const HighlightOutdatedCurrent = window.createTextEditorDecorationType({
    backgroundColor: { id: `reactivejupyter.outdatedCurrentCell` },
    borderColor: { id: `reactivejupyter.outdatedCurrentCell` },
    borderWidth: '0px',
    borderStyle: 'solid'
});

let updateDecorations = async (editor: TextEditor, ranges_out: AnnotatedRange[]) => {
    if (
        !editor ||
        !editor.document ||
        editor.document.uri.scheme === 'output'
    ) {
        return;
    }
    try {
        let sync_ranges = ranges_out.filter((r) => r.state == 'synced' && !r.current).map((r) => r.range);
        editor.setDecorations(HighlightSynced, sync_ranges);
        let sync_curr_ranges = ranges_out.filter((r) => r.state == 'synced' && r.current).map((r) => r.range);
        editor.setDecorations(HighlightSyncedCurrent, sync_curr_ranges);
        let out_ranges = ranges_out.filter((r) => (r.state == 'outdated' || r.state == 'dependsonotherstalecode') && !r.current).map((r) => r.range);
        editor.setDecorations(HighlightOutdated, out_ranges);
        let highlightsBrightCond = (r: AnnotatedRange) => (((r.state == 'outdated' || r.state == 'dependsonotherstalecode') && r.current) || ((r.state == 'syntaxerror' && !r.current)))
        let out_curr_ranges = ranges_out.filter(highlightsBrightCond).map((r) => r.range);
        editor.setDecorations(HighlightOutdatedCurrent, out_curr_ranges);
        
        // If there is a range with a SyntaxError, show a Window message saying that:
        let syntax_error_ranges = ranges_out.filter((r) => (r.state == 'syntaxerror' && !r.current));
        if (syntax_error_ranges.length > 0) {
            window.showErrorMessage('Reactive Jupyter: Syntax Error at line ' + (syntax_error_ranges[0].range.start.line + 1).toString()); 
        }
    } catch (error) {
    }
};




////////////////////////////////////////////////////////////////////////////////////////////////////
// CODELENS
////////////////////////////////////////////////////////////////////////////////////////////////////

export class CellCodelensProvider implements vscode.CodeLensProvider {
    private codeLenses: vscode.CodeLens[] = [];
    private range: Range | undefined;
    private _onDidChangeCodeLenses: vscode.EventEmitter<void> = new vscode.EventEmitter<void>();
    public readonly onDidChangeCodeLenses: vscode.Event<void> = this._onDidChangeCodeLenses.event;

    change_range(new_range: Range | undefined) {
        if (new_range && new_range != this.range) {
            this.range = new_range;
            this._onDidChangeCodeLenses.fire();
        } else if (!new_range && this.range) {
            this.range = undefined;
            this._onDidChangeCodeLenses.fire();
        }
    }
    constructor() {
        vscode.workspace.onDidChangeConfiguration((_) => {
            this._onDidChangeCodeLenses.fire();
        });
        // Context.subscriptions.push(this._onDidChangeCodeLenses);
    }

    public provideCodeLenses(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): vscode.CodeLens[] | PromiseLike<vscode.CodeLens[]> {
        let editor = vscode.window.activeTextEditor;
        // Get the "reactiveJupyter.showCodeLenses" extension setting is True:
        let showCodeLenses = vscode.workspace.getConfiguration('reactiveJupyter').get<boolean>('showCodeLenses');
        if (editor && this.range && editor.document.uri == document.uri && showCodeLenses) {
            this.codeLenses = [
                new vscode.CodeLens(new vscode.Range(this.range.start.line, 0, this.range.end.line, 0), {
                    title: 'sync upstream',
                    tooltip: 'Run all outdated code upstream, including this cell',
                    command: 'reactive-jupyter.sync-upstream',
                    arguments: [this.range]
                }),
                new vscode.CodeLens(new vscode.Range(this.range.start.line, 0, this.range.end.line, 0), {
                    title: 'sync downstream',
                    tooltip: 'Run all outdated code downstream, including this cell',
                    command: 'reactive-jupyter.sync-downstream',
                    arguments: [this.range]
                }),
                new vscode.CodeLens(new vscode.Range(this.range.start.line, 0, this.range.end.line, 0), {
                    title: 'sync current',
                    tooltip: 'Run current block of code, if all its upstream code is up to date',
                    command: 'reactive-jupyter.sync-current',
                    arguments: [this.range]
                }),
                new vscode.CodeLens(new vscode.Range(this.range.start.line, 0, this.range.end.line, 0), {
                    title: 'sync upstream and downstream',
                    tooltip: 'Run all outdated code upstream and downstream, including this cell',
                    command: 'reactive-jupyter.sync-upstream-and-downstream',
                    arguments: [this.range]
                })
            ];
            return this.codeLenses;
        }
        return [];
    }
}

export class InitialCodelensProvider implements vscode.CodeLensProvider {
    public provideCodeLenses(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): vscode.CodeLens[] | PromiseLike<vscode.CodeLens[]> {
        let editor = vscode.window.activeTextEditor;
        if (editor && editor.document.uri == document.uri) {
            let codeLenses = [
                new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), {
                    title: '$(debug-start) Initialize Reactive Jupyter',
                    tooltip: 'Initialize Reactive Jupyter on the current file',
                    command: 'reactive-jupyter.initialize-reactive-python-extension'
                    // arguments: [this.range] // Wanna pass the editor uri?
                }),
                new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), {
                    title: 'Sync all Stale code',
                    tooltip: 'Sync all Stale code in current file',
                    command: 'reactive-jupyter.sync-all'
                    // arguments: [this.range] // Wanna pass the editor uri?
                }),
                new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), {
                    title: '$(refresh) Force Reset',
                    tooltip: 'Force reset Reactive Jupyter state (use if stuck)',
                    command: 'reactive-jupyter.force-reset'
                })
            ];
            return codeLenses;
        }
        return [];
    }
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// COMMANDS
////////////////////////////////////////////////////////////////////////////////////////////////////



function createPreparePythonEnvForReactivePythonAction(globals: Map<string, string>, output: OutputChannel) {
    async function preparePythonEnvForReactivePythonAction() {
        try {
            let editor = window.activeTextEditor;
            if (!editor) { return; }
            preparePythonEnvForReactivePython(editor, globals, output);
        } catch (error: any) {
            debugLog(`preparePythonEnvForReactivePythonAction failed: ${error?.message || error}`, 'error', true);
        }
    }
    return preparePythonEnvForReactivePythonAction;
}

function createComputeAction(config: {
    rebuild: boolean;
    current_line?: number | undefined | null;
    upstream: boolean;
    downstream: boolean;
    stale_only: boolean;
    to_launch_compute: boolean;
}, globals: Map<string, string>, output: OutputChannel) {
    async function computeAction() {
        try {
            let editor = window.activeTextEditor;
            if (!editor) return;
            if (getKernelState(globals, editor) !== KernelState.extension_available) { return; }
            const current_ranges = await getCurrentRangesFromPython(editor, output, globals, config,);
            await queueComputation(current_ranges, editor, globals, output);
        } catch (error: any) {
            debugLog(`computeAction failed: ${error?.message || error}`, 'error', true);
            // Try to recover by resetting state
            const editor = window.activeTextEditor;
            if (editor) {
                try {
                    await safeExecuteCodeInKernel(getUnlockCommand(), editor, output, globals);
                } catch { /* ignore */ }
            }
        }
    }
    return computeAction;
}

function createPrepareEnvironementAndComputeAction(config: {
    rebuild: boolean;
    current_line?: number | undefined | null;
    upstream: boolean;
    downstream: boolean;
    stale_only: boolean;
    to_launch_compute: boolean;
}, globals: Map<string, string>, output: OutputChannel) {
    async function prepareEnvironementAndComputeAction() {
        try {
            let editor = window.activeTextEditor;
            if (!editor) return;
            await preparePythonEnvForReactivePython(editor, globals, output);
            if (getKernelState(globals, editor) !== KernelState.extension_available) { return; }
            const current_ranges = await getCurrentRangesFromPython(editor, output, globals, config,);
            await queueComputation(current_ranges, editor, globals, output);
        } catch (error: any) {
            debugLog(`prepareEnvironementAndComputeAction failed: ${error?.message || error}`, 'error', true);
            // Try to recover
            const editor = window.activeTextEditor;
            if (editor) {
                try {
                    await safeExecuteCodeInKernel(getUnlockCommand(), editor, output, globals);
                } catch { /* ignore */ }
            }
        }
    }
    return prepareEnvironementAndComputeAction;
}









// TODO- Right now, this is not much of a state machine. But there is the infrastructure in place to complicate it...
enum EditingState {
    rebuilt = 'rebuilt',
    rebuildNeeded = 'rebuildNeeded',
    rebuildStarted = 'rebuildStarted',
    rebuildingStaleStuff = 'rebuildingStaleStuff',
}
const highlightIinitialStates = [EditingState.rebuilt];
const highlightStateTransitions: Map<EditingState, EditingState[]> = new Map([
    [EditingState.rebuilt, [EditingState.rebuildNeeded]],
    [EditingState.rebuildNeeded, [EditingState.rebuildNeeded, EditingState.rebuildStarted]],
    [EditingState.rebuildStarted, [EditingState.rebuildNeeded, EditingState.rebuilt]],
    [EditingState.rebuildingStaleStuff, [EditingState.rebuildingStaleStuff, EditingState.rebuildStarted]],
]);

function updateEditingState(globals: Map<string, string>, editor: TextEditor, newState_: string, timestamp: EpochTimeStamp = Date.now()) {
    const key = editorRebuildKey(editor.document.uri.toString());
    updateState<EditingState>(globals, key, newState_, highlightStateTransitions, highlightIinitialStates);
    globals.set(editorLastEventTimestampKey(editor.document.uri.toString()), timestamp.toString());
}

function getEditingState(globals: Map<string, string>, editor: TextEditor): [EditingState | boolean, EpochTimeStamp | undefined] {
    let timestamp = globals.get(editorLastEventTimestampKey(editor.document.uri.toString()));
    return [
        getState<EditingState>(globals, editorRebuildKey(editor.document.uri.toString())),
        timestamp ? parseInt(timestamp) : undefined
    ];
}

function getOnDidChangeTextEditorSelectionAction(globals: Map<string, string>, output: OutputChannel, codelensProvider: CellCodelensProvider) {
    return async (event: vscode.TextEditorSelectionChangeEvent): Promise<void> => {
        try {
            let editor = window.activeTextEditor;
            if (!(event.textEditor && editor && event.textEditor.document === editor.document && editor.selection.isEmpty)) {
                updateDecorations(event.textEditor, []);
                codelensProvider.change_range(undefined);
            } else if (getKernelState(globals, editor) == KernelState.extension_available && getEditingState(globals, editor)[0] == EditingState.rebuilt) {
                let current_ranges = await getCurrentRangesFromPython(editor, output, globals, { rebuild: false });
                updateDecorations(editor, current_ranges ? current_ranges : []);
                let codelense_range = current_ranges ? current_ranges.filter((r) => (r.current && r.state != 'syntaxerror')).map((r) => r.range) : [];
                codelensProvider.change_range(codelense_range.length > 0 ? codelense_range[0] : undefined);
            }
        } catch (error: any) {
            debugLog(`onDidChangeTextEditorSelection error: ${error?.message || error}`, 'warn');
            // Don't rethrow - this is an event handler
        }
    };
}

const CONTENTCHANGE_LENGHT_ABOVE_WHICH_ALWAYS_TRIGGER_REBUILD = 15;


function getOnDidChangeTextDocumentAction(globals: Map<string, string>, output: OutputChannel): (e: vscode.TextDocumentChangeEvent) => any {
    return async (event: vscode.TextDocumentChangeEvent) => {
        try {
            let editor = window.activeTextEditor;
            if (
                    editor 
                    && event.document === editor.document 
                    && event.contentChanges.length > 0
                    && getEditingState(globals, editor)[0] != false) {
                // Set the current timstamp:
                const timestamp = Date.now();
                // Set the state to rebuildNeeded:
                updateEditingState(globals, editor, EditingState.rebuildNeeded, timestamp);
                // Check if ANY of the contentChanges's has length > CONTENTCHANGE_LENGHT_ABOVE_WHICH_ALWAYS_TRIGGER_REBUILD:
                let should_rebuild_immediatly = event.contentChanges.some((change: vscode.TextDocumentContentChangeEvent) => change.text.length > CONTENTCHANGE_LENGHT_ABOVE_WHICH_ALWAYS_TRIGGER_REBUILD);
                await new Promise((resolve) => setTimeout(resolve, should_rebuild_immediatly ? 50 : 1000));
                // Get latest timestamp:
                let [state, last_timestamp] = getEditingState(globals, editor);
                // If last timestamp is greater than the one we set,OR state != rebuildNeeded, then it means that another event has been triggered, so we don't need to do anything:
                if ((last_timestamp && last_timestamp > timestamp) || state != EditingState.rebuildNeeded) { return; }
                // Otherwise, we should rebuild:
                updateEditingState(globals, editor, EditingState.rebuildStarted, );
                const current_ranges = await getCurrentRangesFromPython(editor, output, globals, { rebuild: true });
                console.log('Are you not triggering everything here??: ', current_ranges);
                updateDecorations(editor, current_ranges ? current_ranges : []);
                // IF the state is STILL rebuildStarted, then we update the state to rebuilt:
                let [state_, last_timestamp_] = getEditingState(globals, editor);
                if (state_ == EditingState.rebuildStarted) { updateEditingState(globals, editor, EditingState.rebuilt, ); }
            };
        } catch (error: any) {
            debugLog(`onDidChangeTextDocument error: ${error?.message || error}`, 'warn');
            // Don't rethrow - this is an event handler
        }
    }
}
    

function getOnDidChangeActiveTextEditorAction(globals: Map<string, string>, output: OutputChannel): (e: TextEditor | undefined) => any {
    return async (editor: TextEditor | undefined) => {
        try {
            if (editor) {
                if (getKernelState(globals, editor) !== KernelState.extension_available) { return; }
                const current_ranges = await getCurrentRangesFromPython(editor, output, globals, { rebuild: true });
                if (!current_ranges) return;
                await updateDecorations(editor, current_ranges);
            }
        } catch (error: any) {
            debugLog(`onDidChangeActiveTextEditor error: ${error?.message || error}`, 'warn');
            // Don't rethrow - this is an event handler
        }
    };
}




////////////////////////////////////////////////////////////////////////////////////////////////////
// ACTIVATION
////////////////////////////////////////////////////////////////////////////////////////////////////   


const queue: any[] = [];
export const setCurrentContext = (ctx: ExtensionContext) => {
    currentContext = ctx as typeof currentContext;
    queue.forEach((cb) => cb());
};
const onRegister = (cb: () => void) => queue.push(cb);
let currentContext: ExtensionContext & { set: typeof setCurrentContext; onRegister: typeof onRegister } = {} as any;
const handler = {
    get(_: never, prop: string) {
        if (prop === 'set') return setCurrentContext;
        if (prop === 'onRegister') return onRegister;
        return currentContext[prop as keyof typeof currentContext];
    },
    set() {
        throw new Error('Cannot set values to extension context directly!');
    }
};
const Context = new Proxy<typeof currentContext>(currentContext, handler);

export const globals: Map<string, string> = new Map();

export function activate(context: ExtensionContext) {
    const jupyterExt = extensions.getExtension<Jupyter>('ms-toolsai.jupyter');
    if (!jupyterExt) {
        throw new Error('The Jupyter Extension not installed. Please install it and restart VSCode.');
    }
    if (!jupyterExt.isActive) {
        jupyterExt.activate();
    }
    const output = window.createOutputChannel('Jupyter Kernel Execution');
    context.subscriptions.push(output);

    currentContext = context as typeof currentContext;

    defineAllCommands(context, output, globals);

}




async function defineAllCommands(context: ExtensionContext, output: OutputChannel, globals: Map<string, string>
) {
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'reactive-jupyter.initialize-reactive-python-extension',
            createPreparePythonEnvForReactivePythonAction(globals, output)
        )
    );
    context.subscriptions.push( vscode.commands.registerCommand('reactive-jupyter.sync-downstream', createComputeAction({ rebuild: true, upstream: false, downstream: true, stale_only: true, to_launch_compute: true }, globals, output)) );
    context.subscriptions.push( vscode.commands.registerCommand('reactive-jupyter.sync-upstream', createComputeAction({ rebuild: true, upstream: true, downstream: false, stale_only: true, to_launch_compute: true }, globals, output)) );
    context.subscriptions.push( vscode.commands.registerCommand('reactive-jupyter.sync-upstream-and-downstream', createComputeAction({ rebuild: true, upstream: true, downstream: true, stale_only: true, to_launch_compute: true }, globals, output)) );
    context.subscriptions.push( vscode.commands.registerCommand('reactive-jupyter.sync-current', createComputeAction({ rebuild: true, upstream: false, downstream: false, stale_only: false, to_launch_compute: true }, globals, output)) );
    context.subscriptions.push( vscode.commands.registerCommand('reactive-jupyter.sync-all', createComputeAction({ rebuild: true, current_line: null, upstream: true, downstream: true, stale_only: true, to_launch_compute: true }, globals, output)) );

    context.subscriptions.push( vscode.commands.registerCommand('reactive-jupyter.initialize-and-sync-downstream', createPrepareEnvironementAndComputeAction({ rebuild: true, upstream: false, downstream: true, stale_only: true, to_launch_compute: true }, globals, output)) );
    context.subscriptions.push( vscode.commands.registerCommand('reactive-jupyter.initialize-and-sync-upstream', createPrepareEnvironementAndComputeAction({ rebuild: true, upstream: true, downstream: false, stale_only: true, to_launch_compute: true }, globals, output)) );
    context.subscriptions.push( vscode.commands.registerCommand('reactive-jupyter.initialize-and-sync-upstream-and-downstream', createPrepareEnvironementAndComputeAction({ rebuild: true, upstream: true, downstream: true, stale_only: true, to_launch_compute: true }, globals, output)) );
    context.subscriptions.push( vscode.commands.registerCommand('reactive-jupyter.initialize-and-sync-current', createPrepareEnvironementAndComputeAction({ rebuild: true, upstream: false, downstream: false, stale_only: false, to_launch_compute: true }, globals, output)) );
    context.subscriptions.push( vscode.commands.registerCommand('reactive-jupyter.initialize-and-sync-all', createPrepareEnvironementAndComputeAction({ rebuild: true, current_line: null, upstream: true, downstream: true, stale_only: true, to_launch_compute: true }, globals, output)) );

    ///////// Force Reset Commands: ///////////////////////

    // Force reset current editor state - allows re-initialization
    context.subscriptions.push(
        vscode.commands.registerCommand('reactive-jupyter.force-reset', async () => {
            const editor = window.activeTextEditor;
            if (!editor) {
                window.showWarningMessage('Reactive Jupyter: No active editor to reset');
                return;
            }
            forceResetEditorState(globals, editor);
            updateDecorations(editor, []);
            window.showInformationMessage('Reactive Jupyter: State reset for current editor. You can now Initialize Reactive Jupyter again.');
        })
    );

    // Force reset ALL state - nuclear option
    context.subscriptions.push(
        vscode.commands.registerCommand('reactive-jupyter.force-reset-all', async () => {
            forceResetAllState(globals);
            // Clear decorations on all visible editors
            for (const editor of window.visibleTextEditors) {
                try {
                    updateDecorations(editor, []);
                } catch { /* ignore */ }
            }
            window.showInformationMessage('Reactive Jupyter: All state has been reset. You can now Initialize Reactive Jupyter again.');
        })
    );

    // Toggle debug mode
    context.subscriptions.push(
        vscode.commands.registerCommand('reactive-jupyter.toggle-debug-mode', async () => {
            DEBUG_MODE = !DEBUG_MODE;
            window.showInformationMessage(`Reactive Jupyter: Debug mode is now ${DEBUG_MODE ? 'ON' : 'OFF'}`);
        })
    );

    // Show current state (for debugging)
    context.subscriptions.push(
        vscode.commands.registerCommand('reactive-jupyter.show-state', async () => {
            const editor = window.activeTextEditor;
            if (!editor) {
                window.showInformationMessage('Reactive Jupyter: No active editor');
                return;
            }
            const editorUri = editor.document.uri.toString();
            const kernelState = getKernelState(globals, editor);
            const [editingState, timestamp] = getEditingState(globals, editor);
            const iwKey = globals.get(editorToIWKey(editorUri));
            
            let stateInfo = `Kernel State: ${kernelState}\nEditing State: ${editingState}\nTimestamp: ${timestamp}\nIW Key: ${iwKey || 'not set'}`;
            
            // Show in output channel for easier copying
            output.appendLine('=== Reactive Jupyter State Debug ===');
            output.appendLine(`Editor URI: ${editorUri}`);
            output.appendLine(stateInfo);
            output.appendLine('=== End State Debug ===');
            output.show();
            
            window.showInformationMessage(`Reactive Jupyter State:\nKernel: ${kernelState}, Editing: ${editingState}`);
        })
    );

    ///////// Codelens: ///////////////////////

    const codelensProvider = new CellCodelensProvider();
    languages.registerCodeLensProvider('python', codelensProvider);
    const initializingProvider = new InitialCodelensProvider();
    languages.registerCodeLensProvider('python', initializingProvider);

    ///////// Document Highlights: ///////////////////////

    workspace.onDidChangeTextDocument( getOnDidChangeTextDocumentAction(globals, output), null, Context.subscriptions );
    window.onDidChangeActiveTextEditor( getOnDidChangeActiveTextEditorAction(globals, output), null, Context.subscriptions );
    window.onDidChangeTextEditorSelection( getOnDidChangeTextEditorSelectionAction(globals, output, codelensProvider), null, Context.subscriptions );

    ///////// Wrap In Block: ///////////////////////

    context.subscriptions.push( vscode.commands.registerCommand( "reactive-jupyter.wrap-in-reactive-block", wrapInBlock));

}




const wrapInBlock = async () => {
        // Get the active text editor
        let editor = window.activeTextEditor;
        if (editor) {
            // Collect the selected lines' text
            let selectedText = "";
            for (let lineNum = editor.selection.start.line; lineNum <= editor.selection.end.line; lineNum++) {
                selectedText += editor.document.lineAt(lineNum).text + "\n";
            }

            // Wrap the selected text in reactive block markers
            let wrappedText = "# % [\n" + selectedText + "# % ]\n";

            // Create a range covering the selected lines
            let replaceRange = new Range(
                new Position(editor.selection.start.line, 0),
                new Position(editor.selection.end.line + 1, 0)
            );

            // Replace the selected lines with the wrapped text
            editor.edit((editBuilder: vscode.TextEditorEdit) => {
                editBuilder.replace(replaceRange, wrappedText);
            });

            // Adjust the selection to the newly wrapped block
            let newStart = new Position(
                editor.selection.start.line + 1,
                editor.selection.start.character
            );
            let newEnd = new Position(
                editor.selection.end.line + 1,
                editor.selection.end.character
            );
            editor.selection = new Selection(newStart, newEnd);
        }
    };

////////////////////////////////////////////////////////////////////////////////////////////////////
// NOTES
////////////////////////////////////////////////////////////////////////////////////////////////////

///////// Do highlighting for the first time right now: ///////////////////////
// if (window.activeTextEditor) {
//     updateDecorations(window.activeTextEditor, kernel, output);
// } else {
//     vscode.window.showInformationMessage(
//         'No active text editor on Activation Time. Try to retrigger the highlighter somehow.'
//     );
// }

///////// CodeLenses: ///////////////////////
// Add a codelens above the line where the cursor is, that launches the "reactive-jupyter.test-command" command:
// const codelensProvider = new MyCodeLensProvider();
// const disposable = languages.registerCodeLensProvider({ language: 'python' }, codelensProvider);
// context.subscriptions.push(disposable);
// HINT: ONE version of this is /Users/michele.tasca/Documents/vscode-extensions/vscode-reactive-jupyter/src/interactive-window/editor-integration/codelensprovider.ts !!


// USEFUL THINGS TO KNOW ABOUT: JUPYTER COMMANDS:                           
// "jupyter.execSelectionInteractive"  
// "jupyter.createnewinteractive",  // Create Interactive Window
// "jupyter.deleteCells"  // Delete Selected Cells
// "jupyter.restartkernel"  // Restart Kernel
// "jupyter.removeallcells"  // Delete All Notebook Editor Cells
// "jupyter.interactive.clearAllCells"  // Clear All
// "jupyter.selectDependentCells"  //  :O
// interactive.open - Open interactive window and return notebook editor and input URI

// IMPORTANT IDEA: controller == kernel !!


// USEFUL THINGS TO KNOW ABOUT: VSCODE NOTEBOOKS:                           
// NotebookCellData contains the 
//  - outputs: NotebookCellOutput[] 
//  - AND value: string 
//  - AND the executionSummary: NotebookCellExecutionSummary
//  NotebookCellExecutionSummary contains the
//  - success: boolean
// BUT ALSO, NotebookCell contains the
//  - readonly document: TextDocument;
//  - readonly outputs: readonly NotebookCellOutput[];
//  - readonly executionSummary: NotebookCellExecutionSummary | undefined;
// Apparently, NotebookCellData is the INPUT TO A NotebookEdit, which is the thing that EXTENSION sends vscode to modify a NotebookDocument of NotebookCell's ?
// NotebookCellOutput is a list of NotebookCellOutputItem's which have a data: Uint8Array and a mime: string


