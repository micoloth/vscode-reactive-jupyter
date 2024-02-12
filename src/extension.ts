import {
	CancellationTokenSource,
	Disposable,
	ExtensionContext,
	NotebookCellOutputItem,
	OutputChannel,
	QuickPickItem,
	commands,
	extensions,
	languages,
    window,
    Range,
    workspace,
    TextEditor,
    NotebookDocument,
    Position,
    Selection,
    CodeLensProvider,
    Uri,
    NotebookEditor,
    ViewColumn,
    NotebookCell
} from 'vscode';
import { Jupyter, Kernel, JupyterServerCommandProvider } from '@vscode/jupyter-extension';
import path = require('path');
import { TextDecoder } from 'util';

import {scriptCode} from './reactive_python_engine';
import * as interactiveWindow from './interactiveWindowExperiments';
import { CellOutputDisplayIdTracker } from './cellExecutionMessageHandler';

import * as vscode from 'vscode';
import { get } from 'http';

const ErrorMimeType = NotebookCellOutputItem.error(new Error('')).mime;
const StdOutMimeType = NotebookCellOutputItem.stdout('').mime;
const StdErrMimeType = NotebookCellOutputItem.stderr('').mime;
const MarkdownMimeType = 'text/markdown';
const HtmlMimeType = 'text/html';
const textDecoder = new TextDecoder();

type AnnotatedRange = {
    range: Range;
    state: string; // Remember this exists too: 'synced' | 'outdated';
    current: boolean;
    text?: string;
    hash?: string; // Hash is used so that when you send a node Back to Python, you can check if it actually him or not
    has_children?: boolean;
};




////////////////////////////////////////////////////////////////////////////////////////////////////
// UTILS
////////////////////////////////////////////////////////////////////////////////////////////////////


async function* executeCodeStreamInKernel(code: string, kernel: Kernel, output_channel: OutputChannel | null) {
    /*
    Currently, it ALWAYS logs a line (for the user), and it returns the result if it is NOT AN ERROR, else Undefined. 
    If you need Debugging traces, use console.log(). (currently it logs errors)
    */

    if (output_channel) {
        output_channel.show(true);
    }
    // 	// output_channel.appendLine(`>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>`);
    // 	// output_channel.appendLine(`Executing code against kernel ${code}`);
    const tokenSource = new CancellationTokenSource();
    try {
        for await (const output of kernel.executeCode(code, tokenSource.token)) {
            for (const outputItem of output.items) {
                const decoded = textDecoder.decode(outputItem.data);
                if (outputItem.mime === ErrorMimeType) {
                    const error = JSON.parse(decoded) as Error;
                    if (output_channel) {
                        output_channel.appendLine(`Error executing code ${error.name}: ${error.message},/n ${error.stack}`);
                    }
                    console.log(`Error executing code ${error.name}: ${error.message},/n ${error.stack}`);
                    yield undefined;
                } else {
                    if (output_channel) {
                        output_channel.appendLine( `${outputItem.mime} Output: ${decoded}` );
                    }   
                    yield decoded;
                    if (output_channel) {
                        // Remove one line from the output channel:
                        // output_channel.appendLine('Code execution completed');
                    }
                }
            }
        }
    // 		// output_channel.appendLine('Code execution completed');
    // 		// output_channel.appendLine(`<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`);
    // 	} catch (ex){
    // 		// output_channel.appendLine(`Code execution failed with an error '${ex}'`);
    // 		// output_channel.appendLine(`<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`);
    }
    finally {
        tokenSource.dispose();
    }
}

async function executeCodeInKernel(code: string, kernel: Kernel, output_channel: OutputChannel | null) {
    let result = '';
    for await (const output of executeCodeStreamInKernel(code, kernel, output_channel)) {
        if (output === undefined) {
            return undefined;
        }
        result += output;
    }
    return result;
}


async function executeCodeInInteractiveWindow(
    text: string,
    notebook: NotebookDocument,
    textEditor: TextEditor,
    output: OutputChannel | null,
) {
    let cell: NotebookCell = await interactiveWindow.addNotebookCell(
        text, 
        textEditor.document.uri,
        textEditor.selection.start.line,
        notebook
        )
    for (let i = 0; i < 20; i++) {
        await new Promise((resolve) => setTimeout(resolve, 500));
        let newCell = await getUpdatedCell(cell);
        console.log('INDEX: >> ', newCell.index);
        console.log('TEXT: >> ', newCell.document.getText());
        console.log('OUTPUT: >> ', newCell.outputs);
        console.log('(Btw, the cells are: ', cell.notebook.getCells().map((c) => ([c.index, c.document.getText()])), ')');
    }

    let newCell = await getUpdatedCell(cell);
    return newCell.outputs; // TODO: This is all wrong...
    // In particular, AT LEAT you should check if it is an Error and in that case, return undefined!!

    // OTHER THINGS I TRIED:
    // let res = await vscode.commands.executeCommand('jupyter.execSelectionInteractive', text);
    // let res = await getIWAndRunText(serviceManager, activeTextEditor, text);
    // let res = await executeCodeInKernel(text, kernel, output);

}

// USEFUL:                           
// "jupyter.execSelectionInteractive"  
// "jupyter.createnewinteractive",  // Create Interactive Window
// "jupyter.deleteCells"  // Delete Selected Cells
// "jupyter.restartkernel"  // Restart Kernel
// "jupyter.removeallcells"  // Delete All Notebook Editor Cells
// "jupyter.interactive.clearAllCells"  // Clear All
// "jupyter.selectDependentCells"  //  :O

// interactive.open - Open interactive window and return notebook editor and input URI

// showOptions - Show Options
// resource - Interactive resource Uri
// controllerId - Notebook controller Id
// title - Interactive editor title
// (returns) - Notebook and input URI


// IMPORTANT IDEA: controller == kernel !!
// TODO: Look for NotebookController

// let resDel = await vscode.commands.executeCommand('jupyter.interactive.clearAllCells');




async function safeExecuteCodeInInteractiveWindow(
    command: string, 
    editor: TextEditor, 
    output: OutputChannel | null, 
    globalState: Map<string, string>, 
    expected_initial_state: State = State.extension_available, 
    return_to_initial_state: boolean = true
) {
displayInitializationMessageIfNeeded(globalState, editor);
if (getState(globalState, editor) !== expected_initial_state) { return; }
if (!checkSettings(globalState, editor)) { return; }

let notebookAndKernel = await getNotebookAndKernel(globalState, editor, true);
if (!notebookAndKernel) { 
    window.showErrorMessage("Lost Connection to this editor's Notebook. Please initialize the extension with the command: 'Initialize Reactive Python' or the CodeLens at the top");
    updateState(globalState, editor, State.initializable_messaged);
    return;
}
let [notebook, kernel] = notebookAndKernel;
updateState(globalState, editor, State.explit_execution_started);
const result = await executeCodeInInteractiveWindow(command, notebook, editor, output);
if (return_to_initial_state) { updateState(globalState, editor, expected_initial_state); }
return result;
}

async function safeExecuteCodeInKernel(
    command: string, 
    editor: TextEditor, 
    output: OutputChannel | null, 
    globalState: Map<string, string>, 
    expected_initial_state: State = State.extension_available, 
    return_to_initial_state: boolean = true
) {
displayInitializationMessageIfNeeded(globalState, editor);
if (getState(globalState, editor) !== expected_initial_state) { return; }
if (!checkSettings(globalState, editor)) { return; }

let notebookAndKernel = await getNotebookAndKernel(globalState, editor, true);
if (!notebookAndKernel) { 
    window.showErrorMessage("Lost Connection to this editor's Kernel. Please initialize the extension with the command: 'Initialize Reactive Python' or the CodeLens at the top");
    updateState(globalState, editor, State.initializable_messaged);
    return;
}
let [notebook, kernel] = notebookAndKernel;
updateState(globalState, editor, State.implicit_execution_started);
const result = await executeCodeInKernel(command, kernel, output);
if (return_to_initial_state) { updateState(globalState, editor, expected_initial_state); }
return result;
}
async function safeExecuteCodeInKernelForInitialization(
command: string, 
editor: TextEditor, 
output: OutputChannel | null, 
globalState: Map<string, string>
): Promise<boolean> {
// It's SLIGHTLY different from the above one... 
displayInitializationMessageIfNeeded(globalState, editor);
if (getState(globalState, editor) !== State.kernel_available) { return false; }
if (!checkSettings(globalState, editor)) { return false; }

let notebookAndKernel = await getNotebookAndKernel(globalState, editor, true);
if (!notebookAndKernel) { 
    window.showErrorMessage("Kernel Initialization succeeded, but we lost it already... Please try again.");
    updateState(globalState, editor, State.initializable_messaged);
    return false;
}
let [notebook, kernel] = notebookAndKernel;
updateState(globalState, editor, State.instantialization_started);
const result = await executeCodeInKernel(command, kernel, output);
if (result) { 
    updateState(globalState, editor, State.extension_available);
    window.showInformationMessage('Reactive Python is ready to use');
    return true;
} else {
    updateState(globalState, editor, State.kernel_available);
    window.showErrorMessage('The Reactive Python code could not be executed in the Python Kernel. This is bad...');
    return false;
}
}

async function getUpdatedCell(cell: NotebookCell) {
let notebook = cell.notebook;
let index = cell.index;

let newCell = await notebook.cellAt(index);
return newCell;
}

async function queueComputation(
    current_ranges: AnnotatedRange[] | undefined,
    activeTextEditor: TextEditor,
    globalState: Map<string, string>,
    output: OutputChannel,
) {
    // console.log('>> CURRENT RANGES::::: ' + current_ranges);
    if (current_ranges) {
        // let resDel = await vscode.commands.executeCommand('jupyter.interactive.clearAllCells');
        // console.log('---- >>>> RES DEL: ', resDel);
        for (let range of current_ranges) {
            if (!range.text) break;
            // console.log('>> TEXT: ', range.text);
            
            let res = await safeExecuteCodeInInteractiveWindow(range.text, activeTextEditor, output, globalState);
            if (!res) { break; }
            
            // console.log('>> updateRange_command: ', updateRange_command);
            const update_result = await safeExecuteCodeInKernel(getSyncRangeCommand(range), activeTextEditor, output, globalState);
            if (!update_result) {
                vscode.window.showErrorMessage("Failed to update the range's state in Python: " + range.hash + " -- " + update_result);
                break;
            }
            // Trigger a onDidChangeTextEditorSelection event:
            const refreshed_ranges = await getCurrentRangesFromPython(activeTextEditor, output, globalState, {
                rebuild: false
            });
            if (refreshed_ranges) {
                updateDecorations(activeTextEditor, refreshed_ranges);
            }
        }
    }
    let unlockCommand = getUnlockCommand();
    const update_result = await safeExecuteCodeInKernel(unlockCommand,activeTextEditor, output, globalState);
    if (!update_result) {
        vscode.window.showErrorMessage('Failed to unlock the Python kernel: ' + update_result);
    }
    else{
        console.log('>> unlockCommand successful: ', update_result);
    }
}



////////////////////////////////////////////////////////////////////////////////////////////////////
//    CONNECT TO INTERACTIVE WINDOW AS WELL AS KERNEL:
////////////////////////////////////////////////////////////////////////////////////////////////////

enum State {
    settings_not_ok = 'settings_not_ok',
    initializable = 'initializable',
    initializable_messaged = 'initializable_messaged',
    initialization_started = 'initialization_started',
    kernel_found = 'kernel_found',
    kernel_available = 'kernel_available',
    instantialization_started = 'instantialization_started',
    extension_available = 'extension_available',
    explit_execution_started = 'explit_execution_started',
    implicit_execution_started = 'implicit_execution_started',
}

const initial_states = [State.settings_not_ok, State.initializable_messaged];

const stateTransitions: Map<State, State[]> = new Map([
    [State.settings_not_ok, [State.initializable]],
    [State.initializable, [State.initialization_started].concat(initial_states) ],
    [State.initializable_messaged, [State.initialization_started, State.settings_not_ok]],
    [State.initialization_started, [State.kernel_found].concat(initial_states) ],
    [State.kernel_found, [State.kernel_available].concat(initial_states) ],
    [State.kernel_available, [State.instantialization_started, State.extension_available].concat(initial_states) ],
    [State.instantialization_started, [State.extension_available].concat(initial_states) ],
    [State.extension_available, [State.explit_execution_started, State.implicit_execution_started].concat(initial_states) ],
    [State.explit_execution_started, [State.extension_available] ],
    [State.implicit_execution_started, [State.extension_available] ],
]);

function getState(globalState: Map<string, string>, editor: TextEditor): State {
    return globalState.get(editorConnectionStateKey(editor.document.uri.toString())) as State || State.settings_not_ok;
}

function updateState(globalState: Map<string, string>, editor: TextEditor, newState_: string ) {
    let newState = newState_ as State;
    if (!newState) {
        throw new Error('Invalid state: ' + newState);
    }
    let uri = editor.document.uri.toString();
    let currentState = getState(globalState, editor);
    if (!currentState) {
        if (newState === State.settings_not_ok) {
            globalState.set(editorConnectionStateKey(uri), newState);
            return;
        }
        else {
            window.showErrorMessage('Invalid state transition: ' + currentState + ' -> ' + newState);
            throw new Error('Invalid initial state: ' + newState + ' , please initialize your editor first');
        }
    }
    let acceptedTransitions = stateTransitions.get(currentState as State);
    if (acceptedTransitions && acceptedTransitions.includes(newState)) {
        globalState.set(editorConnectionStateKey(uri), newState);
        console.log(' -> State transition: ' + currentState + ' -> ' + editorConnectionStateKey(uri) + ' : ' + getState(globalState, editor));
    }
    else {
        window.showErrorMessage('Invalid state transition: ' + currentState + ' -> ' + newState);
        throw new Error('Invalid state transition: ' + currentState + ' -> ' + newState);
    }
}

function checkSettings(globalState: Map<string, string>, editor: TextEditor, ) {
    // After this function, you are: in settings_not_ok if they are not ok, or in THE SAME PREVIOUS STATE if they are, except if you were in settings_not_ok, in which case you are in initializable
    // Obviously, returns True if settings are ok, else False
    
    // const setting = vscode.workspace.getConfiguration('jupyter').get<string>('interactiveWindow.viewColumn');
    const creationMode = vscode.workspace.getConfiguration('jupyter').get<string>('interactiveWindow.creationMode');
    const shiftEnter = vscode.workspace.getConfiguration('jupyter').get<string>('interactiveWindow.textEditor.executeSelection');
    const perFileMode = creationMode && ['perFile', 'perfile'].includes(creationMode);
    const shiftEnterOff = shiftEnter && shiftEnter === 'false';

    if (perFileMode && shiftEnterOff && getState(globalState, editor) === State.settings_not_ok) {
        updateState(globalState, editor, State.initializable);
        return true;
    }
    else if (!perFileMode || !shiftEnterOff) {
        updateState(globalState, editor, State.settings_not_ok);
        window.showErrorMessage('Reactive Python: To use Reactive Python, please set the Jupyter extension settings to: \n - "jupyter.interactiveWindow.creationMode": "perFile" \n - "jupyter.interactiveWindow.textEditor.executeSelection": false');
        return false;
    }
}
    
function displayInitializationMessageIfNeeded(globalState: Map<string, string>, editor: TextEditor) {
    if (getState(globalState, editor) === State.initializable) {
        window.showInformationMessage("Reactive Python: Initialize the extension on this File with the command: 'Initialize Reactive Python' or the CodeLens at the top");
        updateState(globalState, editor, State.initializable_messaged);
    }
}

// async function getAllKernelsList(): Promise<Map<NotebookDocument, Kernel | undefined>> {
//     const extension = extensions.getExtension<Jupyter>('ms-toolsai.jupyter');
// 	if (!extension) {
//         window.showErrorMessage('Jupyter extension not installed');
//         throw new Error('Jupyter extension not installed');
// 	}
//     if (!extension.isActive) { await extension.activate(); }
//     const api = extension.exports;
//     let notebookDocuments = workspace.notebookDocuments;
//     let notebookToKernel: Map<NotebookDocument, Kernel | undefined> = new Map();
//     await Promise.all(
//         notebookDocuments.map(async (document) => {
//         const kernel = await api.kernels.getKernel(document.uri);
//         if (kernel && (kernel as any).language === 'python') {  notebookToKernel.set(document, kernel); }
//         else { notebookToKernel.set(document, undefined); }
//     }));
//     return notebookToKernel;
// }
async function getKernelNotebook(document: NotebookDocument): Promise<Kernel | undefined> {
    const extension = extensions.getExtension<Jupyter>('ms-toolsai.jupyter');
	if (!extension) {
        window.showErrorMessage('Jupyter extension not installed');
        throw new Error('Jupyter extension not installed');
	}
    if (!extension.isActive) { await extension.activate(); }
    const api = extension.exports;
    return new Promise<Kernel | undefined>(async (resolve) => {
        const kernel = await api.kernels.getKernel(document.uri);
        if (kernel && (kernel as any).language === 'python') { resolve(kernel); } else { resolve(undefined); }
    });
}

function isThereANewNotebook(oldNotebooks:readonly  NotebookDocument[], newNotebooks: readonly NotebookDocument[]): NotebookDocument | undefined {
    // This returns a notebook, if either there is in BOTH OLD AND NEW, but in New it has ONE MORE CELL.
    // Or it is ONLY IN NEW, and it has at least one cell.
    let newNotebook: NotebookDocument | undefined = undefined;
    for (let i = 0; i < newNotebooks.length; i++) {
        let oldNotebook = oldNotebooks.find((doc) => doc.uri.toString() === newNotebooks[i].uri.toString());
        if (oldNotebook && newNotebooks[i].cellCount > oldNotebook.cellCount)  { newNotebook = newNotebooks[i]; break; }
        else if (!oldNotebook && newNotebooks[i].cellCount > 0) { newNotebook = newNotebooks[i]; break; }
    }
    return newNotebook;
}


async function getNotebookAndKernel(globalState: Map<string, string>, editor: TextEditor, notify: Boolean = false): Promise<[NotebookDocument, Kernel] | undefined> {
    let notebook_uri = globalState.get(editorToIWKey(editor.document.uri.toString()));
    let iWsWCorrectUri = vscode.workspace.notebookDocuments.filter((doc) => doc.uri.toString() === notebook_uri);
    if (iWsWCorrectUri.length === 0) {
        if (notify) { window.showInformationMessage("Lost connection to this editor's Interactive Window. Please initialize it with the command: 'Initialize Reactive Python' or the CodeLens at the top ") }
        return undefined;
        
    }
    let notebook = iWsWCorrectUri[0];
    let kernel = await getKernelNotebook(notebook);
    if (!kernel) {
        if (notify) { window.showInformationMessage("Lost connection to this editor's Python Kernel. Please initialize it with the command: 'Initialize Reactive Python' or the CodeLens at the top ") }
        return undefined;
    }
    return [notebook, kernel];
}


// Break everything:
// let IW_uri = globalState.get(editorToIWKey(uri));
// let kernel_uri = globalState.get(editorToKernelKey(uri));
// let iWsWCorrectUri = notebookDocuments.filter((doc) => doc.uri.toString() === IW_uri);
// await globalState.set(editorIWCreationLockKey(uri), 'true');


const editorToIWKey = (editorUri: string) => 'editorToIWKey' + editorUri;
const editorToKernelKey = (editorUri: string) => 'editorToIWKey' + editorUri;
const editorConnectionStateKey = (editorUri: string) => 'state' + editorUri;

const welcomeText = "# Welcome to Reactive Python";




async function initializeInteractiveWindowAndKernel(globalState: Map<string, string>, editor: TextEditor) {

    let uri = 
    // Check settings:
    checkSettings(globalState, editor);

    // If not in state initializable or initializable_messaged, return:
    let currentState = getState(globalState, editor)
    if (currentState !== State.initializable && currentState !== State.initializable_messaged) {
        console.log('Invalid state for initialization: ' + currentState);
        return false;
    }

    // Start initializing:
    updateState(globalState, editor, State.initialization_started);
    
    let n_attempts = 5;
    while (n_attempts > 0) {
        
        const notebookDocuments = vscode.workspace.notebookDocuments;
        // Wreck some Havoc: This should ALWAYS RESULT IN A RUNNING KERNEL, AND ALSO A NEW CELL SOMEWHERE, EVENTUALLY. This is the idea, at least...
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
        if (!newNotebook) { n_attempts -= 1; continue }
        
        globalState.set(editorToIWKey(editor.document.uri.toString()), newNotebook.uri.toString());
        let okFoundKernel = await getNotebookAndKernel(globalState, editor, );
        if (!okFoundKernel) { n_attempts -= 1; continue}
        updateState(globalState, editor, State.kernel_found);
        
        let [notebook, kernel] = okFoundKernel;
        let is_last_cell_ok = false;
        for (let i = 0; i < 20; i++) {  // Try 20 times to read the last cell:
            await new Promise((resolve) => setTimeout(resolve, 100));
            let lastCell = notebook.cellAt(notebook.cellCount - 1);
            is_last_cell_ok = lastCell.document.getText() === welcomeText;
            if (is_last_cell_ok) { break; }
        }
        if (!is_last_cell_ok) { n_attempts -= 1; updateState(globalState, editor, State.initialization_started); }
        updateState(globalState, editor, State.kernel_available); 
        break;
    }

    // After this, the only possible states SHOULD be State.kernel_available or State.initialization_started:

    let state_now = getState(globalState, editor)
    if (state_now === State.initialization_started) {
        window.showErrorMessage('Failed to initialize the Interactive Window and the Python Kernel');
        updateState(globalState, editor, State.initializable_messaged);
        return false;
    }
    else if (state_now === State.kernel_available) {
        window.showInformationMessage('Successfully initialized the Interactive Window and the Python Kernel');
        return true;
    }
    else {
        // Throw:
        throw new Error('Invalid state: ' + state_now);
    }
}





////////////////////////////////////////////////////////////////////////////////////////////////////
// PYTHON COMMANDS AND SNIPPETS
////////////////////////////////////////////////////////////////////////////////////////////////////

export const getCommandToGetRangeToSelectFromPython = (currentQuery: string): string => {
    /*
    FAKE
    */
    // Pass currentQuery as a SIMPLE STRING, ie all the newlines should be passed in as explicit \n and so on:
    // Turn currentQuery in a string that can be passed to Python, by sanitizing all the Newlines, Quotes and Indentations
    // (ie, keeping them but in a way that Python can receive as a string):
    return 'get_commands_to_execute("""' + currentQuery + '""")';
};

export const getCommandToGetAllRanges = (
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
    // return `if "reactive_python_dag_builder_utils__" in globals():\n\treactive_python_dag_builder_utils__.update_dag_and_get_ranges(code= ${text_}, current_line=${current_line_str}, get_upstream=${upstream_param}, get_downstream=${downstream_param}, stale_only=${stale_only_param})\nelse:\n\t[]`;
    if (to_launch_compute) {
        return `reactive_python_dag_builder_utils__.ask_for_ranges_to_compute(code= ${text_}, current_line=${current_line_str}, get_upstream=${upstream_param}, get_downstream=${downstream_param}, stale_only=${stale_only_param})`;
    } else {
        return `reactive_python_dag_builder_utils__.update_dag_and_get_ranges(code= ${text_}, current_line=${current_line_str}, get_upstream=${upstream_param}, get_downstream=${downstream_param}, stale_only=${stale_only_param})`;
    }
};

export const getSyncRangeCommand = (range: AnnotatedRange): string => {
    return `reactive_python_dag_builder_utils__.set_locked_range_as_synced(${range.hash})`;
};

export const getUnlockCommand = (): string => {
    return `reactive_python_dag_builder_utils__.unlock()`;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// HIGHLIGHTING UTILS
////////////////////////////////////////////////////////////////////////////////////////////////////
// import { TextEditor, Range } from 'vscode';

export const getEditorAllText = (editor: TextEditor): { text: string | null } => {
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
    // >> You MIGHT be interested in
    // /Users/michele.tasca/Documents/vscode-extensions/vscode-reactive-jupyter/src/platform/terminals/codeExecution/codeExecutionHelper.node.ts
    //  >> CodeExecutionHelper >> normalizeLines  ...
    text = text.replace(/'/g, "\\'");
    text = text.replace(/"/g, '\\"');
    text = text.replace(/\n/g, '\\n');
    text = text.replace(/\r/g, '\\r');
    text = text.replace(/\t/g, '\\t');
    text = '"""' + text + '"""';
    return text;
}

export const getEditorCurrentLineNum = (editor: TextEditor): number | null => {
    if (!editor || !editor.document || editor.document.uri.scheme === 'output') {
        return null;
    }
    const currentLineNum = editor.selection.active.line;
    return currentLineNum;
};

export const getEditorCurrentText = (editor: TextEditor): { currentQuery: string; currentRange: Range | null } => {
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

export const parseResultFromPythonAndGetRange = (resultFromPython: string): AnnotatedRange[] | null => {
    // ResultFromPython is a string of the form: "[[startLine, endline, state], [startLine, endline, state], ...]" (the length is indefinite)
    // Parse it and return the list of ranges to select:

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
    // console.log("5: Here we are: ", resultFromPython)
    try {
        resultFromPythonParsed = JSON.parse(resultFromPython);
    } catch (e) {
        console.log('Failed to parse result from Python: ' + resultFromPython);
        vscode.window.showErrorMessage('Reactive Python: Failed to parse JSON result from Python: ' + resultFromPython);
        return null;
    }
    // console.log("6: Here we are: ", resultFromPythonParsed)
    // Assert that it worked:
    if (resultFromPythonParsed === undefined) {
        console.log('Failed to parse result from Python: ' + resultFromPython);
        return null;
    }
    // console.log("7: Here we are: ", resultFromPythonParsed)
    // Convert to Range[]:
    let ranges: AnnotatedRange[] = [];
    for (let i = 0; i < resultFromPythonParsed.length; i++) {
        let startLine = resultFromPythonParsed[i][0];
        let endLine = resultFromPythonParsed[i][1];
        let state = resultFromPythonParsed[i][2];
        let current = resultFromPythonParsed[i][3] == 'current';
        let text_ = resultFromPythonParsed[i].length > 4 ? resultFromPythonParsed[i][4] : undefined;
        let hash = resultFromPythonParsed[i].length > 5 ? resultFromPythonParsed[i][5] : undefined;
        if (startLine >= 0 && endLine >= 0 && startLine <= endLine && (state === 'synced' || state === 'outdated')) {
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
    }
    return ranges;
};

const getCurrentRangesFromPython = async (
    editor: TextEditor,
    output: OutputChannel | null,
    globalState: Map<string, string>, 
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
    if (current_line === null) {
        console.log('getCurrentRangesFromPython: current_line is none');
    }
    // console.log("1: Here we are!")
    let linen = current_line === undefined ? getEditorCurrentLineNum(editor) : current_line;
    let text: string | null = rebuild ? getEditorAllText(editor).text : null;
    if (!text && !linen) return;
    text = text ? formatTextAsPythonString(text) : null;
    // console.log("2: Here we are: ", text)
    let command = getCommandToGetAllRanges(text, linen, upstream, downstream, stale_only, to_launch_compute);
    // console.log('3: Here we are: ', command)
    const result = await safeExecuteCodeInKernel(command, editor, output, globalState);
    // console.log('4: Result from Python: ' + result);
    if (result === undefined ||  result == '[]') return;
    if (to_launch_compute) {
        console.log('Result from Python to compute: ' + result);
    }
    const ranges_out = await parseResultFromPythonAndGetRange(result);
    if (!ranges_out) return;
    return ranges_out;
};

export const getTextInRanges = (ranges: AnnotatedRange[]): string[] => {
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

const HighlightSynced = window.createTextEditorDecorationType({
    backgroundColor: { id: `jupyter.syncedCell` },
    borderColor: { id: `jupyter.syncedCell` },
    borderWidth: '0px',
    borderStyle: 'solid'
});
const HighlightSyncedCurrent = window.createTextEditorDecorationType({
    backgroundColor: { id: `jupyter.syncedCurrentCell` },
    borderColor: { id: `jupyter.syncedCurrentCell` },
    borderWidth: '0px',
    borderStyle: 'solid'
});
const HighlightOutdated = window.createTextEditorDecorationType({
    backgroundColor: { id: `jupyter.outdatedCell` },
    borderColor: { id: `jupyter.outdatedCell` },
    borderWidth: '0px',
    borderStyle: 'solid'
});
const HighlightOutdatedCurrent = window.createTextEditorDecorationType({
    backgroundColor: { id: `jupyter.outdatedCurrentCell` },
    borderColor: { id: `jupyter.outdatedCurrentCell` },
    borderWidth: '0px',
    borderStyle: 'solid'
});

let updateDecorations = async (editor: TextEditor, ranges_out: AnnotatedRange[]) => {
    // if (!Config.highlightQuery) return;
    if (
        !editor ||
        !editor.document ||
        editor.document.uri.scheme === 'output'
        //  || !this.registeredLanguages.includes(editor.document.languageId) // <--- COMES FROM this.registeredLanguages = Config.codelensLanguages;  // See Config below
    ) {
        return;
    }
    try {
        // const { currentRange, currentQuery } = getEditorCurrentText(editor);
        // if (!currentRange || !currentQuery) return;
        // let command = getCommandToGetRangeToSelectFromPython(currentRange.start.line.toString());

        // Save the range associated with this editor:
        // 1. Get the Kernel uuid:
        // let kernel_uuid = editor.document.uri;
        // // 2. Set current editor ranges in global state:
        // await globalState.update(kernel_uuid.toString() + '_ranges', ranges_out);
        // // console.log('>>>>>>>>Global State updated');

        // Set highlight on all the ranges in ranges_out with state == 'synced'
        let sync_ranges = ranges_out.filter((r) => r.state == 'synced' && !r.current).map((r) => r.range);
        editor.setDecorations(HighlightSynced, sync_ranges);
        let sync_curr_ranges = ranges_out.filter((r) => r.state == 'synced' && r.current).map((r) => r.range);
        editor.setDecorations(HighlightSyncedCurrent, sync_curr_ranges);
        let out_ranges = ranges_out.filter((r) => r.state == 'outdated' && !r.current).map((r) => r.range);
        editor.setDecorations(HighlightOutdated, out_ranges);
        let out_curr_ranges = ranges_out
            .filter((r) => (r.state == 'outdated' || r.state == 'syntaxerror') && r.current)
            .map((r) => r.range);
        editor.setDecorations(HighlightOutdatedCurrent, out_curr_ranges);
    } catch (error) {
        // console.log('update decorations failed: %O', error);
    }
};

// let updateDecorations = async (editor: TextEditor, globalState: vscode.Memento) => {
//     try {
//         // Get the Kernel uuid:
//         let kernel_uuid = editor.document.uri;
//         // Get the ranges from global state:
//         let ranges_out: { range: Range; state: string }[] | undefined = await globalState.get(
//             kernel_uuid.toString() + '_ranges'
//         );
//         console.log('>>>>>>>>REFRESHING the ranges: ');
//         console.log(ranges_out);

//         if (ranges_out) {
//             console.log('>>>>>>>>YEE its NOT undefined! !!');

//         }
//         console.log('\n\n');
//     } catch (error) {
//         console.log('update decorations failed: %O', error);
//     }
// };



////////////////////////////////////////////////////////////////////////////////////////////////////
// CODELENS
////////////////////////////////////////////////////////////////////////////////////////////////////

export class CellCodelensProvider implements vscode.CodeLensProvider {
    private codeLenses: vscode.CodeLens[] = [];
    private range: Range | undefined;
    private _onDidChangeCodeLenses: vscode.EventEmitter<void> = new vscode.EventEmitter<void>();
    public readonly onDidChangeCodeLenses: vscode.Event<void> = this._onDidChangeCodeLenses.event;

    change_range(new_range: Range | undefined) {
        // console.log('>>>>>>>>>>>>>>>>>>NICE, FIRED RESET!. ');
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
    ): vscode.CodeLens[] | Thenable<vscode.CodeLens[]> {
        let editor = vscode.window.activeTextEditor;
        if (editor && this.range && editor.document.uri == document.uri) {
            // Current line:
            this.codeLenses = [
                new vscode.CodeLens(new vscode.Range(this.range.start.line, 0, this.range.end.line, 0), {
                    title: 'sync upstream',
                    tooltip: 'Run all outdated code upstream, including this cell',
                    command: 'jupyter.sync-upstream',
                    arguments: [this.range]
                }),
                new vscode.CodeLens(new vscode.Range(this.range.start.line, 0, this.range.end.line, 0), {
                    title: 'sync downstream',
                    tooltip: 'Run all outdated code downstream, including this cell',
                    command: 'jupyter.sync-downstream',
                    arguments: [this.range]
                }),
                new vscode.CodeLens(new vscode.Range(this.range.start.line, 0, this.range.end.line, 0), {
                    title: 'sync all',
                    tooltip: 'Run all outdated code upstream and downstream, including this cell',
                    command: 'jupyter.sync-all',
                    arguments: [this.range]
                })
            ];
            return this.codeLenses;
        }
        return [];
    }
}

export class InitialCodelensProvider implements vscode.CodeLensProvider {
    private started_at_least_once: vscode.CodeLens[] = [];
    public provideCodeLenses(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): vscode.CodeLens[] | Thenable<vscode.CodeLens[]> {
        // console.log('>>>>>>>>>>>>>>>>>>NICE, INVOKED ONCE. ');
        let editor = vscode.window.activeTextEditor;
        // console.log('>>>>>>>>>>>>>>>>>>NICE, INVOKED HERE. ' + editor);
        if (editor && editor.document.uri == document.uri) {
            // console.log('>>>>>>>>>>>>>>>>>>INSIDE THE IF. ' + editor);
            let codeLenses = [
                new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), {
                    title: '$(debug-start) Initialize Reactive Python',
                    tooltip: 'Initialize Reactive Python on the current file',
                    command: 'jupyter.initialize-reactive-python-extension'
                    // arguments: [this.range] // Wanna pass the editor uri?
                }),
                new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), {
                    title: 'Sync all Stale code',
                    tooltip: 'Sync all Stale code in current file',
                    command: 'jupyter.sync-all'
                    // arguments: [this.range] // Wanna pass the editor uri?
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




export function createPreparePythonEnvForReactivePython(globalState: Map<string, string>, output: OutputChannel) {
    async function preparePythonEnvForReactivePython() {
        /* IDEA:
        we DON'T want to run this on Activation, but WE ARE DOING IT FOR NOW. for easier testing.
        */
        let command = scriptCode + '\n\n\n"Reactive Python Activated"\n';
        let editor = window.activeTextEditor;
        if (!editor) { return; }
        
        if ([State.initializable, State.initializable_messaged].includes(getState(globalState, editor))) {
            let success = await initializeInteractiveWindowAndKernel(globalState, editor);
            if (!success) { return; }
        }
        // Here, you should ALWAYS be in State.kernel_available ...

        let instantiated_script = await safeExecuteCodeInKernelForInitialization(command, editor, output, globalState);
        if (!instantiated_script) { return; }
        
        // Immediatly start coloring ranges:

        let refreshed_ranges = await getCurrentRangesFromPython(editor, null, globalState, {
                rebuild: true,
                current_line: null
            }); // Do this in order to immediatly recompute the dag in the python kernel
        if (refreshed_ranges) {
            updateDecorations(editor, refreshed_ranges);
        }
    }
    return preparePythonEnvForReactivePython;
}

export function createComputeAction(config: {
        rebuild: boolean;
        current_line?: number | undefined | null;
        upstream: boolean;
        downstream: boolean;
        stale_only: boolean;
        to_launch_compute: boolean;
    }, globalState: Map<string, string>, output: OutputChannel) {
    async function computeAction() {
        let editor = window.activeTextEditor;
        if (!editor) return;
        if (getState(globalState, editor) !== State.extension_available) { return; }
        const current_ranges = await getCurrentRangesFromPython( editor, output, globalState, config, );
        await queueComputation(current_ranges, editor, globalState, output);
    }
    return computeAction;
}





////////////////////////////////////////////////////////////////////////////////////////////////////
// ACTIVATION
////////////////////////////////////////////////////////////////////////////////////////////////////   class_weight={0: 0.9}


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

export const globalState: Map<string, string> = new Map();

export function activate(context: ExtensionContext) {
    console.log('>>>>  BRO:.....  SO???');
	const jupyterExt = extensions.getExtension<Jupyter>('ms-toolsai.jupyter');
	if (!jupyterExt) {
		throw new Error('The Jupyter Extension not installed. Please install it and try again.');
	}
	if (!jupyterExt.isActive) {
		jupyterExt.activate();
	}
	const output = window.createOutputChannel('Jupyter Kernel Execution');
	context.subscriptions.push(output);

    currentContext = context as typeof currentContext;

    defineAllCommands(context, output, globalState);

    CellOutputDisplayIdTracker.activate();
}




export async function defineAllCommands(context: ExtensionContext, output: OutputChannel, globalState: Map<string, string>
) {
	// context.subscriptions.push(
	// 	commands.registerCommand('jupyterKernelExecution.listKernels', async () => {
	// 		const kernel = await selectKernel();
	// 		if (!kernel) {
	// 			return;
	// 		}
	// 		const code = "12+15";
	// 		if (!code) {
	// 			return;
	// 		}
	// 		await executeCodeInKernel(code, kernel, output);
	// 	})
	// );

    // The command has been defined in the package.json file // Now provide the implementation of the command with registerCommand // The commandId parameter must match the command field in package.json
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'jupyter.initialize-reactive-python-extension',
            createPreparePythonEnvForReactivePython(globalState, output)
        )
    );
    context.subscriptions.push(
        vscode.commands.registerCommand('jupyter.sync-downstream', createComputeAction( { rebuild: true, upstream: false, downstream: true, stale_only: true, to_launch_compute: true }, globalState, output))
    );
    context.subscriptions.push(
        vscode.commands.registerCommand('jupyter.sync-upstream', createComputeAction( { rebuild: true, upstream: true, downstream: false, stale_only: true, to_launch_compute: true }, globalState, output))
    );
    context.subscriptions.push(
        vscode.commands.registerCommand( 'jupyter.sync-upstream-and-downstream', createComputeAction( { rebuild: true, upstream: true, downstream: true, stale_only: true, to_launch_compute: true }, globalState, output))
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('jupyter.sync-all', createComputeAction( { rebuild: true, current_line: null, upstream: true, downstream: true, stale_only: true, to_launch_compute: true }, globalState, output))
    );

    // await preparePythonEnvForReactivePython(output);

    ///////// Codelens: ///////////////////////

    const codelensProvider = new CellCodelensProvider();
    languages.registerCodeLensProvider('python', codelensProvider);
    const initializingProvider = new InitialCodelensProvider();
    languages.registerCodeLensProvider('python', initializingProvider);

    ///////// Document Highlights: ///////////////////////

    window.onDidChangeActiveTextEditor(
        async (editor) => {
            if (editor) {
                if (getState(globalState, editor) !== State.extension_available) { return; }
                const current_ranges = await getCurrentRangesFromPython(editor, output, globalState, { rebuild: true });
                if (!current_ranges) return;
                await updateDecorations(editor, current_ranges);
            }
        },
        null,
        Context.subscriptions
    );
    workspace.onDidChangeTextDocument(
        // This: Exists too! >> onDidSaveTextDocument  >> (even if should be included in the onDidChangeTextDocument one ? )
        // editors, undo/ReactDOM, save, etc
        async (event) => {
            let editor = window.activeTextEditor;
            if (editor && event.document === editor.document) {
                if (getState(globalState, editor) !== State.extension_available) { return; }
                const current_ranges = await getCurrentRangesFromPython(editor, output, globalState, {
                    rebuild: true
                });
                if (!current_ranges) return;
                await updateDecorations(editor, current_ranges);
            }
        },
        null,
        Context.subscriptions
    );
    window.onDidChangeTextEditorSelection(
        async (event) => {
            // console.log('----- Here 1! ');
            let editor = window.activeTextEditor;
            if (
                event.textEditor &&
                editor &&
                event.textEditor.document === editor.document &&
                editor.selection.isEmpty
            ) {
                // console.log('----- Here 2! ');
                if (getState(globalState, editor) !== State.extension_available) { return; }
                const current_ranges = await getCurrentRangesFromPython(editor, output, globalState, {
                    rebuild: false
                });
                // console.log('----- Here 3!, current_ranges: ', current_ranges);
                if (current_ranges) {
                    updateDecorations(editor, current_ranges);
                }
                let codelense_range = current_ranges ? current_ranges.filter((r) => r.current).map((r) => r.range) : [];
                // console.log('----- Here 4! ', codelense_range);
                codelensProvider.change_range(codelense_range.length > 0 ? codelense_range[0] : undefined);
            } else {
                // console.log('----- Here 5! ');
                updateDecorations(event.textEditor, []);
                codelensProvider.change_range(undefined);
            }
        },
        null,
        Context.subscriptions
    );
    ///////// Do highlighting for the first time right now: ///////////////////////
    // if (window.activeTextEditor) {
    //     updateDecorations(window.activeTextEditor, kernel, output);
    // } else {
    //     vscode.window.showInformationMessage(
    //         'No active text editor on Activation Time. Try to retrigger the highlighter somehow.'
    //     );
    // }

    ///////// CodeLenses: ///////////////////////
    // Add a codelens above the line where the cursor is, that launches the "jupyter.test-command" command:
    // const codelensProvider = new MyCodeLensProvider();
    // const disposable = languages.registerCodeLensProvider({ language: 'python' }, codelensProvider);
    // context.subscriptions.push(disposable);
    // HINT: ONE version of this is /Users/michele.tasca/Documents/vscode-extensions/vscode-reactive-jupyter/src/interactive-window/editor-integration/codelensprovider.ts !!
}