
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


////////////////////////////////////////////////////////////////////////////////////////////////////
//    RUN PYTHON CODE: 2 WAYS: IN KERNEL AND IN INTERACTIVE WINDOW
////////////////////////////////////////////////////////////////////////////////////////////////////


async function* executeCodeStreamInKernel(code: string, kernel: Kernel, output_channel: OutputChannel | null): AsyncGenerator<string | ExecutionError, void, unknown> {
    /*
    Currently, it ALWAYS logs a line (for the user), and it returns the result if it is NOT AN ERROR, else Undefined. 
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
    NotebookClodsed = 'NotebookClodsed'
}

async function executeCodeInInteractiveWindow(
    text: string,
    notebook: NotebookDocument,
    output: OutputChannel | null,
): Promise<IWExecutionResult> {
    // Currently returns an IWExecutionResult. TODO: Rethink if you need something more...

    await vscode.commands.executeCommand('jupyter.execSelectionInteractive', text);
    // OTHER THINGS I TRIED:
    // let res = await getIWAndRunText(serviceManager, activeTextEditor, text);
    // let res = await executeCodeInKernel(text, kernel, output);
    // let res = await interactiveWindow.addNotebookCell( text, textEditor.document.uri, textEditor.selection.start.line, notebook )
    
    // ^ In particular, it would be a REALLY GOOD IDEA to do addNotebookCell ^ (which already works)
    // and then TAKE CONTROL OF THE WHOLE cellExecutionQueue mechanism, an Reimplement it here, by ALWAYS SENDING THINGS TO THE KERNEL IMPLICITELY 
    // and then streaming the output to the NotebookDocyment ourselves, the same way Jupyter does it.. 
    // PROBLEM: That's a Lot of work. For now, I'll take advantage of the execSelectionInteractive Command because it's Easier...
    
    let cell: NotebookCell | undefined = undefined;
    for (let i = 0; i < 80; i++) {  // Try 20 times to read the last cell:
        await new Promise((resolve) => setTimeout(resolve, 250));
        let lastCell = notebook.cellAt(notebook.cellCount - 1);
        let last_index = notebook.cellCount - 1;
        cell = getBestMatchingCell(lastCell, notebook, last_index, text);
        if (cell) { break; }
    }

    if (!cell) {
        window.showErrorMessage('Reactive Jupyter: Failed to execute the code in the Interactive Window: No matching cell was identified');
        return IWExecutionResult.NotebookClodsed;
    }

    let cell_state = CellState.Undefined;
    await new Promise((resolve) => setTimeout(resolve, 250));
    for (let i = 0; i > -1; i++) {
        if (!notebook || notebook.isClosed || cell.notebook.isClosed) {
            return IWExecutionResult.NotebookClodsed;
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
        await new Promise((resolve) => setTimeout(resolve, 250));
    }

    window.showErrorMessage('Reactive Jupyter: Failed to execute the code in the Interactive Window: The cell did not finish executing');
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
    displayInitializationMessageIfNeeded(globals, editor);
    if (getKernelState(globals, editor) !== expected_initial_state) { return; }
    if (!checkSettings(globals, editor)) { return; }

    let notebookAndKernel = await getNotebookAndKernel(globals, editor, true);
    if (!notebookAndKernel) {
        window.showErrorMessage("Reactive Jupyter: Lost Connection to this editor's Kernel. Please initialize the extension with the command: 'Initialize Reactive Jupyter' or the CodeLens at the top");
        updateKernelState(globals, editor, KernelState.initializable_messaged);
        return;
    }
    let [notebook, kernel] = notebookAndKernel;
    updateKernelState(globals, editor, KernelState.implicit_execution_started);
    const result = await executeCodeInKernel(command, kernel, null);  // output ?
    if (return_to_initial_state) { updateKernelState(globals, editor, expected_initial_state); }
    return result;
}
async function safeExecuteCodeInKernelForInitialization(
    command: string,
    editor: TextEditor,
    output: OutputChannel | null,
    globals: Map<string, string>
): Promise<boolean> {
    // It's SLIGHTLY different from the above one, in ways I didn't bother to reconcile...
    if (getKernelState(globals, editor) !== KernelState.kernel_available) { return false; }
    if (!checkSettings(globals, editor)) { return false; }

    let notebookAndKernel = await getNotebookAndKernel(globals, editor, true);
    if (!notebookAndKernel) {
        window.showErrorMessage("Reactive Jupyter: Kernel Initialization succeeded, but we lost it already... Please try again.");
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
        window.showErrorMessage('Reactive Jupyter: The initialization code could not be executed in the Python Kernel. This is bad...');
        return false;
    }
}

async function safeExecuteCodeInInteractiveWindow(
    command: string,
    editor: TextEditor,
    output: OutputChannel | null,
    globals: Map<string, string>,
    expected_initial_state: KernelState = KernelState.extension_available,
    return_to_initial_state: boolean = true
) {
    displayInitializationMessageIfNeeded(globals, editor);
    if (getKernelState(globals, editor) !== expected_initial_state) { return; }
    if (!checkSettings(globals, editor)) { return; }

    let notebookAndKernel = await getNotebookAndKernel(globals, editor, true);
    if (!notebookAndKernel) {
        window.showErrorMessage("Reactive Jupyter: Lost Connection to this editor's Notebook. Please initialize the extension with the command 'Initialize Reactive Jupyter' or the CodeLens at the top");
        updateKernelState(globals, editor, KernelState.initializable_messaged);
        return;
    }
    let [notebook, kernel] = notebookAndKernel;
    updateKernelState(globals, editor, KernelState.explicit_execution_started);
    const result = await executeCodeInInteractiveWindow(command, notebook, output);
    if (result == IWExecutionResult.NotebookClodsed) {
        window.showErrorMessage("Reactive Jupyter: Lost Connection to this editor's Notebook. Please initialize the extension with the command 'Initialize Reactive Jupyter' or the CodeLens at the top");
        updateKernelState(globals, editor, KernelState.initializable_messaged);
    }
    else if (return_to_initial_state) { updateKernelState(globals, editor, expected_initial_state); }
    return result;
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
    if (current_ranges) {
        let said_dependsonotherstalecode_message = false;
        for (let range of current_ranges) {
            if (!range.text) break;

            if (range.state === 'dependsonotherstalecode') {
                if (!said_dependsonotherstalecode_message) {
                    let text = range.text.slice(0, 100);
                    window.showErrorMessage('Reactive Jupyter: ' + text + ' depends on other code that is outdated. Please update the other code first.');
                    said_dependsonotherstalecode_message = true;
                }   
                continue;
            }

            let res = await safeExecuteCodeInInteractiveWindow(range.text, activeTextEditor, output, globals);
            if ( res != IWExecutionResult.Succeeded ) { break; }

            const update_result = await safeExecuteCodeInKernel(getSyncRangeCommand(range), activeTextEditor, output, globals);
            if (!update_result) {
                vscode.window.showErrorMessage("Reactive Jupyter: Failed to update the range's state in Python: " + range.hash + " -- " + update_result);
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
        vscode.window.showErrorMessage('Reactive Jupyter: Failed to unlock the Python kernel: ' + update_result);
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


function getState<STATE>(globals: Map<string, string>, key: string): STATE | boolean {
    return (globals.get(key) as STATE) || false;
}

function updateState<STATE>(globals: Map<string, string>, key: string, newState_: string, stateTransitions: Map<STATE, STATE[]>, initialStates: STATE[] = []) {
    let newState = newState_ as STATE;
    if (!newState) {
        throw new Error('Invalid state: ' + newState);
    }
    let currentState = (globals.get(key) as STATE) || false;
    if (!currentState) {
        if (initialStates.includes(newState)) {
            globals.set(key, newState as string); // Cast newState to string
            return;
        }
        else {
            window.showErrorMessage('Reactive Jupyter: Invalid state transition: ' + currentState + ' -> ' + newState);
            throw new Error('Invalid initial state: ' + newState + ' , please initialize your editor first');
        }
    }
    let acceptedTransitions = stateTransitions.get(currentState as STATE);
    if (acceptedTransitions && acceptedTransitions.includes(newState)) {
        globals.set(key, newState as string);
    }
    else {
        window.showErrorMessage('Reactive Jupyter: Invalid state transition: ' + currentState + ' -> ' + newState);
        throw new Error('Invalid state transition: ' + currentState + ' -> ' + newState);
    }
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
    // After this function, you are: in settings_not_ok if they are not ok, or in THE SAME PREVIOUS STATE if they are, except if you were in settings_not_ok, in which case you are in initializable
    // Obviously, returns True if settings are ok, else False

    const creationMode = vscode.workspace.getConfiguration('jupyter').get<string>('interactiveWindow.creationMode');
    const shiftEnter = vscode.workspace.getConfiguration('jupyter').get<boolean>('interactiveWindow.textEditor.executeSelection');
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
    let iWsWCorrectUri = vscode.workspace.notebookDocuments.filter((doc) => doc.uri.toString() === notebook_uri);
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
const editorRebuildPendingKey = (editorUri: string) => 'rebuildPending' + editorUri;


type CachedNotebookDocument = { cellCount: number, uri: Uri };
const toMyNotebookDocument = (doc: NotebookDocument): CachedNotebookDocument => ({ cellCount: doc.cellCount, uri: doc.uri });


async function initializeInteractiveWindowAndKernel(globals: Map<string, string>, editor: TextEditor) {

    let currentState = getKernelState(globals, editor)
    if (currentState !== KernelState.initializable && currentState !== KernelState.initializable_messaged) {
        console.log('Invalid state for initialization: ' + currentState);
        return false;
    }

    // Start initializing:
    updateKernelState(globals, editor, KernelState.initialization_started);

    let n_attempts = 5;
    while (n_attempts > 0) {

        const notebookDocuments: CachedNotebookDocument[] = vscode.workspace.notebookDocuments.map(toMyNotebookDocument);
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

        globals.set(editorToIWKey(editor.document.uri.toString()), newNotebook.uri.toString());
        await new Promise((resolve) => setTimeout(resolve, 2000));
        let okFoundKernel: [NotebookDocument, Kernel] | undefined = undefined;
        for (let i = 0; i < 10; i++) {
            okFoundKernel = await getNotebookAndKernel(globals, editor,);
            if (okFoundKernel) { break; }
            window.showInformationMessage('Waiting for the Python Kernel to start...');
            await new Promise((resolve) => setTimeout(resolve, 2000));
        }

        if (!okFoundKernel) { n_attempts -= 1; continue }
        updateKernelState(globals, editor, KernelState.kernel_found);

        let [notebook, kernel] = okFoundKernel;
        let is_last_cell_ok = false;
        for (let i = 0; i < 20; i++) {  // Try 20 times to read the last cell:
            await new Promise((resolve) => setTimeout(resolve, 100));
            let lastCell = notebook.cellAt(notebook.cellCount - 1);
            is_last_cell_ok = lastCell.document.getText() === welcomeText;
            if (is_last_cell_ok) { break; }
        }
        if (!is_last_cell_ok) { n_attempts -= 1; updateKernelState(globals, editor, KernelState.initialization_started); }
        updateKernelState(globals, editor, KernelState.kernel_available);
        break;
    }

    // After this, the only possible states SHOULD be KernelState.kernel_available or KernelState.initialization_started:

    let state_now = getKernelState(globals, editor)
    if (state_now === KernelState.initialization_started) {
        window.showErrorMessage('Reactive Jupyter: Failed to initialize the Interactive Window and the Python Kernel');
        updateKernelState(globals, editor, KernelState.initializable_messaged);
        return false;
    }
    else if (state_now === KernelState.kernel_available) {
        window.showInformationMessage('Reactive Jupyter: Successfully initialized the Interactive Window and the Python Kernel');
        return true;
    }
    else {
        // Throw:
        throw new Error('Invalid state: ' + state_now);
    }
}



async function preparePythonEnvForReactivePython(editor: TextEditor, globals: Map<string, string>, output: OutputChannel) {

    let command = scriptCode + '\n\n\n"Reactive Jupyter Activated"\n';
    // if (output) { output.show(true); }

    if (getKernelState(globals, editor) == false) { globals.set(editorConnectionStateKey(editor.document.uri.toString()), KernelState.initializable); }

    checkSettings(globals, editor);

    if (editor.document.languageId !== 'python') {
        window.showErrorMessage('Reactive Jupyter: This extension only works when editing Python files. Please open a Python file and try again');
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
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// PYTHON COMMANDS AND SNIPPETS
////////////////////////////////////////////////////////////////////////////////////////////////////

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
    } catch (e) {
        vscode.window.showErrorMessage('Reactive Jupyter: Failed to parse JSON result from Python: ' + resultFromPython);
        return null;
    }
    // Assert that it worked:
    if (resultFromPythonParsed === undefined) {
        console.log('Failed to parse result from Python: ' + resultFromPython);
        return null;
    }
    // Convert to Range[]:
    let ranges: AnnotatedRange[] = [];
    for (let i = 0; i < resultFromPythonParsed.length; i++) {
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
    }
    return ranges;
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
    ): vscode.CodeLens[] | Thenable<vscode.CodeLens[]> {
        let editor = vscode.window.activeTextEditor;
        if (editor && this.range && editor.document.uri == document.uri) {
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
    ): vscode.CodeLens[] | Thenable<vscode.CodeLens[]> {
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

        let editor = window.activeTextEditor;
        if (!editor) { return; }
        preparePythonEnvForReactivePython(editor, globals, output);
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
        let editor = window.activeTextEditor;
        if (!editor) return;
        if (getKernelState(globals, editor) !== KernelState.extension_available) { return; }
        const current_ranges = await getCurrentRangesFromPython(editor, output, globals, config,);
        await queueComputation(current_ranges, editor, globals, output);
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
        let editor = window.activeTextEditor;
        if (!editor) return;
        await preparePythonEnvForReactivePython(editor, globals, output);
        if (getKernelState(globals, editor) !== KernelState.extension_available) { return; }
        const current_ranges = await getCurrentRangesFromPython(editor, output, globals, config,);
        await queueComputation(current_ranges, editor, globals, output);
    }
    return prepareEnvironementAndComputeAction;
}









// TODO- Right now, this is not much of a state machine. But there is the infrastructure in place to complicate it...
enum EditingState {
    rebuilt = 'rebuilt',
    rebuildPending = 'rebuildPending',
}
const highlightIinitialStates = [EditingState.rebuildPending, EditingState.rebuilt];
const highlightStateTransitions: Map<EditingState, EditingState[]> = new Map([
    [EditingState.rebuilt, [EditingState.rebuildPending, EditingState.rebuilt]],
    [EditingState.rebuildPending, [EditingState.rebuilt, EditingState.rebuildPending]],
]);

function updateEditingState(globals: Map<string, string>, editor: TextEditor, newState_: string) {
    const key = editorRebuildPendingKey(editor.document.uri.toString());
    updateState<EditingState>(globals, key, newState_, highlightStateTransitions, highlightIinitialStates);
}

function getEditingState(globals: Map<string, string>, editor: TextEditor): EditingState | boolean {
    return getState<EditingState>(globals, editorRebuildPendingKey(editor.document.uri.toString()));
}


function getOnDidChangeTextEditorSelectionAction(globals: Map<string, string>, output: OutputChannel, codelensProvider: CellCodelensProvider) {
    return async (event: vscode.TextEditorSelectionChangeEvent): Promise<void> => {
        let editor = window.activeTextEditor;
        if (event.textEditor &&
            editor &&
            event.textEditor.document === editor.document &&
            editor.selection.isEmpty && 
            getKernelState(globals, editor) == KernelState.extension_available) {
                let current_ranges = await getCurrentRangesFromPython(editor, output, globals, { 
                    rebuild: (getEditingState(globals, editor) !== EditingState.rebuilt) ? true : false });
                updateEditingState(globals, editor, EditingState.rebuilt);
                updateDecorations(editor, current_ranges ? current_ranges : []);
                let codelense_range = current_ranges ? current_ranges.filter((r) => (r.current && r.state != 'syntaxerror')).map((r) => r.range) : [];
                codelensProvider.change_range(codelense_range.length > 0 ? codelense_range[0] : undefined);
        } else {
            updateDecorations(event.textEditor, []);
            codelensProvider.change_range(undefined);
        }
    };
}


function getOnDidChangeTextDocumentAction(globals: Map<string, string>, output: OutputChannel): (e: vscode.TextDocumentChangeEvent) => any {
    return async (event) => {
            let editor = window.activeTextEditor;
            if (editor && event.document === editor.document) {
                if (getKernelState(globals, editor) !== KernelState.extension_available) { 
                    updateEditingState(globals, editor, EditingState.rebuildPending);
                } else {
                    const current_ranges = await getCurrentRangesFromPython(editor, output, globals, { rebuild: true });
                    updateEditingState(globals, editor, EditingState.rebuilt);
                    updateDecorations(editor, current_ranges ? current_ranges : []);
                }
            }
        };
}

function getOnDidChangeActiveTextEditorAction(globals: Map<string, string>, output: OutputChannel): (e: TextEditor | undefined) => any {
    return async (editor: TextEditor | undefined) => {
            if (editor) {
                if (getKernelState(globals, editor) !== KernelState.extension_available) { return; }
                const current_ranges = await getCurrentRangesFromPython(editor, output, globals, { rebuild: true });
                if (!current_ranges) return;
                await updateDecorations(editor, current_ranges);
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

    ///////// Codelens: ///////////////////////

    const codelensProvider = new CellCodelensProvider();
    languages.registerCodeLensProvider('python', codelensProvider);
    const initializingProvider = new InitialCodelensProvider();
    languages.registerCodeLensProvider('python', initializingProvider);

    ///////// Document Highlights: ///////////////////////

    workspace.onDidChangeTextDocument( getOnDidChangeTextDocumentAction(globals, output), null, Context.subscriptions );
    window.onDidChangeActiveTextEditor( getOnDidChangeActiveTextEditorAction(globals, output), null, Context.subscriptions );
    window.onDidChangeTextEditorSelection( getOnDidChangeTextEditorSelectionAction(globals, output, codelensProvider), null, Context.subscriptions );
}




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


