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
    CodeLensProvider
} from 'vscode';
import { Jupyter, Kernel, JupyterServerCommandProvider } from '@vscode/jupyter-extension';
import path = require('path');
import { TextDecoder } from 'util';

import {scriptCode} from './reactive_python_engine';

import * as vscode from 'vscode';

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
    Currently, it ALWAYS logs a line (for the user), and it is NOT AN ERROR returs the result, else Undefined. 
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
    text: string | null,
    kernel: Kernel,
    output: OutputChannel,
    activeTextEditor: TextEditor
) {
    let resDel = await vscode.commands.executeCommand('jupyter.interactive.clearAllCells');
    console.log('---- >>>> RES DEL: ', resDel);
    if (!text) return;
    // console.log('>> TEXT: ', text);
    // let res = await getIWAndRunText(serviceManager, activeTextEditor, text);
    // let res = await executeCodeInKernel(text, kernel, output);

    console.log('---- >>>> CODE: ', text);
    let res = await vscode.commands.executeCommand('jupyter.execSelectionInteractive', text);
    // "command": "jupyter.execSelectionInteractive",  runDependentCells
    console.log('---- >>>> RES: ', res);
    // console.log('>> RESULT: ', res);
    // Check if res is an Error:
    if (res === undefined) {  // TODO I'm not doing that at all....
        return;
    }
}



// USEFUL:                             
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




async function queueComputation(
    current_ranges: AnnotatedRange[] | undefined,
    kernel: Kernel,
    output: OutputChannel,
    activeTextEditor: TextEditor
) {
    // console.log('>> CURRENT RANGES::::: ' + current_ranges);
    if (current_ranges) {
        // let resDel = await vscode.commands.executeCommand('jupyter.interactive.clearAllCells');
        // console.log('---- >>>> RES DEL: ', resDel);
        for (let range of current_ranges) {
            if (!range.text) break;
            // console.log('>> TEXT: ', range.text);
            
            if (!range.has_children) {  // TODOTODO: CHange this !!
                let res = await executeCodeInKernel(range.text, kernel, output);
                if (res === undefined) {
                    vscode.window.showErrorMessage("Failed to execute the range's state in Python: " + range.hash + " -- " + res);
                    break;
                }
            }
            else {
                let res = await executeCodeInInteractiveWindow(range.text, kernel, output, activeTextEditor);
            }
            
            // console.log('>> updateRange_command: ', updateRange_command);
            const update_result = await executeCodeInKernel(getSyncRangeCommand(range), kernel, null);
            if (!update_result) {
                vscode.window.showErrorMessage("Failed to update the range's state in Python: " + range.hash + " -- " + update_result);
                break;
            }
            // Trigger a onDidChangeTextEditorSelection event:
            const refreshed_ranges = await getCurrentRangesFromPython(activeTextEditor, kernel, output, {
                rebuild: false
            });
            if (refreshed_ranges) {
                updateDecorations(activeTextEditor, refreshed_ranges);
            }
        }
    }
    let unlockCommand = getUnlockCommand();
    const update_result = await executeCodeInKernel(unlockCommand, kernel, null);
    if (!update_result) {
        vscode.window.showErrorMessage('Failed to unlock the Python kernel: ' + update_result);
    }
    else{
        console.log('>> unlockCommand successful: ', update_result);
    }
}



async function selectKernelOnce(): Promise<Kernel | undefined> {

    // NEW SYSTEM !!!
    
    // TODO: Understand if you can Start a new Kernel .....
    // const tokenSource = new CancellationTokenSource();
    // const extensionCommands = extensions.getExtension<JupyterServerCommandProvider>('ms-toolsai.jupyter');
    // // console.log("HELLOOOOOO");
    // // console.log(extensionCommands?.exports);
    // THere is even a openNotebook:() in here....
    // if (extensionCommands)
    // {
    //     let ff = await extensionCommands.exports.provideCommands;
    // //     console.log(ff);
    // //     console.log(ff("", tokenSource.token));
    // }

    // TODO: Use LocalStorage to associate a kernel to the Documents, 
    // and then return the Krnel of the Currently active document

	const extension = extensions.getExtension<Jupyter>('ms-toolsai.jupyter');
	if (!extension) {
        throw new Error('Jupyter extension not installed');
	}
	await extension.activate();

	if (workspace.notebookDocuments.length === 0) {
		window.showErrorMessage(
			'No notebooks open. Open a notebook, run a cell and then try this command'
		);
		return;
	}
	const toDispose: Disposable[] = [];

	return new Promise<Kernel | undefined>(async (resolve) => {
        
		const api = extension.exports;
        
        const kernelDocumentPairs: [Kernel, NotebookDocument][] = [];
		await Promise.all(
            workspace.notebookDocuments.map(async (document) => {
            const kernel = await api.kernels.getKernel(document.uri);
            if (kernel && (kernel as any).language === 'python') { 
                kernelDocumentPairs.push([kernel, document]);
            }
        }));
        // console.log(kernelDocumentPairs);  // This was actually interesting ...

        if (kernelDocumentPairs.length === 1) {
            return resolve(kernelDocumentPairs[0][0]);
        } else if (kernelDocumentPairs.length === 0) {
            window.showErrorMessage(
                'No active kernels found'
            );
            return resolve(undefined);
        }

		const quickPick = window.createQuickPick<QuickPickItem & { kernel: Kernel }>();
		toDispose.push(quickPick);
		const quickPickItems: (QuickPickItem & { kernel: Kernel })[] = [];
		quickPick.title = 'Select a Kernel';
		quickPick.placeholder = 'Select a Python Kernel to execute some code';
		quickPick.busy = true;
		quickPick.show();

		Promise.all(
			kernelDocumentPairs.map(async ([kernel, document]) => {
					quickPickItems.push({
						label: `Kernel for ${path.basename(document.uri.fsPath)}`,
						kernel,
					});
					quickPick.items = quickPickItems;
				}
			)
		).finally(() => {
			quickPick.busy = false;
			if (quickPickItems.length === 0) {
				quickPick.hide();
				window.showErrorMessage(
					'No active kernels associated with any of the open notebooks, try opening a notebook and running a Python cell'
				);
				return resolve(undefined);
			}
		});

		quickPick.onDidAccept(
			() => {
				quickPick.hide();
				if (quickPick.selectedItems.length > 0) {
					return resolve(quickPick.selectedItems[0].kernel);
				}
				resolve(undefined);
			},
			undefined,
			toDispose
		);
		quickPick.onDidHide(() => resolve(undefined), undefined, toDispose);
	}).finally(() => Disposable.from(...toDispose).dispose());

}


async function selectKernel(): Promise<Kernel | undefined>  {
    /*
    If Kernel is undefined, starts a new one by calling command 'jupyter.createnewinteractive'
    */
    let kernel = await selectKernelOnce();
    // console.log('>> kernel >> : ', kernel);
    for (let i = 0; i < 4; i++) {
        if (kernel === undefined) {
            vscode.window.showInformationMessage('Reactive Python: Starting Python Kernel: attempt ' + (i + 1));

            for (let j = 0; j < 3; j++) {
                await new Promise((resolve) => setTimeout(resolve, 3000));
                kernel = await selectKernelOnce();
                if (kernel !== undefined) {
                    vscode.window.showInformationMessage('Reactive Python: Successfully started Python Kernel!');
                    break;
                }
            }
        } else {
            return kernel;
        }
    }
    vscode.window.showInformationMessage('Reactive Python: Failed to start Python Kernel with the Jupiter extension.');
    throw new Error('Could not get kernel');
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// PYTHON COMMANDS AND SNIPPETS
////////////////////////////////////////////////////////////////////////////////////////////////////



/* Read script content from file: */
async function generateCodeToGetVariableTypes(): Promise<string> {
    return scriptCode;

}

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
    kernel: Kernel | undefined = undefined,
    output: OutputChannel,
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
    if (!kernel) {
        kernel = await selectKernel();  // TODO Use editor
    }
    if (!kernel) return;
    // console.log('3: Here we are: ', command)
    const result = await executeCodeInKernel(command, kernel, null);
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
        // console.log('>>>>>>>>>>>>>>>>>>NICE, INVOKED ONCE. ' + editor);
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



export function createPreparePythonEnvForReactivePython(output: OutputChannel) {
    async function preparePythonEnvForReactivePython() {
        /* IDEA:
        we DON'T want to run this on Activation, but WE ARE DOING IT FOR NOW. for easier testing.
        */
        let command = scriptCode + '\n\n\n"Reactive Python Activated"\n';
        
        if (vscode.workspace.notebookDocuments.length < 1)
        {
            let resIW = await vscode.commands.executeCommand('jupyter.createnewinteractive');
            console.log('>> resIW: ', resIW);
        }
        let resWelcome = await vscode.commands.executeCommand('jupyter.execSelectionInteractive', "# Welcome to Reactive Python");
        console.log('>> resWelcome: ', resWelcome);
        let kernel = await selectKernel();
        if (!kernel) return;
        const result = executeCodeInKernel(command, kernel, output);
        if (result !== undefined) {
            vscode.window.showInformationMessage('Result: ' + result);
            if (window.activeTextEditor) {
                let refreshed_ranges = await getCurrentRangesFromPython(window.activeTextEditor, kernel, output, {
                    rebuild: true,
                    current_line: null
                }); // Do this in order to immediatly recompute the dag in the python kernel
                if (refreshed_ranges) {
                    updateDecorations(window.activeTextEditor, refreshed_ranges);
                }
            }
        }
        // if (vscode.window.activeTextEditor) {
        //     await updateDecorations(vscode.window.activeTextEditor, serviceManager);
        //     updateDecorations(vscode.window.activeTextEditor);
        // }
    }
    return preparePythonEnvForReactivePython;
}

export function createComputeAllDownstreamAction(output: OutputChannel) {
    async function computeAllDownstreamAction() {
        if (!window.activeTextEditor) return;
        let kernel = await selectKernel();
        if (!kernel) return;
        const current_ranges = await getCurrentRangesFromPython(
            window.activeTextEditor,
            kernel, output,
            {
                rebuild: true,
                upstream: false,
                downstream: true,
                stale_only: true,
                to_launch_compute: true
            },
        );
        await queueComputation(current_ranges, kernel, output, window.activeTextEditor);
    }
    return computeAllDownstreamAction;
}
export function createComputeAllUpstreamAction(output: OutputChannel) {
    async function computeAllUpstreamAction() {
        if (!window.activeTextEditor) return;
        let kernel = await selectKernel();
        if (!kernel) return;
        const current_ranges = await getCurrentRangesFromPython(
            window.activeTextEditor,
            kernel, output,
            {
                rebuild: true,
                upstream: true,
                downstream: false,
                stale_only: true,
                to_launch_compute: true
            }
        );
        await queueComputation(current_ranges, kernel, output, window.activeTextEditor);
    }
    return computeAllUpstreamAction;
}
export function createComputeAllDownstreamAndUpstreamAction(output: OutputChannel) {
    async function computeAllAction() {
        if (!window.activeTextEditor) return;
        let kernel = await selectKernel();
        if (!kernel) return;
        const current_ranges = await getCurrentRangesFromPython(
            window.activeTextEditor,
            kernel, output,
            {
                rebuild: true,
                upstream: true,
                downstream: true,
                stale_only: true,
                to_launch_compute: true
            }
        );
        await queueComputation(current_ranges, kernel, output, window.activeTextEditor);
    }
    return computeAllAction;
}

export function createComputeAllAction(output: OutputChannel) {
    async function computeAllAction() {
        if (!window.activeTextEditor) return;
        let kernel = await selectKernel();
        if (!kernel) return;
        const current_ranges = await getCurrentRangesFromPython(
            window.activeTextEditor,
            kernel, output,
            {
                rebuild: true,
                current_line: null,
                upstream: true,
                downstream: true,
                stale_only: true,
                to_launch_compute: true
            }
        );
        await queueComputation(current_ranges, kernel, output, window.activeTextEditor);
    }
    return computeAllAction;
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

export function activate(context: ExtensionContext) {
	const jupyterExt = extensions.getExtension<Jupyter>('ms-toolsai.jupyter');
	if (!jupyterExt) {
		throw new Error('Jupyter Extension not installed');
	}
	if (!jupyterExt.isActive) {
		jupyterExt.activate();
	}
	const output = window.createOutputChannel('Jupyter Kernel Execution');
	context.subscriptions.push(output);


    defineAllCommands(context, output);
}




export async function defineAllCommands(context: ExtensionContext, output: OutputChannel
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
            createPreparePythonEnvForReactivePython(output)
        )
    );
    context.subscriptions.push(
        vscode.commands.registerCommand('jupyter.sync-upstream', createComputeAllUpstreamAction(output))
    );
    context.subscriptions.push(
        vscode.commands.registerCommand('jupyter.sync-downstream', createComputeAllDownstreamAction(output))
    );
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'jupyter.sync-upstream-and-downstream',
            createComputeAllDownstreamAndUpstreamAction(output)
        )
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('jupyter.sync-all', createComputeAllAction(output))
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
                let kernel = await selectKernel();
                if (!kernel) return;
                const current_ranges = await getCurrentRangesFromPython(editor, kernel, output, { rebuild: true });
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
            if (window.activeTextEditor && event.document === window.activeTextEditor.document) {
                let kernel = await selectKernel();
                if (!kernel) return;
                const current_ranges = await getCurrentRangesFromPython(window.activeTextEditor, kernel, output, {
                    rebuild: true
                });
                if (!current_ranges) return;
                await updateDecorations(window.activeTextEditor, current_ranges);
            }
        },
        null,
        Context.subscriptions
    );
    window.onDidChangeTextEditorSelection(
        async (event) => {
            // console.log('----- Here 1! ');
            if (
                event.textEditor &&
                window.activeTextEditor &&
                event.textEditor.document === window.activeTextEditor.document &&
                window.activeTextEditor.selection.isEmpty
            ) {
                // console.log('----- Here 2! ');
                let kernel = await selectKernel();
                if (!kernel) return;
                const current_ranges = await getCurrentRangesFromPython(window.activeTextEditor, kernel, output, {
                    rebuild: false
                });
                // console.log('----- Here 3!, current_ranges: ', current_ranges);
                if (current_ranges) {
                    updateDecorations(window.activeTextEditor, current_ranges);
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