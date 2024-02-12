
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
    NotebookCell,
    NotebookCellKind,
    NotebookCellData,
    NotebookEdit,
    WorkspaceEdit,
    Event, CodeLens, TextDocument
} from 'vscode';
import { Jupyter, Kernel, JupyterServerCommandProvider } from '@vscode/jupyter-extension';
import path = require('path');
import { TextDecoder } from 'util';

import {scriptCode} from './reactive_python_engine';

import * as vscode from 'vscode';
import {v4 as uuidv4} from 'uuid';

import { createDeferred, isPromise } from './platform/common/utils/async';
import { noop } from './platform/common/utils/misc';


const JVSC_EXTENSION_ID = 'ms-toolsai.jupyter';
type INativeInteractiveWindow = { notebookUri: Uri; inputUri: Uri; notebookEditor: NotebookEditor };
type InteractiveWindowMode = 'perFile' | 'single' | 'multiple';
const PYTHON_LANGUAGE = 'python';
const MARKDOWN_LANGUAGE = 'markdown';

export async function createEditor(
    preferredController: any,  // : IVSCodeNotebookController | undefined,  // CHANGE: This was originally IVSCodeNotebookController | undefined
    resource: Uri | undefined
): Promise<[Uri, NotebookEditor]> {
    // CHANGE: In general, this function was a bit longer...
    // CHANGE: TODO: Lock creation so that you don't create 2. The original getOrCreate did this...
    const controllerId = preferredController ? `${JVSC_EXTENSION_ID}/${preferredController.id}` : undefined;
    const setting = vscode.workspace.getConfiguration('jupyter').get<string>('interactiveWindow.viewColumn');
    const mode = vscode.workspace.getConfiguration('jupyter').get<string>('interactiveWindow.creationMode');
    const hasOwningFile = resource !== undefined;
    let viewColumn = (resource) ? ViewColumn.Beside : setting === 'secondGroup' ? ViewColumn.One : setting === 'active' ? ViewColumn.Active : ViewColumn.Beside;
    const { inputUri, notebookEditor } = (await commands.executeCommand(
        'interactive.open',
        { viewColumn: viewColumn, preserveFocus: hasOwningFile },  // Keep focus on the owning file if there is one
        undefined,
        controllerId,
        resource && mode === 'perFile' ? ('Interactive - ' + (path.basename(resource.path))) : undefined
        
    )) as unknown as INativeInteractiveWindow;
    if (!notebookEditor) {
        // This means VS Code failed to create an interactive window.
        // This should never happen.
        throw new Error('Failed to request creation of interactive window from VS Code.');
    }
    return [inputUri, notebookEditor];
}


function getRootFolder() {
    const firstWorkspace =
        Array.isArray(workspace.workspaceFolders) && workspace.workspaceFolders.length > 0
            ? workspace.workspaceFolders[0]
            : undefined;
    return firstWorkspace?.uri;
}



/**
 * Use this class to perform updates on all cells.
 * We cannot update cells in parallel, this could result in data loss.
 * E.g. assume we update execution order, while that's going on, assume we update the output (as output comes back from jupyter).
 * At this point, VSC is still updating the execution order & we then update the output.
 * Depending on the sequence its possible for some of the updates to get lost.
 *
 * Excellent example:
 * Assume we perform the following updates without awaiting on the promise.
 * Without awaiting, its very easy to replicate issues where the output is never displayed.
 * - We update execution count
 * - We update output
 * - We update status after completion
 */
const pendingCellUpdates = new WeakMap<NotebookDocument, Promise<unknown>>();

async function chainWithPendingUpdates(
    document: NotebookDocument,
    update: (edit: WorkspaceEdit) => void | Promise<void>
): Promise<boolean> {
    const notebook = document;
    if (document.isClosed) {
        return true;
    }
    const pendingUpdates = pendingCellUpdates.has(notebook) ? pendingCellUpdates.get(notebook)! : Promise.resolve();
    const deferred = createDeferred<boolean>();
    const aggregatedPromise = pendingUpdates
        // We need to ensure the update operation gets invoked after previous updates have been completed.
        // This way, the callback making references to cell metadata will have the latest information.
        // Even if previous update fails, we should not fail this current update.
        .finally(async () => {
            const edit = new WorkspaceEdit();
            const result = update(edit);
            if (isPromise(result)) {
                await result;
            }
            await workspace.applyEdit(edit).then(
                (result) => deferred.resolve(result),
                (ex) => deferred.reject(ex)
            );
        })
        .catch(noop);
    pendingCellUpdates.set(notebook, aggregatedPromise);
    return deferred.promise;
}

function clearPendingChainedUpdatesForTests() {
    const editor: NotebookEditor | undefined = window.activeNotebookEditor;
    if (editor?.notebook) {
        pendingCellUpdates.delete(editor.notebook);
    }
}


// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import { RegExpValues } from './platform/common/utils/typesAndConsts';
import { splitLines } from './platform/common/utils/helpers';
// import { IJupyterSettings } from './platform/common/types';


/**
 * CellMatcher is used to match either markdown or code cells using the regex's provided in the settings.
 */
class CellMatcher {
    public codeExecRegEx: RegExp;
    public markdownExecRegEx: RegExp;

    private codeMatchRegEx: RegExp;
    private markdownMatchRegEx: RegExp;
    private defaultCellMarker: string;

    constructor() {  // CHANGE:  (settings?: IJupyterSettings) {
        this.codeMatchRegEx = this.createRegExp(
            undefined,  // CHANGE: settings ? settings.codeRegularExpression : undefined,
            RegExpValues.PythonCellMarker
        );
        this.markdownMatchRegEx = this.createRegExp(
            undefined,  // CHANGE: settings ? settings.markdownRegularExpression : undefined,
            RegExpValues.PythonMarkdownCellMarker
        );
        this.codeExecRegEx = new RegExp(`${this.codeMatchRegEx.source}(.*)`);
        this.markdownExecRegEx = new RegExp(`${this.markdownMatchRegEx.source}(.*)`);
        this.defaultCellMarker = '# %%' // CHANGE: settings?.defaultCellMarker ? settings.defaultCellMarker : '# %%';
    }

    public isCell(code: string): boolean {
        return this.isCode(code) || this.isMarkdown(code);
    }

    public isMarkdown(code: string): boolean {
        return this.markdownMatchRegEx.test(code.trim());
    }

    public isCode(code: string): boolean {
        return this.codeMatchRegEx.test(code.trim()) || code.trim() === this.defaultCellMarker;
    }

    public getCellType(code: string): string {
        return this.isMarkdown(code) ? 'markdown' : 'code';
    }

    public isEmptyCell(code: string): boolean {
        return this.stripFirstMarker(code).trim().length === 0;
    }

    public stripFirstMarker(code: string): string {
        const lines = splitLines(code, { trim: false, removeEmptyEntries: false });

        // Only strip this off the first line. Otherwise we want the markers in the code.
        if (lines.length > 0 && (this.isCode(lines[0]) || this.isMarkdown(lines[0]))) {
            return lines.slice(1).join('\n');
        }
        return code;
    }

    public stripFirstMarkerNoConcat(lines: string[]): string[] {
        // Only strip this off the first line. Otherwise we want the markers in the code.
        if (lines.length > 0 && (this.isCode(lines[0]) || this.isMarkdown(lines[0]))) {
            return lines.slice(1);
        }
        return lines;
    }

    public getFirstMarker(code: string): string | undefined {
        const lines = splitLines(code, { trim: false, removeEmptyEntries: false });

        if (lines.length > 0 && (this.isCode(lines[0]) || this.isMarkdown(lines[0]))) {
            return lines[0];
        }
    }

    private createRegExp(potential: string | undefined, backup: RegExp): RegExp {
        try {
            if (potential) {
                return new RegExp(potential);
            }
        } catch {
            noop();
        }

        return backup;
    }
}



import { ICellRange, IDisposable } from './platform/common/utils/typesAndConsts';

// Wraps the vscode CodeLensProvider base class
const IDataScienceCodeLensProvider = Symbol('IDataScienceCodeLensProvider');
interface IDataScienceCodeLensProvider extends CodeLensProvider {
    getCodeWatcher(document: TextDocument): ICodeWatcher | undefined;
}

type CodeLensPerfMeasures = {
    totalCodeLensUpdateTimeInMs: number;
    codeLensUpdateCount: number;
    maxCellCount: number;
};

// Wraps the Code Watcher API
const ICodeWatcher = Symbol('ICodeWatcher');
interface ICodeWatcher extends IDisposable {
    readonly uri: Uri | undefined;
    codeLensUpdated: Event<void>;
    setDocument(document: TextDocument): void;
    getVersion(): number;
    getCodeLenses(): CodeLens[];
    runAllCells(): Promise<void>;
    runCell(range: Range): Promise<void>;
    debugCell(range: Range): Promise<void>;
    runCurrentCell(): Promise<void>;
    runCurrentCellAndAdvance(): Promise<void>;
    runSelectionOrLine(activeEditor: TextEditor | undefined, text: string | undefined): Promise<void>;
    runToLine(targetLine: number): Promise<void>;
    runFromLine(targetLine: number): Promise<void>;
    runAllCellsAbove(stopLine: number, stopCharacter: number): Promise<void>;
    runCellAndAllBelow(startLine: number, startCharacter: number): Promise<void>;
    runFileInteractive(): Promise<void>;
    debugFileInteractive(): Promise<void>;
    addEmptyCellToBottom(): Promise<void>;
    runCurrentCellAndAddBelow(): Promise<void>;
    insertCellBelowPosition(): void;
    insertCellBelow(): void;
    insertCellAbove(): void;
    deleteCells(): void;
    selectCell(): void;
    selectCellContents(): void;
    extendSelectionByCellAbove(): void;
    extendSelectionByCellBelow(): void;
    moveCellsUp(): Promise<void>;
    moveCellsDown(): Promise<void>;
    changeCellToMarkdown(): void;
    changeCellToCode(): void;
    debugCurrentCell(): Promise<void>;
    gotoNextCell(): void;
    gotoPreviousCell(): void;
}

const ICodeLensFactory = Symbol('ICodeLensFactory');
interface ICodeLensFactory {
    updateRequired: Event<void>;
    createCodeLenses(document: TextDocument): CodeLens[];
    getCellRanges(document: TextDocument): ICellRange[];
    getPerfMeasures(): CodeLensPerfMeasures;
}

interface IGeneratedCode {
    /**
     * 1 based, excluding the cell marker.
     */
    line: number;
    endLine: number; // 1 based and inclusive
    runtimeLine: number; // Line in the jupyter source to start at
    runtimeFile: string; // Name of the cell's file
    executionCount: number;
    id: string; // Cell id as sent to jupyter
    timestamp: number;
    code: string; // Code that was actually hashed (might include breakpoint and other code)
    debuggerStartLine: number; // 1 based line in source .py that we start our file mapping from
    startOffset: number;
    endOffset: number;
    deleted: boolean;
    realCode: string;
    trimmedRightCode: string;
    firstNonBlankLineIndex: number; // zero based. First non blank line of the real code.
    lineOffsetRelativeToIndexOfFirstLineInCell: number;
    hasCellMarker: boolean;
}

interface IFileGeneratedCodes {
    uri: Uri;
    generatedCodes: IGeneratedCode[];
}

const IGeneratedCodeStore = Symbol('IGeneratedCodeStore');
interface IGeneratedCodeStore {
    clear(): void;
    readonly all: IFileGeneratedCodes[];
    getFileGeneratedCode(fileUri: Uri): IGeneratedCode[];
    store(fileUri: Uri, info: IGeneratedCode): void;
}

const IGeneratedCodeStorageFactory = Symbol('IGeneratedCodeStorageFactory');
interface IGeneratedCodeStorageFactory {
    getOrCreate(notebook: NotebookDocument): IGeneratedCodeStore;
    get(options: { notebook: NotebookDocument } | { fileUri: Uri }): IGeneratedCodeStore | undefined;
}
type InteractiveCellMetadata = {
    interactiveWindowCellMarker?: string;
    interactive: {
        uristring: string;
        lineIndex: number;
        originalSource: string;
    };
    generatedCode?: IGeneratedCode;
    id: string;
};

interface IInteractiveWindowCodeGenerator extends IDisposable {
    reset(): void;
    generateCode(
        metadata: Pick<InteractiveCellMetadata, 'interactive' | 'id' | 'interactiveWindowCellMarker'>,
        cellIndex: number,
        debug: boolean,
        usingJupyterDebugProtocol?: boolean
    ): Promise<IGeneratedCode | undefined>;
}

const ICodeGeneratorFactory = Symbol('ICodeGeneratorFactory');
interface ICodeGeneratorFactory {
    getOrCreate(notebook: NotebookDocument): IInteractiveWindowCodeGenerator;
    get(notebook: NotebookDocument): IInteractiveWindowCodeGenerator | undefined;
}



export async function addNotebookCell(code: string, file: Uri, line: number, notebookDocument: NotebookDocument | undefined): Promise<NotebookCell> {
    // CHANGE: Original function didnt receive notebookDocument, ith got it from this.notebookDocument
    if (!notebookDocument) {
        throw new Error('No notebook document');
    }

    // CHANGE: These comments
    // Strip #%% and store it in the cell metadata so we can reconstruct the cell structure when exporting to Python files
    // const settings = this.configuration.getSettings(this.owningResource);
    // const isMarkdown = this.cellMatcher.getCellType(code) === MARKDOWN_LANGUAGE;
    // const strippedCode = isMarkdown
    //     ? generateMarkdownFromCodeLines(splitLines(code)).join('')
    //     : generateInteractiveCode(code, settings, this.cellMatcher);
    // const interactiveWindowCellMarker = this.cellMatcher.getFirstMarker(code);
    // TODO ^
    const strippedCode = code;
    const isMarkdown = false;
    const cellMatcher = new CellMatcher();  // CHANGE: This was originally this.cellMatcher = new CellMatcher(this.configuration.getSettings(this.owningResource));
    const interactiveWindowCellMarker = cellMatcher.getFirstMarker(code);

    // IDEA: this.owner is a property of InteractiveWindow, which receives it as undefined or this.document?.uri <<< (ie the editor's uri)

    // Insert cell into NotebookDocument
    const language =
        workspace.textDocuments.find((document) => document.uri.toString() === file?.toString())  // CHANGE: Originally it said this.owner?.toString()
            ?.languageId ?? PYTHON_LANGUAGE;
    const notebookCellData = new NotebookCellData(
        isMarkdown ? NotebookCellKind.Markup : NotebookCellKind.Code,
        strippedCode,
        isMarkdown ? MARKDOWN_LANGUAGE : language
    );
    const interactive = {
        uristring: file.toString(), // Has to be simple types
        lineIndex: line,
        originalSource: code
    };

    const metadata: InteractiveCellMetadata = {
        interactiveWindowCellMarker,
        interactive,
        id: uuidv4()
    };
    notebookCellData.metadata = metadata;
    await chainWithPendingUpdates(notebookDocument, (edit) => {
        const nbEdit = NotebookEdit.insertCells(notebookDocument.cellCount, [notebookCellData]);
        edit.set(notebookDocument.uri, [nbEdit]);
    });
    const newCellIndex = notebookDocument.cellCount - 1;
    return notebookDocument.cellAt(newCellIndex);
}