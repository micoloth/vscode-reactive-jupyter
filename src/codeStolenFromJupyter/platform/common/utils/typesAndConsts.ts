
export const NotebookCellScheme = 'vscode-notebook-cell';
import type { TextDocument, Uri, Range, ExtensionContext } from 'vscode';
import type * as nbformat from '@jupyterlab/nbformat';

export type EnvironmentVariables = Object & Record<string, string | undefined>;

export namespace RegExpValues {
    export const PythonCellMarker = /^(#\s*%%|#\s*\<codecell\>|#\s*In\[\d*?\]|#\s*In\[ \])/;
    export const PythonMarkdownCellMarker = /^(#\s*%%\s*\[markdown\]|#\s*\<markdowncell\>)/;
    export const UrlPatternRegEx =
        '(?<PREFIX>https?:\\/\\/)((\\(.+\\s+or\\s+(?<IP>.+)\\))|(?<LOCAL>[^\\s]+))(?<REST>:.+)';
    export const HttpPattern = /https?:\/\//;
    export const ShapeSplitterRegEx = /.*,\s*(\d+).*/;
    export const SvgHeightRegex = /(\<svg.*height=\")(.*?)\"/;
    export const SvgWidthRegex = /(\<svg.*width=\")(.*?)\"/;
    export const SvgSizeTagRegex = /\<svg.*tag=\"sizeTag=\{(.*),\s*(.*)\}\"/;
}

/**
 * The supported Python environment types.
 */
export enum EnvironmentType {
    Unknown = 'Unknown',
    Conda = 'Conda',
    VirtualEnv = 'VirtualEnv',
    Pipenv = 'PipEnv',
    Pyenv = 'Pyenv',
    Venv = 'Venv',
    Poetry = 'Poetry',
    VirtualEnvWrapper = 'VirtualEnvWrapper',
}
export type InterpreterId = string;
/**
 * Details about a Python runtime.
 *
 * @prop path - the location of the executable file
 * @prop version - the runtime version
 * @prop sysPrefix - the environment's install root (`sys.prefix`)
 */
export type InterpreterInformation = {
    id: InterpreterId;
    uri: Uri;
};

/**
 * Details about a Python environment.
 * @prop envType - the kind of Python environment
 */
export type PythonEnvironment = InterpreterInformation & {
    displayName?: string;
    envType?: EnvironmentType;
    envName?: string;
    /**
     * Directory of the Python environment.
     */
    envPath?: Uri;
    /**
     * This contains the path to the environment.
     * Used for display purposes only (in kernel picker or other places).
     */
    displayPath?: Uri;
    isCondaEnvWithoutPython?: boolean;
};


export type InterpreterUri = Resource | PythonEnvironment;
export type Resource = Uri | undefined
export interface IDisposable {
    dispose(): void | undefined;
}
export type IDisposableRegistry = IDisposable[];


export interface ICellRange {
    range: Range;
    cell_type: string;
}

export const IExtensionContext = Symbol('ExtensionContext');
export interface IExtensionContext extends ExtensionContext {}


export const WIDGET_STATE_MIMETYPE = 'application/vnd.jupyter.widget-state+json';
export const InteractiveWindowView = 'interactive';
export const JupyterNotebookView = 'jupyter-notebook';
export const jupyterLanguageToMonacoLanguageMapping = new Map([
    ['bash', 'shellscript'],
    ['c#', 'csharp'],
    ['f#', 'fsharp'],
    ['q#', 'qsharp'],
    ['c++11', 'c++'],
    ['c++12', 'c++'],
    ['c++14', 'c++']
]);



export interface ICell {
    uri?: Uri;
    data: nbformat.ICodeCell | nbformat.IRawCell | nbformat.IMarkdownCell;
}

export const PYTHON_LANGUAGE = 'python';


export namespace Identifiers {
    export const GeneratedThemeName = 'ipython-theme'; // This needs to be all lower class and a valid class name.
    export const MatplotLibDefaultParams = '_VSCode_defaultMatplotlib_Params';
    export const MatplotLibFigureFormats = '_VSCode_matplotLib_FigureFormats';
    export const DefaultCodeCellMarker = '# %%';
    export const DefaultCommTarget = 'jupyter.widget';
    export const ALL_VARIABLES = 'ALL_VARIABLES';
    export const KERNEL_VARIABLES = 'KERNEL_VARIABLES';
    export const DEBUGGER_VARIABLES = 'DEBUGGER_VARIABLES';
    export const PYTHON_VARIABLES_REQUESTER = 'PYTHON_VARIABLES_REQUESTER';
    export const MULTIPLEXING_DEBUGSERVICE = 'MULTIPLEXING_DEBUGSERVICE';
    export const RUN_BY_LINE_DEBUGSERVICE = 'RUN_BY_LINE_DEBUGSERVICE';
    export const REMOTE_URI = 'https://remote/';
    export const REMOTE_URI_ID_PARAM = 'id';
    export const REMOTE_URI_HANDLE_PARAM = 'uriHandle';
    export const REMOTE_URI_EXTENSION_ID_PARAM = 'extensionId';
}
export const WIDGET_MIMETYPE = 'application/vnd.jupyter.widget-view+json';
