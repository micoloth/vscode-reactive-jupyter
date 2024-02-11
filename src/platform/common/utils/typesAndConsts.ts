
export const NotebookCellScheme = 'vscode-notebook-cell';
import type { TextDocument, Uri, Range } from 'vscode';

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

export interface ICellRange {
    range: Range;
    cell_type: string;
}