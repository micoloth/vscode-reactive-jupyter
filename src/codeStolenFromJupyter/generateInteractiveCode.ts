
import { appendLineFeed, removeLinesFromFrontAndBackNoConcat } from './platform/common/utils/utils';
import { workspace } from 'vscode';
/**
 * Split a string using the cr and lf characters and return them as an array.
 * By default lines are trimmed and empty lines are removed.
 * @param {SplitLinesOptions=} splitOptions - Options used for splitting the string.
 */
export function splitLines(
    value: string,
    splitOptions: { trim: boolean; removeEmptyEntries?: boolean } = { removeEmptyEntries: true, trim: true }
): string[] {
    value = value || '';
    let lines = value.split(/\r?\n/g);
    if (splitOptions && splitOptions.trim) {
        lines = lines.map((line) => line.trim());
    }
    if (splitOptions && splitOptions.removeEmptyEntries) {
        lines = lines.filter((line) => line.length > 0);
    }
    return lines;
}

export function dedentCode(code: string) {
    const lines = code.split('\n');
    const firstNonEmptyLine = lines.find((line) => line.trim().length > 0 && !line.trim().startsWith('#'));
    if (firstNonEmptyLine) {
        const leadingSpaces = firstNonEmptyLine.match(/^\s*/)![0];
        return lines
            .map((line) => {
                if (line.startsWith(leadingSpaces)) {
                    return line.replace(leadingSpaces, '');
                }
                return line;
            })
            .join('\n');
    }
    return code;
}

export function uncommentMagicCommands(line: string): string {
    // Uncomment lines that are shell assignments (starting with #!),
    // line magic (starting with #!%) or cell magic (starting with #!%%).
    if (/^#\s*!/.test(line)) {
        // If the regex test passes, it's either line or cell magic.
        // Hence, remove the leading # and ! including possible white space.
        if (/^#\s*!\s*%%?/.test(line)) {
            return line.replace(/^#\s*!\s*/, '');
        }
        // If the test didn't pass, it's a shell assignment. In this case, only
        // remove leading # including possible white space.
        return line.replace(/^#\s*/, '');
    } else {
        // If it's regular Python code, just return it.
        return line;
    }
}


import { RegExpValues } from './platform/common/utils/typesAndConsts';
import { noop } from './platform/common/utils/misc';

interface IJupyterSettings {
    codeRegularExpression: string;
    markdownRegularExpression: string;
    defaultCellMarker: string;
}

/**
 * CellMatcher is used to match either markdown or code cells using the regex's provided in the settings.
 * 
 * TODO: Actually read the Settings for these data.
 */
export class CellMatcher {
    public codeExecRegEx: RegExp;
    public markdownExecRegEx: RegExp;

    private codeMatchRegEx: RegExp;
    private markdownMatchRegEx: RegExp;
    private defaultCellMarker: string;

    constructor(settings?: IJupyterSettings) {
        this.codeMatchRegEx = this.createRegExp(
            settings ? settings.codeRegularExpression : undefined,
            RegExpValues.PythonCellMarker
        );
        this.markdownMatchRegEx = this.createRegExp(
            settings ? settings.markdownRegularExpression : undefined,
            RegExpValues.PythonMarkdownCellMarker
        );
        this.codeExecRegEx = new RegExp(`${this.codeMatchRegEx.source}(.*)`);
        this.markdownExecRegEx = new RegExp(`${this.markdownMatchRegEx.source}(.*)`);
        this.defaultCellMarker = settings?.defaultCellMarker ? settings.defaultCellMarker : '# %%';
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



/**
 * Given a string representing Python code, return a processed
 * code string suitable for adding to a NotebookCell and executing.
 * @param code The code string text from a #%% code cell to be executed.
 */
export function generateInteractiveCode(code: string): string {
    let cellMatcher: CellMatcher = new CellMatcher();
    const lines = splitLines(code, { trim: false, removeEmptyEntries: false });

    // Remove the first marker
    const withoutFirstMarker = cellMatcher.stripFirstMarkerNoConcat(lines);
    // Skip leading and trailing lines
    const noLeadingOrTrailing = removeLinesFromFrontAndBackNoConcat(withoutFirstMarker);
    // Uncomment magics while adding linefeeds
    let magicCommandsAsComments = workspace.getConfiguration('jupyter').get<boolean>('interactiveWindow.textEditor.magicCommandsAsComments');
    const withMagicsAndLinefeeds = appendLineFeed(
        noLeadingOrTrailing,
        '\n',
        magicCommandsAsComments ? uncommentMagicCommands : undefined
    );

    return dedentCode(withMagicsAndLinefeeds.join(''));
}
