"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = void 0;
const vscode_1 = require("vscode");
const path = require("path");
const util_1 = require("util");
function activate(context) {
    const jupyterExt = vscode_1.extensions.getExtension('ms-toolsai.jupyter');
    if (!jupyterExt) {
        throw new Error('Jupyter Extension not installed');
    }
    if (!jupyterExt.isActive) {
        jupyterExt.activate();
    }
    const output = vscode_1.window.createOutputChannel('Jupyter Kernel Execution');
    context.subscriptions.push(output);
    context.subscriptions.push(vscode_1.commands.registerCommand('jupyterKernelExecution.listKernels', async () => {
        const kernel = await selectKernel();
        if (!kernel) {
            return;
        }
        const code = await selectCodeToRunAgainstKernel();
        if (!code) {
            return;
        }
        await executeCode(kernel, code, output);
    }));
}
exports.activate = activate;
const ErrorMimeType = vscode_1.NotebookCellOutputItem.error(new Error('')).mime;
const StdOutMimeType = vscode_1.NotebookCellOutputItem.stdout('').mime;
const StdErrMimeType = vscode_1.NotebookCellOutputItem.stderr('').mime;
const MarkdownMimeType = 'text/markdown';
const HtmlMimeType = 'text/html';
const textDecoder = new util_1.TextDecoder();
async function executeCode(kernel, code, logger) {
    logger.show();
    logger.appendLine(`>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>`);
    logger.appendLine(`Executing code against kernel ${code}`);
    const tokenSource = new vscode_1.CancellationTokenSource();
    try {
        for await (const output of kernel.executeCode(code, tokenSource.token)) {
            for (const outputItem of output.items) {
                if (outputItem.mime === ErrorMimeType) {
                    const error = JSON.parse(textDecoder.decode(outputItem.data));
                    logger.appendLine(`Error executing code ${error.name}: ${error.message},/n ${error.stack}`);
                }
                else {
                    logger.appendLine(`${outputItem.mime} Output: ${textDecoder.decode(outputItem.data)}`);
                }
            }
        }
        logger.appendLine('Code execution completed');
        logger.appendLine(`<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`);
    }
    catch (ex) {
        logger.appendLine(`Code execution failed with an error '${ex}'`);
        logger.appendLine(`<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`);
    }
    finally {
        tokenSource.dispose();
    }
}
const printHelloWorld = `print('Hello World')`;
const throwAnError = `raise Exception('Hello World')`;
const displayMarkdown = `from IPython.display import display, Markdown
display(Markdown('*some markdown*'))`;
const displayHtml = `from IPython.display import display, HTML
display(HTML('<div>Hello World</div>'))`;
const printToStdErr = `import sys
print('Hello World', file=sys.stderr)`;
const streamOutput = `import time
for i in range(10):
	print(i)
	time.sleep(1)`;
const doArithmetic = `print(12+13)`;
const codeSnippets = new Map([
    ['Print Hello World', printHelloWorld],
    ['Stream Output', streamOutput],
    ['Display Markdown', displayMarkdown],
    ['Display HTML', displayHtml],
    ['Print to StdErr', printToStdErr],
    ['Throw an Error', throwAnError],
    ['doArithmetic', doArithmetic],
]);
async function selectCodeToRunAgainstKernel() {
    const selection = await vscode_1.window.showQuickPick(Array.from(codeSnippets.keys()), {
        placeHolder: 'Select code to execute against the kernel',
    });
    if (!selection) {
        return;
    }
    return codeSnippets.get(selection);
}
async function selectKernel() {
    const extension = vscode_1.extensions.getExtension('ms-toolsai.jupyter');
    if (!extension) {
        throw new Error('Jupyter extension not installed');
    }
    await extension.activate();
    if (vscode_1.workspace.notebookDocuments.length === 0) {
        vscode_1.window.showErrorMessage('No notebooks open. Open a notebook, run a cell and then try this command');
        return;
    }
    const toDispose = [];
    return new Promise((resolve) => {
        const quickPick = vscode_1.window.createQuickPick();
        toDispose.push(quickPick);
        const quickPickItems = [];
        quickPick.title = 'Select a Kernel';
        quickPick.placeholder = 'Select a Python Kernel to execute some code';
        quickPick.busy = true;
        quickPick.show();
        const api = extension.exports;
        Promise.all(vscode_1.workspace.notebookDocuments.map(async (document) => {
            const kernel = await api.kernels.getKernel(document.uri);
            if (kernel && kernel.language === 'python') {
                quickPickItems.push({
                    label: `Kernel for ${path.basename(document.uri.fsPath)}`,
                    kernel,
                });
                quickPick.items = quickPickItems;
            }
        })).finally(() => {
            quickPick.busy = false;
            if (quickPickItems.length === 0) {
                quickPick.hide();
                vscode_1.window.showErrorMessage('No active kernels associated with any of the open notebooks, try opening a notebook and running a Python cell');
                return resolve(undefined);
            }
        });
        quickPick.onDidAccept(() => {
            quickPick.hide();
            if (quickPick.selectedItems.length > 0) {
                return resolve(quickPick.selectedItems[0].kernel);
            }
            resolve(undefined);
        }, undefined, toDispose);
        quickPick.onDidHide(() => resolve(undefined), undefined, toDispose);
    }).finally(() => vscode_1.Disposable.from(...toDispose).dispose());
}
//# sourceMappingURL=extension.js.map