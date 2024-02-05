# Jupyter Server Provider Sample

This is a very simple extension sample demonstrating the use of the Jupyter Extension API allowing other extensions to execute code against Jupyter Kernels.

- The sample lists finds kernels associated with notebooks that are currently open in the workspace.
- The sample the filters the kernels by language, focusing on Python kernels.
- Upon selecting a Python kernel, code selected by the user is executed against the selected kernel
- The output is displayed in an output panel.
- The sample demonstrates the ability to retrieve outputs of various mime types, including streamed output.

## Running this sample

 1. `cd jupyter-kernel-execution-sample`
 1. `code .`: Open the folder in VS Code
 1. Run `npm install` in terminal to install the dependencies
 1. Run the `Run Extension` target in the Debug View. This will:
	- Start a task `npm: watch` to compile the code
	- Run the extension in a new VS Code window
 1. Open a Jupyter Notebook and select a Python kernel and execute some code.
 1. Select the command `Jupyter Kernel API: Execute code against a Python Kernel`
 1. Select the a Kernel and then select the Code to execute.
 1. Watch the output panel for outputs returned by the kernel.

### Notes:

1. Make use of the `language` property of the kernel to ensure the language of the code matches the kernel.
2. `getKernel` API can can return `undefined` if the user does not grant the extension access to the kernel.
3. Access to kernels for each extension is managed via the command `Manage Access To Jupyter Kernels`.







# A Reactive Python Extension for Visual Studio Code
<p align="center">
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/fastapi-crudrouter">
</p>



An experimental [Visual Studio Code](https://code.visualstudio.com/) extension to add support for **Reactive Execution** of a Python script.

It is a fork of the [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) for Visual Studio Code.


# Demo:
https://github.com/micoloth/vscode-reactive-jupyter/assets/12880257/f363b91e-c8a3-450a-a185-45ee7d291978


# Installation:

This project is a fork of the official [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) for Visual Studio Code, since it relies heavily on capabilities developed there.

Because the development of that extension is tightly controlled by the [Microsoft](https://github.com/microsoft) team (see [here](https://github.com/microsoft/vscode-jupyter/discussions/13331), I won't go into the details), this extension cannot be published on the official [VSCode Marketplace](https://marketplace.visualstudio.com/).

You can use this extension by:
 -  downloading the published `.vsix` extension file 
 - Dropping it into the extensions folder of your VSCode installation, as described [here](https://code.visualstudio.com/docs/editor/extension-marketplace#_install-from-a-vsix) 
  - Uninstalling the original Jupyter Extension, if you have it installed, since this extension bundles the original one, and this would cause conflicts.

# Limitations

`vscode-reactive-jupyter` works by performing a simple Static Code Analysis of a Python script to find dependencies between the various statements.

Because of the very imperative nature of the Python language, it is impossible to reliably capture the effects of **functions that dynamically modify their arguments**.

This extension alwasy assumes that functions are **pure**, i.e. they don't have **side effects**.

This means that impure statements like `mylist.append(1)` or `model.train()` will not trigger the execution of the statements that depend on them.

As a workaround for this limitation, you can do 2 things:
 - Always wrap your impure statements into a function which returns the mutated object in the end, and reassign the variable. For example:

    ```python
    def append_to_list(mylist, item):
        mylist.append(item)
        return mylist
      
    mylist = append_to_list(mylist, 1)
    ```

 - When you perform an impure operation, always join it to a statement that reassigns the variable to itself, like this: `mylist = mylist`. This is free in Python, and will correctly propagate the dependency to the next statements.

This second methods requires joining several statements into a single execution "cell", that will always be executed as a single unit.

There are 2 ways of doing this in `vscode-reactive-jupyter`:

 - Simply put the statements on a single line. So, 
  
      ```python
      mylist.append(1); mylist = mylist
      ```

    will always be executed together, and will work as expected.
  
 - For longer/ more complex statements, you can use the special Cell Markers, `# % [` and `# % ]` to mark the beginning and the end of a cell: so, 
  
      ```python
      # % [
      mylist.append(1)
      mylist.append(2)
      mylist = mylist
      # % ]
      ```

    will be joined into a single cell, and will be propagated as expected.


# ###############

# TODO:

 - DONE (2h) Separate my function into a Class 
 - DONE (2h) Highlight Provider
 - CodeLens Provider
 - Read Python code from file
 - Understand how to inject python lines into command
 - Shortcut: Activate Interactive Window if not there already
 - Package Python code in a class + a State
 - Python util to compose Python commands into 1 executable comman (wth: debug prints, commands to update state)
 - Extract NetworkX
 - Understand how to Link Kernel to File (trace how CreateInteractive command is called by Jupyter)
 - Shortcut (ideally: should DISABLE the default shift enter)
 - Fix Python Doubleclass Bug
 - Fix at least 1 edge case
 - Import Python Package from a repo like vscode-reactive-python OR ReactivePython, i guess


-> QUESTION: Should i execute all dag nodes:
    -Silently?
    -All in the same interactive cells?
    -Or in separate cells? (this would be the best way to do it, but is it even possible?)


IMPORTANT:


in Package, should be 
    "main": "./out/extension.node.js",
    "browser": "./out/extension.web.bundle.js",

OR

	    "main": "./src/extension.node.js",

OR 

	"main": "./out/extension.js",  (This example has this) ?







# COMMANDS:


npm install -g @vscode/vsce

brew install zmq

npm install  # Cannot download zeromq binary from github release
npm ci  # idem
// And later, this fails in gulpfile.js > verifyZmqBinaries()


npx gulp prePublishNonBundle
npm run compile
npm run compile-webviews-watch 

npm run clean
npm run package # This step takes around 10 minutes.
Resulting in a ms-toolsai-jupyter-insiders.vsix file in your vscode-jupyter folder.

vsce package

code --install-extension my-extension-0.0.1.vsix.

Your extension folder
macOS ~/.vscode/extensions