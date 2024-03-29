



<h1 style="text-align: center;">A Reactive Jupyter Extension for Visual Studio Code</h1>


<p align="center">
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/networkx">
</p>



An experimental [Visual Studio Code](https://code.visualstudio.com/) extension to add support for **Reactive Execution** of a Python script.


This extension performs simple Static Code Analysis to find dependencies between the various statements in a Python script. When you modify a line, it will be marked as Stale together with all the lines that depend on it.

This extension is in **PREVIEW**. Please don't expect it to run flawlessly. Also, read about the limitations below.


# Demo:


![demo](https://github.com/micoloth/vscode-reactive-jupyter/assets/12880257/0e713fd5-ea46-498e-866f-f5c5aa18658b)


# Activation:

  1. Install the extension. The [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extension should be installed as well.

  2. Add these settings for the Jupyter extension in your `settings.json` file: these are required to work with Reactive Jupyter. If they are set differently, you will be prompted to change them.

    "jupyter.interactiveWindow.creationMode": "perFile",
    "jupyter.interactiveWindow.textEditor.executeSelection": false
    

  3. Open a Python file.
  
  4. Initialize Reactive Jupyter by clicking the "Initialize Reactive Jupyter" codelens at the top of the editor, or launch the `Initialize Reactive Jupyter` command from the command palette.  This will start a new Jupyter Kernel in an interactive window.

  5. The first time, you will be asked to trust this extension running Python code in your Kernel. It is completely safe to do so.

  6. You are now ready to run your code.

# Shortcuts:

You can execute code statements using the CodeLenses that appear over the code, or with these shortcuts:

  -  `shift+ctrl+enter` (Windows/Linux) or `shift+cmd+enter` (MacOS): Sync all the stale code blocks in the file.
  - `shift+ctrl+up shift+ctrl+enter` (Windows/Linux) or `shift+cmd+up shift+cmd+enter` (MacOS): Sync the current code block and all the code blocks the current code blocks depends on. (i.e. all the Upstream code)
  - `shift+ctrl+down shift+ctrl+enter` (Windows/Linux) or `shift+cmd+down shift+cmd+enter` (MacOS): Sync the current code block and all the code blocks that depend on it (i.e. all the Downstream code)
  - `shift+enter`: Sync the current code block, if all the upstream code is already synced.


# Limitations:

Currently, `reactive-jupyter` works by performing simple Static Code Analysis of a Python script to find dependencies between the various statements.

Because of the very imperative nature of the Python language, it's impossible to capture the effects of **impure** statements, i.e. statements that **modify a variable in place**.

These are ***NOT*** handled by this extension, and will not trigger the execution of the statements that depend on them.

By default, try to write your script in a Functional style:
 - Don't reassign variables with the same name
 - Only use **pure** functions.

This makes your code easier to reason about, for humans as well as for computers.

Still, some impure statements, like `mylist.append(x)` or `model.train()`, are inevitable. 

As a workaround for this limitation, you can do 2 things:
 - Always wrap your impure statements into a function which returns the mutated object in the end, and reassign the variable. For example:

    ```python
    def append_to_list(mylist, item):
        mylist.append(item)
        return mylist
      
    mylist = append_to_list(mylist, 1)
    ```

 - Or, when you perform an impure operation, always join it to a statement that reassigns the variable to itself, like this: `mylist = mylist`. This is free in Python, and will correctly propagate the dependency to the next statements.

This second methods requires joining several statements into a single execution "block" or "cell", that will always be executed as a single unit.

There are 2 ways of doing this in `reactive-jupyter`:

 - Simply put the statements on a single line. So, 
  
      ```python
      mylist.append(1); mylist = mylist
      ```

    will always be executed together, and will work as expected.
  
 - For longer/ more complex statements, you can use the special Cell Markers, `# % [` and `# % ]` to mark the beginning and the end of a block: so, 
  
      ```python
      # % [
      mylist.append(1)
      mylist.append(2)
      mylist = mylist
      # % ]
      ```

    will be joined into a single block, and will be propagated as expected.


# Advanced Usage:

If you want to jump right into the action with a single click, these commands are also exposed:

 - `reactive-jupyter.initialize-and-sync-downstream`
 - `reactive-jupyter.initialize-and-sync-upstream`
 - `reactive-jupyter.initialize-and-sync-upstream-and-downstream`
 - `reactive-jupyter.initialize-and-sync-all`
 - `reactive-jupyter.initialize-and-sync-current`

These will initialize the extension if it was inactive, and immediatly start running code, hence saving you a step. Feel free to bind them to your favorite shortcuts.



Furthermore: if after a while the Codelenses start getting annoying, you can disable them with the `reactive-jupyter.showCodelenses` setting.
