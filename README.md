
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