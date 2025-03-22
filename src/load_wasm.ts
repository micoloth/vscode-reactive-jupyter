// loadWasm.ts
import * as vscode from 'vscode';

export async function loadWasm(context: vscode.ExtensionContext) {
  // Set the public path manually before loading the WASM module.
  globalThis.__webpack_public_path__ = vscode.Uri.joinPath(context.extensionUri, 'wasm').toString() + '/';
  
  // Now perform the dynamic import.
  return await import('../wasm/wasm_vscode_reactive_jupyter.js');
}