// loadWasm.ts
import * as vscode from 'vscode';

export async function loadWasm(context: vscode.ExtensionContext) {
  // The bundler target auto-initializes the WASM when imported
  const wasmModule = await import('../wasm-vscode-reactive-jupyter/pkg/wasm_vscode_reactive_jupyter.js');
  return wasmModule;
}