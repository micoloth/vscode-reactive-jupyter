
// enum State {
//     settings_not_ok = 'settings_not_ok',
//     initializable = 'initializable',
//     initializable_message_displayed = 'initializable_message_displayed',
//     initialization_started = 'initialization_started',
//     interactive_window_found = 'interactive_window_found',
//     kernel_found = 'kernel_found',
//     ready = 'ready',
//     got_kernel = 'got_kernel',
// }



// const editorToIWKey = (editorUri: string) => 'editorToIWKey' + editorUri;
// const editorConnectionStateKey = (editorUri: string) => 'state' + editorUri;


// const stateTransitions: Map<State, State[]> = new Map([
//     [State.settings_not_ok, [State.initializable]],
//     [State.initializable, [State.initialization_started, State.initializable_message_displayed, State.settings_not_ok]],
//     [State.initializable_message_displayed, [State.initialization_started, State.settings_not_ok]],
//     [State.initialization_started, [State.interactive_window_found, State.settings_not_ok]],
//     [State.interactive_window_found, [State.kernel_found, State.settings_not_ok, State.initialization_started]],
//     [State.kernel_found, [State.ready, State.settings_not_ok, State.initialization_started]],
//     [State.ready, [State.got_kernel, State.settings_not_ok, State.initializable]],
// ]);


// function updateState(globalState: Map<string, string>, editor: TextEditor, newState_: string ) {
//     let newState = newState_ as State;
//     if (!newState) {
//         throw new Error('Invalid state: ' + newState);
//     }
//     let uri = editor.document.uri.toString();
//     let currentState = globalState.get(editorConnectionStateKey(uri));
//     if (!currentState) {
//         if (newState === State.settings_not_ok) {
//             globalState.set(editorConnectionStateKey(uri), newState);
//             return;
//         }
//         else {
//             window.showErrorMessage('Invalid state transition: ' + currentState + ' -> ' + newState);
//             throw new Error('Invalid initial state: ' + newState + ' , please initialize your editor first');
//         }
//     }
//     let acceptedTransitions = stateTransitions.get(State[currentState as State]);
//     if (acceptedTransitions && acceptedTransitions.includes(newState)) {
//         globalState.set(editorConnectionStateKey(uri), newState);
//     }
//     else {
//         window.showErrorMessage('Invalid state transition: ' + currentState + ' -> ' + newState);
//         throw new Error('Invalid state transition: ' + currentState + ' -> ' + newState);
//     }
// }


// export const globalState: Map<string, string> = new Map();


// let currentState = 'settings_not_ok'

// // State as a Sate:

// let state = currentState as State;

// console.log(state);


// console.log(currentState === State.settings_not_ok)


